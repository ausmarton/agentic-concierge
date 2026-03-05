"""Specialist registry: resolve pack by id from config.

Pack selection order for a given specialist_id:
1. If ``tools`` list is provided, build a dynamic pack via ``build_dynamic_pack()``.
2. If the specialist_id matches a ``PACK_TEMPLATES`` entry, build from the template.
3. If ``SpecialistConfig.builder`` is set, dynamically import and call that factory.
4. Raise ``ValueError`` if none of the above apply.

Adding a new pack without editing this file:
- Set ``builder: "mypackage.packs.custom:build_custom_pack"`` in your YAML config.
- The factory must have signature ``(workspace_path: str, network_allowed: bool) -> SpecialistPack``.
"""

from __future__ import annotations

import importlib
import logging
from typing import Callable, List, Optional

from agentic_concierge.config import ConciergeConfig
from agentic_concierge.application.ports import SpecialistPack, SpecialistRegistry
from agentic_concierge.config.features import FeatureSet, ProfileTier

from .dynamic_pack import PACK_TEMPLATES, build_dynamic_pack, build_template_pack

logger = logging.getLogger(__name__)


def _load_builder(dotted_path: str) -> Callable[[str, bool], SpecialistPack]:
    """Import and return a pack factory from a dotted path (``'module.path:func_name'``)."""
    if ":" not in dotted_path:
        raise ValueError(
            f"Invalid builder path {dotted_path!r}: expected 'module.path:function_name'"
        )
    module_path, func_name = dotted_path.rsplit(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Cannot import builder module {module_path!r}: {exc}"
        ) from exc
    try:
        func = getattr(module, func_name)
    except AttributeError as exc:
        raise ImportError(
            f"Module {module_path!r} has no attribute {func_name!r}"
        ) from exc
    return func


class ConfigSpecialistRegistry(SpecialistRegistry):
    """Resolve specialist pack by id; only specialists declared in config are available.

    Supports three resolution paths:
    1. **Dynamic packs**: when ``tools`` and ``role`` are provided (from orchestrator).
    2. **Template packs**: when specialist_id matches a known template.
    3. **Custom builders**: when ``SpecialistConfig.builder`` is set in config.
    """

    def __init__(self, config: ConciergeConfig):
        self._config = config
        self._feature_set = self._load_feature_set()

    def _load_feature_set(self) -> FeatureSet:
        """Detect the profile tier once and build the feature set."""
        try:
            from agentic_concierge.bootstrap.detected import load_detected
            detected = load_detected()
            tier = detected.tier if detected is not None else ProfileTier.SMALL
        except Exception:
            tier = ProfileTier.SMALL
        return FeatureSet.from_profile(tier, self._config.features)

    def get_pack(
        self,
        specialist_id: str,
        workspace_path: str,
        network_allowed: bool,
        *,
        tools: Optional[List[str]] = None,
        role: Optional[str] = None,
    ) -> SpecialistPack:
        # Dynamic pack: tools explicitly provided (from orchestrator)
        if tools is not None:
            logger.debug(
                "Building dynamic pack for %r with tools=%s", specialist_id, tools
            )
            pack = build_dynamic_pack(
                specialist_id=specialist_id,
                tool_names=tools,
                role_description=role or f"You are a specialist agent (id={specialist_id}).",
                workspace_path=workspace_path,
                network_allowed=network_allowed,
            )
            if hasattr(pack, "set_feature_set"):
                pack.set_feature_set(self._feature_set)
            return self._wrap_pack(specialist_id, pack)

        # Template or config-based pack
        if specialist_id not in self._config.specialists:
            raise ValueError(f"Unknown specialist: {specialist_id!r}")

        spec_cfg = self._config.specialists[specialist_id]

        if spec_cfg.builder:
            logger.debug(
                "Loading custom builder for %r: %s", specialist_id, spec_cfg.builder
            )
            builder = _load_builder(spec_cfg.builder)
            pack = builder(workspace_path, network_allowed)
        elif specialist_id in PACK_TEMPLATES:
            pack = build_template_pack(specialist_id, workspace_path, network_allowed)
        else:
            raise ValueError(
                f"No pack implementation for specialist {specialist_id!r}. "
                "Set 'builder' in config to point at a pack factory function."
            )

        if hasattr(pack, "set_feature_set"):
            pack.set_feature_set(self._feature_set)

        return self._wrap_pack(specialist_id, pack)

    def _wrap_pack(self, specialist_id: str, pack: SpecialistPack) -> SpecialistPack:
        """Apply MCP and container wrapping if configured."""
        spec_cfg = self._config.specialists.get(specialist_id)
        if spec_cfg is None:
            return pack

        if spec_cfg.mcp_servers:
            try:
                from agentic_concierge.infrastructure.mcp import MCPAugmentedPack, MCPSessionManager
            except ImportError as exc:
                raise RuntimeError(
                    "mcp_servers configured but 'mcp' package is not installed. "
                    "Install with: pip install agentic-concierge[mcp]"
                ) from exc
            sessions = [MCPSessionManager(s) for s in spec_cfg.mcp_servers]
            pack = MCPAugmentedPack(pack, sessions)
            logger.debug(
                "Wrapped pack %r with MCPAugmentedPack (%d server(s))",
                specialist_id, len(sessions),
            )

        if spec_cfg.container_image:
            from agentic_concierge.infrastructure.specialists.containerised import (
                ContainerisedSpecialistPack,
            )
            workspace_path = getattr(pack, "_workspace_path", "")
            pack = ContainerisedSpecialistPack(pack, spec_cfg.container_image, workspace_path)
            logger.debug(
                "Wrapped pack %r with ContainerisedSpecialistPack (image=%r)",
                specialist_id, spec_cfg.container_image,
            )

        return pack

    def list_ids(self) -> List[str]:
        return list(self._config.specialists.keys())
