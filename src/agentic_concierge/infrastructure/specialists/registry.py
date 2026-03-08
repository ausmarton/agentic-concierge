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
        self._all_chat_models: List[str] = []

    def _load_feature_set(self) -> FeatureSet:
        """Detect the profile tier once and build the feature set."""
        try:
            from agentic_concierge.bootstrap.detected import load_detected
            detected = load_detected()
            tier = detected.tier if detected is not None else ProfileTier.SMALL
        except Exception:
            tier = ProfileTier.SMALL
        return FeatureSet.from_profile(tier, self._config.features)

    def set_runtime_models(self, all_chat_models: List[str]) -> None:
        """Set the full list of chat-capable models (including non-tool-calling).

        Called once after LLM discovery so the registry can inject
        ``consult_specialist_model`` when non-tool-calling specialists are
        available.
        """
        self._all_chat_models = list(all_chat_models)

    def _needs_consult_tool(self) -> bool:
        """Check whether any discovered model lacks tool-calling support."""
        if not self._all_chat_models:
            return False
        from agentic_concierge.infrastructure.model_profiles import get_profile
        return any(
            not get_profile(m).supports_tool_calling
            for m in self._all_chat_models
        )

    def _llm_kwargs(self) -> dict:
        """Extract LLM connection params from config for consult tool executors."""
        # Use the first available model config for base_url/backend/api_key.
        for cfg in self._config.models.values():
            return {
                "all_chat_models": self._all_chat_models,
                "base_url": cfg.base_url,
                "backend": cfg.backend,
                "api_key": cfg.api_key,
            }
        return {
            "all_chat_models": self._all_chat_models,
            "base_url": "http://localhost:11434/v1",
            "backend": "ollama",
            "api_key": "",
        }

    def _maybe_add_consult(self, tool_names: List[str]) -> List[str]:
        """Return tool_names with consult_specialist_model appended if needed."""
        if self._needs_consult_tool() and "consult_specialist_model" not in tool_names:
            return tool_names + ["consult_specialist_model"]
        return tool_names

    def get_pack(
        self,
        specialist_id: str,
        workspace_path: str,
        network_allowed: bool,
        *,
        tools: Optional[List[str]] = None,
        role: Optional[str] = None,
        finish_schema_key: Optional[str] = None,
    ) -> SpecialistPack:
        llm_kw = self._llm_kwargs()

        # Dynamic pack: tools explicitly provided (from orchestrator)
        if tools is not None:
            effective_tools = self._maybe_add_consult(list(tools))
            logger.debug(
                "Building dynamic pack for %r with tools=%s", specialist_id, effective_tools
            )
            pack = build_dynamic_pack(
                specialist_id=specialist_id,
                tool_names=effective_tools,
                role_description=role or f"You are a specialist agent (id={specialist_id}).",
                workspace_path=workspace_path,
                network_allowed=network_allowed,
                finish_schema_key=finish_schema_key,
                **llm_kw,
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
            tpl = PACK_TEMPLATES[specialist_id]
            if self._needs_consult_tool():
                tpl_tools = self._maybe_add_consult(list(tpl.tool_names))
                pack = build_dynamic_pack(
                    specialist_id=specialist_id,
                    tool_names=tpl_tools,
                    role_description=tpl.role_description,
                    workspace_path=workspace_path,
                    network_allowed=network_allowed,
                    finish_schema=tpl.finish_schema,
                    quality_gates=tpl.quality_gates,
                    finish_schema_key=finish_schema_key,
                    **llm_kw,
                )
            else:
                pack = build_template_pack(
                    specialist_id, workspace_path, network_allowed,
                    finish_schema_key=finish_schema_key,
                )
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
