"""Dynamic pack builder: construct specialist packs from tool selections + role descriptions.

``build_dynamic_pack()`` is the central entry point for constructing packs at
runtime.  It looks up tools in the :mod:`tool_catalog`, creates executors bound
to the workspace, and assembles a system prompt from the role description.

Templates are loaded from YAML files in ``config/defaults/templates/`` with
user override support via ``~/.config/concierge/templates/``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from agentic_concierge.infrastructure.tools.sandbox import SandboxPolicy

from .base import BaseSpecialistPack
from .finish_schemas import (
    FINISH_SCHEMAS,
    get_finish_schema,
)
from .prompts import (
    ROLE_DESCRIPTIONS,
    generate_system_prompt,
)
from .tool_catalog import TOOL_CATALOG, get_tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pack templates — loaded from YAML
# ---------------------------------------------------------------------------

@dataclass
class PackTemplate:
    """Data-driven pack configuration template."""

    template_id: str
    tool_names: List[str]
    role_description: str
    finish_schema: Dict[str, Any]
    quality_gates: List[str] = field(default_factory=list)


# Hardcoded fallback — used only when YAML cannot be loaded.
def _fallback_templates() -> Dict[str, PackTemplate]:
    return {
        "engineering": PackTemplate(
            template_id="engineering",
            tool_names=["shell", "read_file", "write_file", "list_files", "run_tests"],
            role_description=ROLE_DESCRIPTIONS["engineering"],
            finish_schema=FINISH_SCHEMAS["code"],
            quality_gates=["tests_verified"],
        ),
        "research": PackTemplate(
            template_id="research",
            tool_names=["web_search", "fetch_url", "read_file", "list_files"],
            role_description=ROLE_DESCRIPTIONS["research"],
            finish_schema=FINISH_SCHEMAS["quick_answer"],
        ),
        "enterprise_research": PackTemplate(
            template_id="enterprise_research",
            tool_names=["cross_run_search", "web_search", "fetch_url", "write_file", "read_file", "list_files"],
            role_description=ROLE_DESCRIPTIONS["enterprise_research"],
            finish_schema=FINISH_SCHEMAS["enterprise_report"],
        ),
    }


_cached_templates: Optional[Dict[str, PackTemplate]] = None


def _template_from_yaml(raw: Dict[str, Any]) -> Optional[PackTemplate]:
    """Parse a single template YAML dict into a PackTemplate.

    Returns None if the dict is malformed.
    """
    template_id = raw.get("template_id")
    if not template_id or not isinstance(template_id, str):
        return None

    tools = raw.get("tools", [])
    if not isinstance(tools, list):
        return None

    role_key = raw.get("role", "")
    role_description = ROLE_DESCRIPTIONS.get(role_key, "")
    if not role_description:
        logger.warning(
            "Template %r references unknown role %r; "
            "available roles: %s",
            template_id, role_key, sorted(ROLE_DESCRIPTIONS.keys()),
        )
        return None

    finish_schema_key = raw.get("finish_schema", "general")
    finish_schema = FINISH_SCHEMAS.get(finish_schema_key)
    if finish_schema is None:
        logger.warning(
            "Template %r references unknown finish_schema %r; using 'general'",
            template_id, finish_schema_key,
        )
        finish_schema = FINISH_SCHEMAS["general"]

    quality_gates = raw.get("quality_gates", [])
    if not isinstance(quality_gates, list):
        quality_gates = []

    return PackTemplate(
        template_id=str(template_id),
        tool_names=[str(t) for t in tools],
        role_description=role_description,
        finish_schema=finish_schema,
        quality_gates=[str(g) for g in quality_gates],
    )


def _load_templates() -> Dict[str, PackTemplate]:
    """Load templates from YAML files via the config hierarchy."""
    try:
        from agentic_concierge.config.external import load_all_yaml_configs
        raw_templates = load_all_yaml_configs("templates")
    except Exception:
        logger.debug("Failed to load templates from YAML; using fallback", exc_info=True)
        return _fallback_templates()

    if not raw_templates:
        return _fallback_templates()

    result: Dict[str, PackTemplate] = {}
    for name, raw in raw_templates.items():
        tpl = _template_from_yaml(raw)
        if tpl is not None:
            result[tpl.template_id] = tpl
        else:
            logger.warning("Skipping malformed template YAML: %s", name)

    if not result:
        return _fallback_templates()

    return result


def pack_templates() -> Dict[str, PackTemplate]:
    """Return the loaded pack templates, loading from YAML on first access."""
    global _cached_templates
    if _cached_templates is None:
        _cached_templates = _load_templates()
    return _cached_templates


def reload_templates() -> None:
    """Clear cached templates — forces re-load on next access."""
    global _cached_templates
    _cached_templates = None


# ---------------------------------------------------------------------------
# Dynamic pack builder
# ---------------------------------------------------------------------------

def build_dynamic_pack(
    specialist_id: str,
    tool_names: List[str],
    role_description: str,
    workspace_path: str,
    network_allowed: bool,
    finish_schema: Optional[Dict[str, Any]] = None,
    workspace_root: Optional[str] = None,
    quality_gates: Optional[List[str]] = None,
    finish_schema_key: Optional[str] = None,
    all_chat_models: Optional[List[str]] = None,
    base_url: str = "http://localhost:11434/v1",
    backend: str = "ollama",
    api_key: str = "",
) -> BaseSpecialistPack:
    """Construct a specialist pack dynamically from a list of tool names.

    Args:
        specialist_id: Unique identifier for this pack instance.
        tool_names: Tool names to include (must exist in TOOL_CATALOG).
        role_description: Role text for the system prompt.
        workspace_path: Workspace directory path for this run.
        network_allowed: Whether network tools are permitted.
        finish_schema: Custom finish_task schema (defaults to generic schema).
        workspace_root: Root of the workspace tree (for cross_run_search).
        quality_gates: Quality gate IDs (e.g. ["tests_verified"]).
        finish_schema_key: Optional key into FINISH_SCHEMAS. Overrides
            ``finish_schema`` when provided (e.g. "quick_answer", "code").

    Returns:
        A configured ``BaseSpecialistPack`` ready for use.
    """
    # finish_schema_key takes precedence over finish_schema dict
    if finish_schema_key is not None:
        finish_schema = get_finish_schema(finish_schema_key)
    policy = SandboxPolicy(root=Path(workspace_path), network_allowed=network_allowed)

    # Derive workspace_root from workspace_path if not provided
    if workspace_root is None:
        # Structure: {workspace_root}/runs/{run_id}/workspace → workspace_root = parent * 3
        workspace_root = str(Path(workspace_path).parent.parent.parent)

    # Collect quality gates from included tools if not explicitly provided
    effective_gates: List[str] = list(quality_gates or [])
    for tname in tool_names:
        entry = get_tool(tname)
        if entry.quality_gate and entry.quality_gate not in effective_gates:
            effective_gates.append(entry.quality_gate)

    # Build tools dict: filter network tools when not allowed
    tools: Dict[str, Tuple[Dict[str, Any], Any]] = {}
    for tname in tool_names:
        entry = get_tool(tname)
        if entry.requires_network and not network_allowed:
            continue
        executor = entry.executor_factory(
            policy,
            workspace_root=workspace_root,
            all_chat_models=all_chat_models or [],
            base_url=base_url,
            backend=backend,
            api_key=api_key,
        )
        tools[tname] = (entry.openai_def, executor)

    # Generate system prompt
    actual_tool_names = list(tools.keys())
    system_prompt = generate_system_prompt(
        role_description, actual_tool_names, quality_gates=effective_gates
    )

    return BaseSpecialistPack(
        specialist_id=specialist_id,
        system_prompt=system_prompt,
        tools=tools,
        finish_tool_def=finish_schema or FINISH_SCHEMAS["general"],
        workspace_path=workspace_path,
        network_allowed=network_allowed,
        quality_gates=effective_gates,
    )


def build_template_pack(
    template_id: str,
    workspace_path: str,
    network_allowed: bool,
    finish_schema_key: Optional[str] = None,
) -> BaseSpecialistPack:
    """Build a pack from a registered template.

    Args:
        template_id: Template name (must exist in loaded templates).
        workspace_path: Workspace directory path.
        network_allowed: Whether network tools are permitted.
        finish_schema_key: Optional override for the finish schema.
            When provided, overrides the template's built-in schema.

    Raises ``KeyError`` if ``template_id`` is not found.
    """
    templates = pack_templates()
    if template_id not in templates:
        raise KeyError(
            f"Unknown pack template {template_id!r}. "
            f"Available: {sorted(templates.keys())}"
        )
    tpl = templates[template_id]
    return build_dynamic_pack(
        specialist_id=template_id,
        tool_names=tpl.tool_names,
        role_description=tpl.role_description,
        workspace_path=workspace_path,
        network_allowed=network_allowed,
        finish_schema=tpl.finish_schema,
        quality_gates=tpl.quality_gates,
        finish_schema_key=finish_schema_key,
    )
