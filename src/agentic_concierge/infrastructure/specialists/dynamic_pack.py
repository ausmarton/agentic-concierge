"""Dynamic pack builder: construct specialist packs from tool selections + role descriptions.

``build_dynamic_pack()`` is the central entry point for constructing packs at
runtime.  It looks up tools in the :mod:`tool_catalog`, creates executors bound
to the workspace, and assembles a system prompt from the role description.

``PackTemplate`` and ``PACK_TEMPLATES`` provide data-driven replacements for the
three legacy builder functions (engineering, research, enterprise_research).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from agentic_concierge.infrastructure.tools.sandbox import SandboxPolicy

from .base import BaseSpecialistPack
from .prompts import (
    ROLE_ENGINEERING,
    ROLE_ENTERPRISE_RESEARCH,
    ROLE_RESEARCH,
    generate_system_prompt,
)
from .tool_catalog import TOOL_CATALOG, get_tool
from .tool_defs import make_finish_tool_def


# ---------------------------------------------------------------------------
# Finish-tool schemas (one per template)
# ---------------------------------------------------------------------------

_ENGINEERING_FINISH_SCHEMA = make_finish_tool_def(
    description=(
        "Call this when the task is complete. Provide a clear summary of what was "
        "accomplished, list any artefact file paths, and note any remaining steps "
        "(e.g. deployment commands that require human approval). "
        "You MUST call run_tests first and set tests_verified=true."
    ),
    properties={
        "summary": {
            "type": "string",
            "description": "What was accomplished (be specific).",
        },
        "artifacts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Relative paths of files created or modified.",
        },
        "next_steps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Any remaining steps, especially ones needing human approval.",
        },
        "notes": {
            "type": "string",
            "description": "Caveats, test commands, or anything useful to know.",
        },
        "tests_verified": {
            "type": "boolean",
            "description": (
                "Set to true only after run_tests confirms all tests pass. "
                "Do not call finish_task with false — fix failures first."
            ),
        },
    },
    required=["summary", "tests_verified"],
)

_RESEARCH_FINISH_SCHEMA = make_finish_tool_def(
    description=(
        "Call this when research is complete. Provide your executive summary, key "
        "findings, citations for all fetched URLs, paths to artefact files in the "
        "workspace, and any gaps or future work."
    ),
    properties={
        "executive_summary": {
            "type": "string",
            "description": "High-level summary of findings.",
        },
        "key_findings": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The most important findings, as a list.",
        },
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "fetched_at": {"type": "string"},
                    "claim": {"type": "string", "description": "What this source supports."},
                },
                "required": ["url", "claim"],
            },
            "description": "Only URLs actually fetched via fetch_url.",
        },
        "evidence_table_path": {
            "type": "string",
            "description": "Workspace-relative path to the evidence table file.",
        },
        "screening_log_path": {
            "type": "string",
            "description": "Workspace-relative path to the screening log file.",
        },
        "bibliography_path": {
            "type": "string",
            "description": "Workspace-relative path to the bibliography file.",
        },
        "gaps_and_future_work": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Open questions or areas for further research.",
        },
        "notes": {
            "type": "string",
            "description": "How to reproduce searches, caveats, etc.",
        },
    },
    required=["executive_summary"],
)

_ENTERPRISE_RESEARCH_FINISH_SCHEMA = make_finish_tool_def(
    description=(
        "Call this when enterprise research is complete. Provide an executive summary, "
        "source attributions with confidence ratings ([HIGH]/[MEDIUM]/[LOW]/[STALE?]), "
        "staleness notes, and paths to the written report and artefact files."
    ),
    properties={
        "executive_summary": {
            "type": "string",
            "description": "High-level summary of findings with staleness/confidence overview.",
        },
        "key_findings": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Key findings, each annotated with confidence: "
                "[HIGH]/[MEDIUM]/[LOW]/[STALE?]/[UNVERIFIED]."
            ),
        },
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Tool name or URL used."},
                    "content_summary": {"type": "string"},
                    "confidence": {
                        "type": "string",
                        "enum": ["HIGH", "MEDIUM", "LOW", "STALE?", "UNVERIFIED"],
                    },
                    "staleness_note": {"type": "string"},
                },
                "required": ["source", "content_summary", "confidence"],
            },
            "description": "All sources retrieved during research.",
        },
        "report_path": {
            "type": "string",
            "description": "Workspace-relative path to the written report file.",
        },
        "gaps_and_future_work": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Open questions, missing information, or recommended follow-up.",
        },
        "notes": {
            "type": "string",
            "description": "Caveats, reproducibility notes, or session metadata.",
        },
    },
    required=["executive_summary"],
)

# Default finish schema for dynamic (non-template) packs.
_DEFAULT_FINISH_SCHEMA = make_finish_tool_def(
    description=(
        "Call this when the task is complete. Provide a clear summary of what was "
        "accomplished and any artefact file paths."
    ),
    properties={
        "summary": {
            "type": "string",
            "description": "What was accomplished (be specific).",
        },
        "artifacts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Relative paths of files created or modified.",
        },
        "notes": {
            "type": "string",
            "description": "Caveats or anything useful to know.",
        },
    },
    required=["summary"],
)


# ---------------------------------------------------------------------------
# Pack templates
# ---------------------------------------------------------------------------

@dataclass
class PackTemplate:
    """Data-driven pack configuration template."""

    template_id: str
    tool_names: List[str]
    role_description: str
    finish_schema: Dict[str, Any]
    quality_gates: List[str] = field(default_factory=list)


PACK_TEMPLATES: Dict[str, PackTemplate] = {
    "engineering": PackTemplate(
        template_id="engineering",
        tool_names=["shell", "read_file", "write_file", "list_files", "run_tests"],
        role_description=ROLE_ENGINEERING,
        finish_schema=_ENGINEERING_FINISH_SCHEMA,
        quality_gates=["tests_verified"],
    ),
    "research": PackTemplate(
        template_id="research",
        tool_names=["web_search", "fetch_url", "write_file", "read_file", "list_files"],
        role_description=ROLE_RESEARCH,
        finish_schema=_RESEARCH_FINISH_SCHEMA,
    ),
    "enterprise_research": PackTemplate(
        template_id="enterprise_research",
        tool_names=["cross_run_search", "web_search", "fetch_url", "write_file", "read_file", "list_files"],
        role_description=ROLE_ENTERPRISE_RESEARCH,
        finish_schema=_ENTERPRISE_RESEARCH_FINISH_SCHEMA,
    ),
}


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

    Returns:
        A configured ``BaseSpecialistPack`` ready for use.
    """
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
        executor = entry.executor_factory(policy, workspace_root=workspace_root)
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
        finish_tool_def=finish_schema or _DEFAULT_FINISH_SCHEMA,
        workspace_path=workspace_path,
        network_allowed=network_allowed,
        quality_gates=effective_gates,
    )


def build_template_pack(
    template_id: str,
    workspace_path: str,
    network_allowed: bool,
) -> BaseSpecialistPack:
    """Build a pack from a registered template.

    Raises ``KeyError`` if ``template_id`` is not in ``PACK_TEMPLATES``.
    """
    if template_id not in PACK_TEMPLATES:
        raise KeyError(
            f"Unknown pack template {template_id!r}. "
            f"Available: {sorted(PACK_TEMPLATES.keys())}"
        )
    tpl = PACK_TEMPLATES[template_id]
    return build_dynamic_pack(
        specialist_id=template_id,
        tool_names=tpl.tool_names,
        role_description=tpl.role_description,
        workspace_path=workspace_path,
        network_allowed=network_allowed,
        finish_schema=tpl.finish_schema,
        quality_gates=tpl.quality_gates,
    )
