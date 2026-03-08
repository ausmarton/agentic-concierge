"""Adaptive finish schemas: pick the right shape based on task complexity.

Replaces the rigid per-template finish schemas with a selectable set.
The orchestrator chooses the schema key during plan creation; the pack
builder uses it to construct the finish tool definition.

Schema keys:
- ``quick_answer``: Simple factual answers — no artifacts, no citations.
- ``research_report``: Full academic structure (evidence table, bibliography).
- ``enterprise_report``: Enterprise research with confidence/staleness notes.
- ``code``: Engineering output with tests_verified gate.
- ``general``: Generic summary + artifacts.

When no schema key is specified, the template's built-in default is used.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .tool_defs import make_finish_tool_def


# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

_QUICK_ANSWER_SCHEMA = make_finish_tool_def(
    description=(
        "Call this when you have a direct answer. Provide a clear, concise "
        "answer to the question. No need for artifacts or bibliography."
    ),
    properties={
        "answer": {
            "type": "string",
            "description": "Direct answer to the question.",
        },
        "sources": {
            "type": "array",
            "items": {"type": "string"},
            "description": "URLs actually visited (if any).",
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
            "description": "Confidence in the answer.",
        },
    },
    required=["answer"],
)

_RESEARCH_REPORT_SCHEMA = make_finish_tool_def(
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

_ENTERPRISE_REPORT_SCHEMA = make_finish_tool_def(
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

_CODE_SCHEMA = make_finish_tool_def(
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

_GENERAL_SCHEMA = make_finish_tool_def(
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


FINISH_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "quick_answer": _QUICK_ANSWER_SCHEMA,
    "research_report": _RESEARCH_REPORT_SCHEMA,
    "enterprise_report": _ENTERPRISE_REPORT_SCHEMA,
    "code": _CODE_SCHEMA,
    "general": _GENERAL_SCHEMA,
}

# Valid schema keys (for validation in orchestrator)
FINISH_SCHEMA_KEYS = frozenset(FINISH_SCHEMAS.keys())


def get_finish_schema(key: Optional[str] = None) -> Dict[str, Any]:
    """Look up a finish schema by key.

    Returns the ``general`` schema when key is None or unknown.
    """
    if key is None:
        return _GENERAL_SCHEMA
    return FINISH_SCHEMAS.get(key, _GENERAL_SCHEMA)
