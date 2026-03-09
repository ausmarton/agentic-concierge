"""Tests for adaptive finish schemas (ADR-026)."""
from __future__ import annotations

import pytest

from agentic_concierge.infrastructure.specialists.finish_schemas import (
    FINISH_SCHEMA_KEYS,
    FINISH_SCHEMAS,
    get_finish_schema,
)


def test_all_schema_keys_present():
    assert "quick_answer" in FINISH_SCHEMAS
    assert "research_report" in FINISH_SCHEMAS
    assert "enterprise_report" in FINISH_SCHEMAS
    assert "code" in FINISH_SCHEMAS
    assert "general" in FINISH_SCHEMAS


def test_schema_keys_frozenset():
    assert isinstance(FINISH_SCHEMA_KEYS, frozenset)
    assert FINISH_SCHEMA_KEYS == set(FINISH_SCHEMAS.keys())


def test_get_finish_schema_known_key():
    schema = get_finish_schema("quick_answer")
    func = schema["function"]
    assert func["name"] == "finish_task"
    params = func["parameters"]
    assert "answer" in params["properties"]
    assert "answer" in params["required"]


def test_get_finish_schema_none_returns_general():
    schema = get_finish_schema(None)
    assert "summary" in schema["function"]["parameters"]["properties"]


def test_get_finish_schema_unknown_returns_general():
    schema = get_finish_schema("nonexistent")
    assert "summary" in schema["function"]["parameters"]["properties"]


def test_quick_answer_no_bibliography():
    """quick_answer schema should NOT require bibliography, evidence_table, or screening_log."""
    schema = get_finish_schema("quick_answer")
    props = schema["function"]["parameters"]["properties"]
    assert "bibliography_path" not in props
    assert "evidence_table_path" not in props
    assert "screening_log_path" not in props
    assert "executive_summary" not in props


def test_research_report_has_full_structure():
    schema = get_finish_schema("research_report")
    props = schema["function"]["parameters"]["properties"]
    assert "executive_summary" in props
    assert "citations" in props
    assert "bibliography_path" in props


def test_code_schema_has_tests_verified():
    schema = get_finish_schema("code")
    props = schema["function"]["parameters"]["properties"]
    assert "tests_verified" in props
    required = schema["function"]["parameters"]["required"]
    assert "tests_verified" in required


# ---------------------------------------------------------------------------
# Integration: finish_schema_key on build_dynamic_pack
# ---------------------------------------------------------------------------

def test_build_dynamic_pack_with_finish_schema_key(tmp_path):
    from agentic_concierge.infrastructure.specialists.dynamic_pack import build_dynamic_pack

    pack = build_dynamic_pack(
        specialist_id="test",
        tool_names=["read_file", "write_file"],
        role_description="test role",
        workspace_path=str(tmp_path),
        network_allowed=False,
        finish_schema_key="quick_answer",
    )
    finish_def = pack._finish_tool_def
    props = finish_def["function"]["parameters"]["properties"]
    assert "answer" in props
    assert "bibliography_path" not in props


def test_build_template_pack_with_finish_schema_override(tmp_path):
    """build_template_pack with finish_schema_key overrides the template's default."""
    from agentic_concierge.infrastructure.specialists.dynamic_pack import build_template_pack

    pack = build_template_pack(
        "research", str(tmp_path), network_allowed=True,
        finish_schema_key="quick_answer",
    )
    finish_def = pack._finish_tool_def
    props = finish_def["function"]["parameters"]["properties"]
    assert "answer" in props
    assert "bibliography_path" not in props


# ---------------------------------------------------------------------------
# SpecialistBrief.finish_schema
# ---------------------------------------------------------------------------

def test_specialist_brief_has_finish_schema():
    from agentic_concierge.application.orchestrator import SpecialistBrief
    b = SpecialistBrief(specialist_id="research", brief="test", finish_schema="quick_answer")
    assert b.finish_schema == "quick_answer"


def test_specialist_brief_default_finish_schema_is_none():
    from agentic_concierge.application.orchestrator import SpecialistBrief
    b = SpecialistBrief(specialist_id="research", brief="test")
    assert b.finish_schema is None
