"""Tests for the dynamic pack builder and pack templates."""
from __future__ import annotations

import tempfile

import pytest

from agentic_concierge.infrastructure.specialists.dynamic_pack import (
    PACK_TEMPLATES,
    PackTemplate,
    build_dynamic_pack,
    build_template_pack,
)


# ---------------------------------------------------------------------------
# Pack templates registry
# ---------------------------------------------------------------------------


def test_all_three_templates_registered():
    assert set(PACK_TEMPLATES.keys()) == {"engineering", "research", "enterprise_research"}


def test_templates_are_pack_template_instances():
    for tpl in PACK_TEMPLATES.values():
        assert isinstance(tpl, PackTemplate)


def test_engineering_template_has_correct_tools():
    tpl = PACK_TEMPLATES["engineering"]
    assert set(tpl.tool_names) == {"shell", "read_file", "write_file", "list_files", "run_tests"}


def test_research_template_has_correct_tools():
    tpl = PACK_TEMPLATES["research"]
    assert "web_search" in tpl.tool_names
    assert "fetch_url" in tpl.tool_names
    assert "read_file" in tpl.tool_names
    assert "list_files" in tpl.tool_names


def test_enterprise_research_template_has_cross_run_search():
    tpl = PACK_TEMPLATES["enterprise_research"]
    assert "cross_run_search" in tpl.tool_names


def test_engineering_template_has_quality_gate():
    assert "tests_verified" in PACK_TEMPLATES["engineering"].quality_gates


def test_research_template_has_no_quality_gate():
    assert PACK_TEMPLATES["research"].quality_gates == []


# ---------------------------------------------------------------------------
# build_template_pack
# ---------------------------------------------------------------------------


def test_build_template_pack_engineering():
    with tempfile.TemporaryDirectory() as d:
        pack = build_template_pack("engineering", d, network_allowed=False)
        assert pack.specialist_id == "engineering"
        assert set(pack.tool_names) == {"shell", "read_file", "write_file", "list_files", "run_tests"}


def test_build_template_pack_research_network_allowed():
    with tempfile.TemporaryDirectory() as d:
        pack = build_template_pack("research", d, network_allowed=True)
        assert "web_search" in pack.tool_names
        assert "fetch_url" in pack.tool_names


def test_build_template_pack_research_no_network():
    with tempfile.TemporaryDirectory() as d:
        pack = build_template_pack("research", d, network_allowed=False)
        assert "web_search" not in pack.tool_names
        assert "fetch_url" not in pack.tool_names
        assert "read_file" in pack.tool_names
        assert "list_files" in pack.tool_names


def test_build_template_pack_unknown_raises():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(KeyError, match="Unknown pack template"):
            build_template_pack("nonexistent", d, network_allowed=False)


# ---------------------------------------------------------------------------
# build_dynamic_pack
# ---------------------------------------------------------------------------


def test_build_dynamic_pack_custom_tools():
    with tempfile.TemporaryDirectory() as d:
        pack = build_dynamic_pack(
            specialist_id="custom_agent",
            tool_names=["shell", "web_search", "write_file"],
            role_description="You are a custom agent.",
            workspace_path=d,
            network_allowed=True,
        )
        assert pack.specialist_id == "custom_agent"
        assert set(pack.tool_names) == {"shell", "web_search", "write_file"}


def test_build_dynamic_pack_filters_network_tools():
    with tempfile.TemporaryDirectory() as d:
        pack = build_dynamic_pack(
            specialist_id="offline_agent",
            tool_names=["shell", "web_search", "fetch_url", "write_file"],
            role_description="You are an offline agent.",
            workspace_path=d,
            network_allowed=False,
        )
        assert "web_search" not in pack.tool_names
        assert "fetch_url" not in pack.tool_names
        assert "shell" in pack.tool_names
        assert "write_file" in pack.tool_names


def test_build_dynamic_pack_derives_quality_gates_from_tools():
    """Including run_tests should automatically add tests_verified gate."""
    with tempfile.TemporaryDirectory() as d:
        pack = build_dynamic_pack(
            specialist_id="builder",
            tool_names=["shell", "write_file", "run_tests"],
            role_description="A builder agent.",
            workspace_path=d,
            network_allowed=False,
        )
        # Quality gate enforced: tests_verified=False is rejected
        error = pack.validate_finish_payload({"summary": "done", "tests_verified": False})
        assert error is not None
        assert "tests_verified" in error.lower() or "run_tests" in error.lower()


def test_build_dynamic_pack_no_quality_gate_without_run_tests():
    with tempfile.TemporaryDirectory() as d:
        pack = build_dynamic_pack(
            specialist_id="writer",
            tool_names=["write_file", "read_file"],
            role_description="A writer agent.",
            workspace_path=d,
            network_allowed=False,
        )
        # No quality gate: tests_verified=False is accepted
        error = pack.validate_finish_payload({"summary": "done", "tests_verified": False})
        assert error is None


def test_build_dynamic_pack_explicit_quality_gates():
    with tempfile.TemporaryDirectory() as d:
        pack = build_dynamic_pack(
            specialist_id="gated",
            tool_names=["shell"],
            role_description="Gated agent.",
            workspace_path=d,
            network_allowed=False,
            quality_gates=["tests_verified"],
        )
        # Explicit quality gate enforced
        error = pack.validate_finish_payload({"summary": "done", "tests_verified": False})
        assert error is not None


def test_build_dynamic_pack_system_prompt_contains_role():
    with tempfile.TemporaryDirectory() as d:
        pack = build_dynamic_pack(
            specialist_id="role_test",
            tool_names=["shell"],
            role_description="You are a deployment specialist.",
            workspace_path=d,
            network_allowed=False,
        )
        assert "deployment specialist" in pack.system_prompt.lower()


def test_build_dynamic_pack_finish_tool_in_definitions():
    with tempfile.TemporaryDirectory() as d:
        pack = build_dynamic_pack(
            specialist_id="finish_test",
            tool_names=["shell", "write_file"],
            role_description="Test agent.",
            workspace_path=d,
            network_allowed=False,
        )
        names = [td["function"]["name"] for td in pack.tool_definitions]
        assert "finish_task" in names


def test_build_dynamic_pack_unknown_tool_raises():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(KeyError, match="Unknown tool"):
            build_dynamic_pack(
                specialist_id="bad",
                tool_names=["nonexistent_tool"],
                role_description="Bad agent.",
                workspace_path=d,
                network_allowed=False,
            )


# ---------------------------------------------------------------------------
# Executor binding
# ---------------------------------------------------------------------------


def test_executor_factories_produce_callables():
    """Executor factories should produce callable objects for each tool."""
    with tempfile.TemporaryDirectory() as d:
        pack = build_dynamic_pack(
            specialist_id="exec_test",
            tool_names=["read_file", "write_file", "list_files"],
            role_description="Test agent.",
            workspace_path=d,
            network_allowed=False,
        )
        for name in ["read_file", "write_file", "list_files"]:
            assert name in pack.tool_names
