"""Tests for the tool catalog: registration, lookup, categories, and filtering."""
from __future__ import annotations

import pytest

from agentic_concierge.infrastructure.specialists.tool_catalog import (
    TOOL_CATALOG,
    ToolEntry,
    get_tool,
    list_tools,
    list_tools_by_category,
    tool_catalog_summary,
)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_catalog_has_all_expected_tools():
    expected = {"shell", "read_file", "write_file", "list_files",
                "run_tests", "web_search", "fetch_url", "cross_run_search",
                "consult_specialist_model"}
    assert set(TOOL_CATALOG.keys()) == expected


def test_all_entries_are_tool_entry_instances():
    for entry in TOOL_CATALOG.values():
        assert isinstance(entry, ToolEntry)


def test_entry_name_matches_key():
    for key, entry in TOOL_CATALOG.items():
        assert entry.name == key


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def test_get_tool_returns_correct_entry():
    entry = get_tool("shell")
    assert entry.name == "shell"
    assert entry.category == "code"


def test_get_tool_raises_for_unknown():
    with pytest.raises(KeyError, match="Unknown tool"):
        get_tool("nonexistent_tool")


# ---------------------------------------------------------------------------
# Category filtering
# ---------------------------------------------------------------------------


def test_list_tools_returns_all():
    assert len(list_tools()) == len(TOOL_CATALOG)


def test_list_tools_by_category_file_io():
    file_tools = list_tools_by_category("file_io")
    names = {t.name for t in file_tools}
    assert names == {"read_file", "write_file", "list_files"}


def test_list_tools_by_category_web():
    web_tools = list_tools_by_category("web")
    names = {t.name for t in web_tools}
    assert names == {"web_search", "fetch_url"}


def test_list_tools_by_category_code():
    code_tools = list_tools_by_category("code")
    names = {t.name for t in code_tools}
    assert names == {"shell", "run_tests"}


def test_list_tools_by_category_empty_for_unknown():
    assert list_tools_by_category("nonexistent") == []


# ---------------------------------------------------------------------------
# Network flag
# ---------------------------------------------------------------------------


def test_network_required_tools():
    net_tools = {e.name for e in TOOL_CATALOG.values() if e.requires_network}
    assert net_tools == {"web_search", "fetch_url"}


def test_non_network_tools():
    non_net = {e.name for e in TOOL_CATALOG.values() if not e.requires_network}
    assert "shell" in non_net
    assert "read_file" in non_net
    assert "cross_run_search" in non_net


# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------


def test_run_tests_has_quality_gate():
    assert get_tool("run_tests").quality_gate == "tests_verified"


def test_other_tools_have_no_quality_gate():
    for name in ("shell", "read_file", "write_file", "list_files", "web_search", "fetch_url"):
        assert get_tool(name).quality_gate is None


# ---------------------------------------------------------------------------
# OpenAI definitions
# ---------------------------------------------------------------------------


def test_openai_defs_have_required_structure():
    for entry in TOOL_CATALOG.values():
        td = entry.openai_def
        assert td["type"] == "function"
        assert "name" in td["function"]
        assert "parameters" in td["function"]
        assert td["function"]["name"] == entry.name


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_tool_catalog_summary_mentions_all_tools():
    summary = tool_catalog_summary()
    for name in TOOL_CATALOG:
        assert name in summary


def test_tool_catalog_summary_marks_network_tools():
    summary = tool_catalog_summary()
    assert "(requires network)" in summary
