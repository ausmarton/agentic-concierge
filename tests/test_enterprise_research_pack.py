"""Tests for the enterprise research specialist pack."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build(tmp_path: Path, *, network_allowed: bool = False):
    """Build an enterprise_research pack with a real workspace directory."""
    from agentic_concierge.infrastructure.specialists.dynamic_pack import build_template_pack

    workspace = str(tmp_path / "runs" / "run1" / "workspace")
    Path(workspace).mkdir(parents=True, exist_ok=True)
    return build_template_pack("enterprise_research", workspace, network_allowed=network_allowed)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def test_system_prompt_mentions_enterprise_sources(tmp_path: Path):
    pack = _build(tmp_path)
    prompt = pack.system_prompt
    assert "Confluence" in prompt or "confluence" in prompt.lower()
    assert "Jira" in prompt or "jira" in prompt.lower()
    assert "GitHub" in prompt or "github" in prompt.lower()


def test_system_prompt_mentions_staleness(tmp_path: Path):
    pack = _build(tmp_path)
    assert "staleness" in pack.system_prompt.lower() or "stale" in pack.system_prompt.lower()


def test_system_prompt_mentions_confidence(tmp_path: Path):
    pack = _build(tmp_path)
    assert "confidence" in pack.system_prompt.lower() or "HIGH" in pack.system_prompt


# ---------------------------------------------------------------------------
# Capabilities / specialist_id
# ---------------------------------------------------------------------------


def test_specialist_id(tmp_path: Path):
    pack = _build(tmp_path)
    assert pack.specialist_id == "enterprise_research"


def test_default_config_includes_enterprise_research():
    from agentic_concierge.config.schema import DEFAULT_CONFIG

    assert "enterprise_research" in DEFAULT_CONFIG.specialists


def test_enterprise_research_capabilities_in_default_config():
    from agentic_concierge.config.schema import DEFAULT_CONFIG

    caps = DEFAULT_CONFIG.specialists["enterprise_research"].capabilities
    assert "enterprise_search" in caps
    assert "github_search" in caps


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


def test_has_cross_run_search_tool(tmp_path: Path):
    pack = _build(tmp_path)
    names = {td["function"]["name"] for td in pack.tool_definitions}
    assert "cross_run_search" in names


def test_has_file_tools(tmp_path: Path):
    pack = _build(tmp_path)
    names = {td["function"]["name"] for td in pack.tool_definitions}
    assert "write_file" in names
    assert "read_file" in names
    assert "list_files" in names


def test_no_web_tools_when_network_not_allowed(tmp_path: Path):
    pack = _build(tmp_path, network_allowed=False)
    names = {td["function"]["name"] for td in pack.tool_definitions}
    assert "web_search" not in names
    assert "fetch_url" not in names


def test_has_web_tools_when_network_allowed(tmp_path: Path):
    pack = _build(tmp_path, network_allowed=True)
    names = {td["function"]["name"] for td in pack.tool_definitions}
    assert "web_search" in names
    assert "fetch_url" in names


def test_cross_run_search_tool_def_has_required_query(tmp_path: Path):
    pack = _build(tmp_path)
    crs_def = next(
        td for td in pack.tool_definitions if td["function"]["name"] == "cross_run_search"
    )
    schema = crs_def["function"]["parameters"]
    assert "query" in schema["properties"]
    assert "query" in schema.get("required", [])


def test_finish_tool_has_executive_summary_field(tmp_path: Path):
    pack = _build(tmp_path)
    finish = next(
        td for td in pack.tool_definitions if td["function"]["name"] == pack.finish_tool_name
    )
    props = finish["function"]["parameters"]["properties"]
    assert "executive_summary" in props
    assert "sources" in props
    assert "key_findings" in props


# ---------------------------------------------------------------------------
# cross_run_search tool execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_run_search_returns_empty_when_no_index(tmp_path: Path):
    pack = _build(tmp_path)
    result = await pack.execute_tool("cross_run_search", {"query": "kubernetes"})
    assert "results" in result
    assert result["results"] == []


@pytest.mark.asyncio
async def test_cross_run_search_finds_entries_in_index(tmp_path: Path):
    from agentic_concierge.infrastructure.workspace.run_index import RunIndexEntry, append_to_index

    workspace = str(tmp_path / "runs" / "run1" / "workspace")
    Path(workspace).mkdir(parents=True, exist_ok=True)
    workspace_root = str(tmp_path)

    # Write an entry to the index
    append_to_index(
        workspace_root,
        RunIndexEntry(
            run_id="prior_run_1",
            timestamp=1000.0,
            specialist_ids=["research"],
            prompt_prefix="survey of kubernetes deployment strategies",
            summary="k8s deployment patterns reviewed",
            workspace_path=str(tmp_path / "runs" / "prior_run_1" / "workspace"),
            run_dir=str(tmp_path / "runs" / "prior_run_1"),
        ),
    )

    from agentic_concierge.infrastructure.specialists.dynamic_pack import build_template_pack

    pack = build_template_pack("enterprise_research", workspace, network_allowed=False)
    result = await pack.execute_tool("cross_run_search", {"query": "kubernetes"})

    assert result["count"] == 1
    assert result["results"][0]["run_id"] == "prior_run_1"
    assert "kubernetes" in result["results"][0]["prompt"].lower()
