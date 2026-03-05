"""Tests for the persistent run index (P6-1).

Covers RunIndexEntry creation, append_to_index, search_index,
and the fabric logs search CLI subcommand.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from agentic_concierge.infrastructure.workspace.run_index import (
    RunIndexEntry,
    append_to_index,
    search_index,
    _entry_from_dict,
)
from agentic_concierge.interfaces.cli import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(
    run_id: str = "run-001",
    prompt: str = "implement a REST API in Python",
    summary: str = "Created FastAPI service with /health endpoint.",
    specialist_ids: list | None = None,
    routing_method: str = "keyword",
    model_name: str = "llama3.1:8b",
    ts: float | None = None,
) -> RunIndexEntry:
    return RunIndexEntry(
        run_id=run_id,
        timestamp=ts or time.time(),
        specialist_ids=specialist_ids or ["engineering"],
        prompt_prefix=prompt[:200],
        summary=summary,
        workspace_path=f"/tmp/fabric/runs/{run_id}/workspace",
        run_dir=f"/tmp/fabric/runs/{run_id}",
        routing_method=routing_method,
        model_name=model_name,
    )


# ---------------------------------------------------------------------------
# append_to_index
# ---------------------------------------------------------------------------

def test_append_creates_index_file(tmp_path):
    """append_to_index creates run_index.jsonl when it doesn't exist."""
    entry = _entry()
    append_to_index(str(tmp_path), entry)
    index = tmp_path / "run_index.jsonl"
    assert index.is_file()


def test_append_writes_valid_json_line(tmp_path):
    """Each appended entry is valid JSON on a single line."""
    entry = _entry(run_id="run-abc")
    append_to_index(str(tmp_path), entry)
    line = (tmp_path / "run_index.jsonl").read_text().strip()
    data = json.loads(line)
    assert data["run_id"] == "run-abc"
    assert "timestamp" in data
    assert "specialist_ids" in data


def test_append_multiple_entries_each_on_own_line(tmp_path):
    """Multiple appends produce multiple JSONL lines."""
    for i in range(3):
        append_to_index(str(tmp_path), _entry(run_id=f"run-{i}"))
    lines = [line for line in (tmp_path / "run_index.jsonl").read_text().splitlines() if line.strip()]
    assert len(lines) == 3


def test_append_creates_parent_directories(tmp_path):
    """append_to_index creates workspace_root if it doesn't exist."""
    deep = tmp_path / "a" / "b" / "c"
    append_to_index(str(deep), _entry())
    assert (deep / "run_index.jsonl").is_file()


# ---------------------------------------------------------------------------
# search_index
# ---------------------------------------------------------------------------

def test_search_returns_empty_list_when_index_missing(tmp_path):
    """search_index returns [] when run_index.jsonl doesn't exist."""
    results = search_index(str(tmp_path), "python")
    assert results == []


def test_search_matches_prompt_prefix(tmp_path):
    """search_index matches against prompt_prefix (case-insensitive)."""
    append_to_index(str(tmp_path), _entry(prompt="implement a kubernetes operator"))
    append_to_index(str(tmp_path), _entry(run_id="r2", prompt="systematic review of NLP papers"))
    results = search_index(str(tmp_path), "kubernetes")
    assert len(results) == 1
    assert "kubernetes" in results[0].prompt_prefix


def test_search_matches_summary(tmp_path):
    """search_index matches against summary (case-insensitive)."""
    append_to_index(str(tmp_path), _entry(summary="Deployed to GKE cluster successfully."))
    append_to_index(str(tmp_path), _entry(run_id="r2", summary="No relevant results found."))
    results = search_index(str(tmp_path), "gke")
    assert len(results) == 1
    assert "GKE" in results[0].summary


def test_search_is_case_insensitive(tmp_path):
    """search_index match is case-insensitive for both query and content."""
    append_to_index(str(tmp_path), _entry(prompt="Build a Rust CLI tool"))
    assert search_index(str(tmp_path), "RUST")
    assert search_index(str(tmp_path), "rust")
    assert search_index(str(tmp_path), "Rust")


def test_search_returns_most_recent_first(tmp_path):
    """search_index returns results sorted most-recent-first."""
    old_ts = time.time() - 3600
    new_ts = time.time()
    append_to_index(str(tmp_path), _entry(run_id="old", prompt="build api", ts=old_ts))
    append_to_index(str(tmp_path), _entry(run_id="new", prompt="build cli", ts=new_ts))
    results = search_index(str(tmp_path), "build")
    assert results[0].run_id == "new"
    assert results[1].run_id == "old"


def test_search_respects_limit(tmp_path):
    """search_index returns at most `limit` entries."""
    for i in range(10):
        append_to_index(str(tmp_path), _entry(run_id=f"r{i}", prompt="python service"))
    results = search_index(str(tmp_path), "python", limit=3)
    assert len(results) == 3


def test_search_skips_malformed_lines(tmp_path):
    """search_index silently skips lines that are not valid JSON."""
    index = tmp_path / "run_index.jsonl"
    index.write_text('{"run_id": "ok", "prompt_prefix": "python api", "summary": "", "timestamp": 1, "specialist_ids": [], "workspace_path": "", "run_dir": "", "routing_method": "", "model_name": ""}\nnot-valid-json\n')
    results = search_index(str(tmp_path), "python")
    assert len(results) == 1
    assert results[0].run_id == "ok"


# ---------------------------------------------------------------------------
# fabric logs search CLI
# ---------------------------------------------------------------------------

def test_logs_search_cli_no_results(tmp_path):
    """fabric logs search returns a 'no results' message when index is empty."""
    runner = CliRunner()
    result = runner.invoke(app, ["logs", "search", "python", "--workspace", str(tmp_path)])
    assert result.exit_code == 0
    assert "No runs matching" in result.output


def test_logs_search_cli_shows_matching_runs(tmp_path):
    """fabric logs search prints a table with matching runs."""
    append_to_index(str(tmp_path), _entry(prompt="implement fastapi service", summary="Done."))
    runner = CliRunner()
    result = runner.invoke(app, ["logs", "search", "fastapi", "--workspace", str(tmp_path)])
    assert result.exit_code == 0
    assert "fastapi" in result.output.lower()


# ---------------------------------------------------------------------------
# execute_task integration: index is written after a successful run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_task_appends_to_index(tmp_path):
    """execute_task appends one entry to run_index.jsonl after a successful run."""
    from agentic_concierge.application.execute_task import execute_task
    from agentic_concierge.config.schema import ConciergeConfig, ModelConfig, SpecialistConfig
    from agentic_concierge.domain import Task, LLMResponse, ToolCallRequest, RunResult

    # Minimal config — must include "quality" key (execute_task falls back to it)
    config = ConciergeConfig(
        models={"quality": ModelConfig(base_url="http://localhost:11434/v1", model="test")},
        specialists={
            "engineering": SpecialistConfig(description="eng", keywords=[], workflow="engineering")
        },
    )

    # Stub pack
    class _Pack:
        specialist_id = "engineering"
        system_prompt = "sys"
        finish_tool_name = "finish_task"
        finish_required_fields = ["summary"]
        tool_definitions = [
            {"type": "function", "function": {"name": "shell"}},
            {"type": "function", "function": {"name": "finish_task", "parameters": {"properties": {"summary": {}}, "required": ["summary"]}}},
        ]
        async def execute_tool(self, name, args): return {"stdout": "ok"}
        async def aopen(self): pass
        async def aclose(self): pass

    # Chat client: first call returns shell tool call; second call returns finish_task
    call_count = 0
    async def _chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(call_id="c1", tool_name="shell", arguments={"cmd": "ls"})],
            )
        return LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(call_id="c2", tool_name="finish_task", arguments={"summary": "all done", "artifacts": [], "next_steps": [], "notes": ""})],
        )

    chat_client = MagicMock()
    chat_client.chat = _chat

    run_dir_path = str(tmp_path / "runs" / "test-run")
    Path(run_dir_path).mkdir(parents=True)

    class _Repo:
        def create_run(self):
            from agentic_concierge.domain import RunId
            return RunId("test-run"), run_dir_path, str(tmp_path / "workspace")
        def append_event(self, *a, **kw): pass

    class _Registry:
        def get_pack(self, sid, ws, net, **kw): return _Pack()

    task = Task(prompt="list files in workspace", specialist_id="engineering")
    await execute_task(
        task,
        chat_client=chat_client,
        run_repository=_Repo(),
        specialist_registry=_Registry(),
        config=config,
        max_review_iterations=0,
    )

    index = tmp_path / "run_index.jsonl"
    assert index.is_file(), "run_index.jsonl should have been created"
    data = json.loads(index.read_text().strip())
    assert data["run_id"] == "test-run"
    assert data["prompt_prefix"] == "list files in workspace"
    assert data["summary"] == "all done"
    assert data["specialist_ids"] == ["engineering"]
