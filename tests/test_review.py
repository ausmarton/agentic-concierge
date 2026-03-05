"""Tests for the universal work review mechanism (Gate 4).

These tests verify that the independent reviewer step works correctly:
- Review approval allows finish_task to succeed
- Review rejection sends critique back to the doer
- Max review iterations prevents infinite loops
- Reviewer has only safe (read-only) tools
- Review events appear in the runlog
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agentic_concierge.application.execute_task import (
    execute_task,
    _APPROVE_WORK_TOOL_DEF,
    _REQUEST_REVISION_TOOL_DEF,
    _SAFE_REVIEWER_TOOLS,
)
from agentic_concierge.config import load_config
from agentic_concierge.domain import LLMResponse, Task, ToolCallRequest
from agentic_concierge.infrastructure.ollama import OllamaChatClient
from agentic_concierge.infrastructure.specialists import ConfigSpecialistRegistry
from agentic_concierge.infrastructure.workspace import FileSystemRunRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finish_response(call_id: str = "c1", **kwargs) -> LLMResponse:
    defaults = {
        "summary": "Done",
        "artifacts": [],
        "next_steps": [],
        "notes": "",
        "tests_verified": True,
    }
    defaults.update(kwargs)
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(call_id=call_id, tool_name="finish_task", arguments=defaults)],
    )


def _tool_response(tool_name: str, call_id: str = "c1", **kwargs) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(call_id=call_id, tool_name=tool_name, arguments=kwargs)],
    )


def _approve_response(call_id: str = "r1", comment: str = "Looks good") -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            call_id=call_id, tool_name="approve_work",
            arguments={"comment": comment},
        )],
    )


def _reject_response(call_id: str = "r1", critique: str = "Tests not verified") -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            call_id=call_id, tool_name="request_revision",
            arguments={"critique": critique},
        )],
    )


def _read_runlog(run_dir: str) -> list[dict]:
    lines = Path(run_dir, "runlog.jsonl").read_text().strip().splitlines()
    return [json.loads(ln) for ln in lines if ln]


async def _run_review(
    chat_mock_responses,
    *,
    tmp_path,
    specialist_id="engineering",
    max_steps=10,
    max_review_iterations=2,
):
    """Execute a task with review enabled."""
    config = load_config()
    run_repository = FileSystemRunRepository(workspace_root=str(tmp_path))
    specialist_registry = ConfigSpecialistRegistry(config)
    with patch.object(
        OllamaChatClient, "chat", new_callable=AsyncMock,
        side_effect=chat_mock_responses,
    ):
        chat_client = OllamaChatClient(
            base_url="http://localhost:11434/v1", timeout_s=5.0,
        )
        task = Task(
            prompt="test task", specialist_id=specialist_id,
            network_allowed=False,
        )
        return await execute_task(
            task,
            chat_client=chat_client,
            run_repository=run_repository,
            specialist_registry=specialist_registry,
            config=config,
            max_steps=max_steps,
            max_review_iterations=max_review_iterations,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_review_approved_lets_finish_through(tmp_path):
    """Doer: list_files -> finish_task; reviewer: approve_work -> payload accepted."""
    result = await _run_review(
        [
            _tool_response("list_files", call_id="c0"),
            _finish_response(call_id="c1"),
            _approve_response(),
        ],
        tmp_path=tmp_path,
    )
    assert result.payload["action"] == "final"
    assert result.payload["summary"] == "Done"


@pytest.mark.asyncio
async def test_review_rejected_sends_critique_to_doer(tmp_path):
    """Reviewer rejects first time, doer retries, reviewer approves second time."""
    result = await _run_review(
        [
            _tool_response("list_files", call_id="c0"),
            _finish_response(call_id="c1"),
            _reject_response(critique="Missing error handling"),
            _finish_response(call_id="c2", summary="Done with error handling"),
            _approve_response(call_id="r2"),
        ],
        tmp_path=tmp_path,
    )
    assert result.payload["action"] == "final"
    assert result.payload["summary"] == "Done with error handling"


@pytest.mark.asyncio
async def test_max_review_iterations_accepts_with_warning(tmp_path):
    """After 2 rejections, the work is accepted with _review_warning."""
    result = await _run_review(
        [
            _tool_response("list_files", call_id="c0"),
            _finish_response(call_id="c1"),
            _reject_response(critique="Issue 1"),
            _finish_response(call_id="c2"),
            _reject_response(call_id="r2", critique="Issue 2"),
        ],
        tmp_path=tmp_path,
        max_review_iterations=2,
    )
    assert result.payload["action"] == "final"
    assert "_review_warning" in result.payload


@pytest.mark.asyncio
async def test_reviewer_can_inspect_workspace(tmp_path):
    """Reviewer calls read_file before approving."""
    read_file_response = LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            call_id="r0", tool_name="read_file",
            arguments={"path": "nonexistent.py"},
        )],
    )
    result = await _run_review(
        [
            _tool_response("list_files", call_id="c0"),
            _finish_response(call_id="c1"),
            read_file_response,
            _approve_response(call_id="r1"),
        ],
        tmp_path=tmp_path,
    )
    assert result.payload["action"] == "final"
    assert result.payload["summary"] == "Done"


@pytest.mark.asyncio
async def test_reviewer_plain_text_treated_as_approval(tmp_path):
    """Reviewer returns plain text (no tool call) -> treated as implicit approval."""
    plain_text_review = LLMResponse(
        content="Everything looks fine.", tool_calls=[],
    )
    result = await _run_review(
        [
            _tool_response("list_files", call_id="c0"),
            _finish_response(call_id="c1"),
            plain_text_review,
        ],
        tmp_path=tmp_path,
    )
    assert result.payload["action"] == "final"
    assert result.payload["summary"] == "Done"


@pytest.mark.asyncio
async def test_review_disabled_with_zero_iterations(tmp_path):
    """max_review_iterations=0 disables review; no review events in runlog."""
    result = await _run_review(
        [
            _tool_response("list_files", call_id="c0"),
            _finish_response(call_id="c1"),
        ],
        tmp_path=tmp_path,
        max_review_iterations=0,
    )
    assert result.payload["action"] == "final"
    events = _read_runlog(result.run_dir)
    assert not any(e["kind"].startswith("review_") for e in events)


def test_reviewer_gets_only_safe_tools():
    """Verify the _SAFE_REVIEWER_TOOLS set excludes shell/write_file."""
    assert "shell" not in _SAFE_REVIEWER_TOOLS
    assert "write_file" not in _SAFE_REVIEWER_TOOLS
    assert "web_search" not in _SAFE_REVIEWER_TOOLS
    assert "fetch_url" not in _SAFE_REVIEWER_TOOLS
    assert "cross_run_search" not in _SAFE_REVIEWER_TOOLS
    # Safe tools should include:
    assert "read_file" in _SAFE_REVIEWER_TOOLS
    assert "list_files" in _SAFE_REVIEWER_TOOLS
    assert "run_tests" in _SAFE_REVIEWER_TOOLS


@pytest.mark.asyncio
async def test_review_events_in_runlog(tmp_path):
    """Verify review_start + review_approved events are logged."""
    result = await _run_review(
        [
            _tool_response("list_files", call_id="c0"),
            _finish_response(call_id="c1"),
            _approve_response(),
        ],
        tmp_path=tmp_path,
    )
    events = _read_runlog(result.run_dir)
    review_starts = [e for e in events if e["kind"] == "review_start"]
    review_approveds = [e for e in events if e["kind"] == "review_approved"]
    assert len(review_starts) == 1
    assert review_starts[0]["payload"]["specialist_id"] == "engineering"
    assert len(review_approveds) == 1
    assert review_approveds[0]["payload"]["review_iteration"] == 0
