"""Tests for multi-pack task force: sequential execution via orchestrator.

These tests cover:
- execute_task with multiple specialists: both packs run, share workspace and runlog.
- Context handoff: second pack receives first pack's finish payload in messages.
- Runlog structure: pack_start events, step names prefixed with specialist ID.
- RunResult: specialist_ids, is_task_force, specialist_id (primary = first).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agentic_concierge.application.execute_task import execute_task
from agentic_concierge.config import load_config
from agentic_concierge.domain import LLMResponse, Task, ToolCallRequest
from agentic_concierge.infrastructure.ollama import OllamaChatClient
from agentic_concierge.infrastructure.specialists import ConfigSpecialistRegistry
from agentic_concierge.infrastructure.workspace import FileSystemRunRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eng_finish(call_id: str = "c1", summary: str = "Engineering done") -> LLMResponse:
    """Engineering pack finish_task response."""
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            call_id=call_id,
            tool_name="finish_task",
            arguments={
                "summary": summary,
                "artifacts": ["tool.py"],
                "next_steps": ["write docs"],
                "notes": "",
                "tests_verified": True,
            },
        )],
    )


def _research_finish(call_id: str = "c2", answer: str = "Research done") -> LLMResponse:
    """Research pack finish_task response."""
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            call_id=call_id,
            tool_name="finish_task",
            arguments={
                "answer": answer,
            },
        )],
    )


def _tool_resp(call_id: str = "t0") -> LLMResponse:
    """A list_files call used to satisfy the 'prior tool call' structural requirement."""
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(call_id=call_id, tool_name="list_files", arguments={})],
    )


def _create_plan_response(specialist_ids: list[str] | None = None, mode: str = "sequential") -> LLMResponse:
    """Mock orchestrator create_plan response."""
    sids = specialist_ids or ["engineering", "research"]
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            call_id="orch0",
            tool_name="create_plan",
            arguments={
                "assignments": [{"specialist_id": sid, "brief": ""} for sid in sids],
                "mode": mode,
                "synthesis_required": len(sids) > 1,
                "reasoning": "test orchestration",
            },
        )],
    )


def _read_runlog(run_dir: str) -> list[dict]:
    lines = Path(run_dir, "runlog.jsonl").read_text().strip().splitlines()
    return [json.loads(ln) for ln in lines if ln]


# ---------------------------------------------------------------------------
# Integration: execute_task with task force
# ---------------------------------------------------------------------------

async def _run_task_force(prompt: str, mock_responses: list, *, tmp_path,
                          max_review_iterations=0) -> tuple:
    """Run execute_task with the given prompt and mock LLM responses.
    Returns (result, events).
    """
    config = load_config()
    run_repository = FileSystemRunRepository(workspace_root=str(tmp_path))
    specialist_registry = ConfigSpecialistRegistry(config)
    with patch.object(
        OllamaChatClient, "chat", new_callable=AsyncMock, side_effect=mock_responses
    ):
        chat_client = OllamaChatClient(base_url="http://localhost:11434/v1", timeout_s=5.0)
        task = Task(prompt=prompt, specialist_id=None, network_allowed=False)
        result = await execute_task(
            task,
            chat_client=chat_client,
            run_repository=run_repository,
            specialist_registry=specialist_registry,
            config=config,
            max_steps=10,
            max_review_iterations=max_review_iterations,
        )
    events = _read_runlog(result.run_dir)
    return result, events


@pytest.mark.asyncio
async def test_task_force_runs_both_packs(tmp_path):
    """A mixed-capability prompt executes engineering then research packs."""
    result, events = await _run_task_force(
        "build a tool that does a systematic review of arxiv papers",
        [_create_plan_response(), _tool_resp("t0"), _eng_finish(), _tool_resp("t1"), _research_finish()],
        tmp_path=tmp_path,
    )

    assert result.is_task_force
    assert "engineering" in result.specialist_ids
    assert "research" in result.specialist_ids
    assert result.specialist_id == "engineering"  # primary = first


@pytest.mark.asyncio
async def test_task_force_runlog_has_pack_start_events(tmp_path):
    """Multi-pack runs log a pack_start event at the beginning of each pack."""
    result, events = await _run_task_force(
        "build a tool that does a systematic review of arxiv papers",
        [_create_plan_response(), _tool_resp("t0"), _eng_finish(), _tool_resp("t1"), _research_finish()],
        tmp_path=tmp_path,
    )

    pack_starts = [e for e in events if e["kind"] == "pack_start"]
    assert len(pack_starts) == 2

    ids_in_order = [e["payload"]["specialist_id"] for e in pack_starts]
    assert ids_in_order == ["engineering", "research"]
    assert pack_starts[0]["payload"]["pack_index"] == 0
    assert pack_starts[1]["payload"]["pack_index"] == 1


@pytest.mark.asyncio
async def test_task_force_runlog_step_names_are_pack_prefixed(tmp_path):
    """In a task force, step events use '{specialist_id}_step_N' naming."""
    result, events = await _run_task_force(
        "build a tool that does a systematic review of arxiv papers",
        [_create_plan_response(), _tool_resp("t0"), _eng_finish(), _tool_resp("t1"), _research_finish()],
        tmp_path=tmp_path,
    )

    llm_request_steps = [
        e.get("step") for e in events if e["kind"] == "llm_request"
    ]
    assert any(s and s.startswith("engineering_step_") for s in llm_request_steps)
    assert any(s and s.startswith("research_step_") for s in llm_request_steps)


@pytest.mark.asyncio
async def test_task_force_shared_workspace(tmp_path):
    """Both packs in a task force write to the same workspace directory."""
    result, events = await _run_task_force(
        "build a tool that does a systematic review of arxiv papers",
        [_create_plan_response(), _tool_resp("t0"), _eng_finish(), _tool_resp("t1"), _research_finish()],
        tmp_path=tmp_path,
    )

    # Both packs operate in the same run_dir/workspace.
    assert Path(result.workspace_path).is_dir()
    # Only one workspace (one run_dir) per task.
    assert result.run_dir  # single run dir


@pytest.mark.asyncio
async def test_task_force_context_passed_to_second_pack(tmp_path):
    """The second pack receives the first pack's finish payload as context.

    We verify this by checking that the research pack's first LLM request starts
    with exactly 2 messages (system prompt + user message containing the context
    from engineering's finish payload) before any tool calls are added.
    """
    result, events = await _run_task_force(
        "build a tool that does a systematic review of arxiv papers",
        [_create_plan_response(), _tool_resp("t0"), _eng_finish(summary="Created tool.py"),
         _tool_resp("t1"), _research_finish()],
        tmp_path=tmp_path,
    )

    # The research pack (2nd) gets a user message with context; its first LLM
    # request (before any tool turns are appended) has message_count = 2.
    research_llm_requests = [
        e for e in events
        if e["kind"] == "llm_request" and e.get("step", "").startswith("research_")
    ]
    assert len(research_llm_requests) >= 1
    # message_count should be 2 (system prompt + user message with context).
    assert research_llm_requests[0]["payload"]["message_count"] == 2


@pytest.mark.asyncio
async def test_task_force_result_payload_is_from_last_pack(tmp_path):
    """RunResult.payload comes from the last pack's finish_task call."""
    result, _ = await _run_task_force(
        "build a tool that does a systematic review of arxiv papers",
        [_create_plan_response(), _tool_resp("t0"), _eng_finish(summary="Engineering done"),
         _tool_resp("t1"), _research_finish(answer="Research complete")],
        tmp_path=tmp_path,
    )

    # Research pack uses 'answer' (quick_answer schema).
    assert result.payload.get("answer") == "Research complete"


@pytest.mark.asyncio
async def test_task_force_recruitment_event_includes_specialist_ids(tmp_path):
    """The recruitment runlog event includes specialist_ids (plural) and is_task_force."""
    result, events = await _run_task_force(
        "build a tool that does a systematic review of arxiv papers",
        [_create_plan_response(), _tool_resp("t0"), _eng_finish(), _tool_resp("t1"), _research_finish()],
        tmp_path=tmp_path,
    )

    recruitment_events = [e for e in events if e["kind"] == "recruitment"]
    assert len(recruitment_events) == 1

    payload = recruitment_events[0]["payload"]
    assert "specialist_ids" in payload
    assert set(payload["specialist_ids"]) == {"engineering", "research"}
    assert payload["is_task_force"] is True


@pytest.mark.asyncio
async def test_single_pack_run_is_not_a_task_force(tmp_path):
    """Single-specialist runs have is_task_force=False and specialist_ids of length 1."""
    config = load_config()
    run_repository = FileSystemRunRepository(workspace_root=str(tmp_path))
    specialist_registry = ConfigSpecialistRegistry(config)
    with patch.object(
        OllamaChatClient, "chat", new_callable=AsyncMock,
        side_effect=[_tool_resp(), _eng_finish()],
    ):
        chat_client = OllamaChatClient(base_url="http://localhost:11434/v1", timeout_s=5.0)
        task = Task(prompt="test", specialist_id="engineering", network_allowed=False)
        result = await execute_task(
            task,
            chat_client=chat_client,
            run_repository=run_repository,
            specialist_registry=specialist_registry,
            config=config,
            max_steps=10,
            max_review_iterations=0,
        )

    assert not result.is_task_force
    assert result.specialist_ids == ["engineering"]
    assert result.specialist_id == "engineering"


@pytest.mark.asyncio
async def test_single_pack_runlog_has_no_pack_start_events(tmp_path):
    """pack_start events are only emitted for task forces, not single-pack runs."""
    config = load_config()
    run_repository = FileSystemRunRepository(workspace_root=str(tmp_path))
    specialist_registry = ConfigSpecialistRegistry(config)
    with patch.object(
        OllamaChatClient, "chat", new_callable=AsyncMock,
        side_effect=[_tool_resp(), _eng_finish()],
    ):
        chat_client = OllamaChatClient(base_url="http://localhost:11434/v1", timeout_s=5.0)
        task = Task(prompt="test", specialist_id="engineering", network_allowed=False)
        result = await execute_task(
            task,
            chat_client=chat_client,
            run_repository=run_repository,
            specialist_registry=specialist_registry,
            config=config,
            max_steps=10,
            max_review_iterations=0,
        )

    events = _read_runlog(result.run_dir)
    assert not any(e["kind"] == "pack_start" for e in events)
