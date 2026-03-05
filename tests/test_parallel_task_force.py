"""Tests for P8-1: parallel task force execution.

Tests cover:
- _merge_parallel_payloads combines summaries and pack_results correctly
- Exceptions in payloads produce error dicts, not crashes
- _run_task_force_parallel calls _execute_pack_loop for each specialist
- task_force_mode='parallel' triggers the parallel path in execute_task
- task_force_mode='sequential' (default) still works as before
- Single-specialist task with mode='parallel' falls through to sequential
- ConciergeConfig.task_force_mode field defaults to 'sequential'
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_concierge.application.execute_task import (
    _emit,
    _merge_parallel_payloads,
)
from agentic_concierge.config.schema import ConciergeConfig, ModelConfig, SpecialistConfig


# ---------------------------------------------------------------------------
# _merge_parallel_payloads tests
# ---------------------------------------------------------------------------

def test_merge_two_payloads_combines_pack_results():
    payloads = [
        {"action": "final", "summary": "Engineering done", "artifacts": [], "next_steps": []},
        {"action": "final", "summary": "Research done", "artifacts": [], "next_steps": []},
    ]
    result = _merge_parallel_payloads(payloads, ["engineering", "research"])

    assert result["action"] == "final"
    assert result["pack_results"]["engineering"]["summary"] == "Engineering done"
    assert result["pack_results"]["research"]["summary"] == "Research done"


def test_merge_summaries_joined_with_pipe():
    payloads = [
        {"action": "final", "summary": "A", "artifacts": [], "next_steps": []},
        {"action": "final", "summary": "B", "artifacts": [], "next_steps": []},
    ]
    result = _merge_parallel_payloads(payloads, ["a", "b"])
    assert result["summary"] == "a: A | b: B"


def test_merge_empty_summary_skipped():
    payloads = [
        {"action": "final", "summary": "", "artifacts": [], "next_steps": []},
        {"action": "final", "summary": "B done", "artifacts": [], "next_steps": []},
    ]
    result = _merge_parallel_payloads(payloads, ["a", "b"])
    # Only "b: B done" — empty summary for "a" is omitted
    assert "a:" not in result["summary"]
    assert "b: B done" in result["summary"]


def test_merge_executive_summary_fallback():
    payloads = [
        {"action": "final", "executive_summary": "Exec sum", "artifacts": [], "next_steps": []},
    ]
    result = _merge_parallel_payloads(payloads, ["research"])
    assert "Exec sum" in result["summary"]


def test_merge_exception_produces_error_dict():
    exc = RuntimeError("pack exploded")
    payloads: list = [
        {"action": "final", "summary": "OK", "artifacts": [], "next_steps": []},
        exc,
    ]
    result = _merge_parallel_payloads(payloads, ["engineering", "research"])

    assert "error" in result["pack_results"]["research"]
    assert result["pack_results"]["research"]["error_type"] == "RuntimeError"
    assert "pack exploded" in result["summary"]


def test_merge_all_exceptions_no_crash():
    payloads: list = [ValueError("a"), RuntimeError("b")]
    result = _merge_parallel_payloads(payloads, ["eng", "res"])
    assert "error" in result["pack_results"]["eng"]
    assert "error" in result["pack_results"]["res"]
    assert result["action"] == "final"


def test_merge_no_summaries_uses_default():
    payloads = [
        {"action": "final", "summary": "", "artifacts": [], "next_steps": []},
    ]
    result = _merge_parallel_payloads(payloads, ["engineering"])
    assert result["summary"] == "Parallel task force completed."


# ---------------------------------------------------------------------------
# _emit helper tests
# ---------------------------------------------------------------------------

def test_emit_puts_event_on_queue():
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    _emit(queue, "tool_call", {"tool": "shell"}, step="step_0")
    item = queue.get_nowait()
    assert item["kind"] == "tool_call"
    assert item["data"]["tool"] == "shell"
    assert item["step"] == "step_0"


def test_emit_noop_when_queue_none():
    # Should not raise
    _emit(None, "llm_request", {"step": 0})


def test_emit_drops_when_queue_full():
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    queue.put_nowait({"kind": "existing"})
    # Should not raise even though queue is full
    _emit(queue, "tool_call", {"tool": "shell"})
    # Original item still there, new one dropped
    assert queue.qsize() == 1


# ---------------------------------------------------------------------------
# ConciergeConfig.task_force_mode field
# ---------------------------------------------------------------------------

def test_task_force_mode_defaults_to_sequential():
    cfg = ConciergeConfig(
        models={"fast": ModelConfig(base_url="http://x/v1", model="m")},
        specialists={"eng": SpecialistConfig(description="d", workflow="engineering")},
    )
    assert cfg.task_force_mode == "sequential"


def test_task_force_mode_accepts_parallel():
    cfg = ConciergeConfig(
        models={"fast": ModelConfig(base_url="http://x/v1", model="m")},
        specialists={"eng": SpecialistConfig(description="d", workflow="engineering")},
        task_force_mode="parallel",
    )
    assert cfg.task_force_mode == "parallel"


# ---------------------------------------------------------------------------
# execute_task parallel path (integration-level, mocked LLM)
# ---------------------------------------------------------------------------

def _make_stub_pack(specialist_id: str, summary: str) -> MagicMock:
    pack = MagicMock()
    pack.specialist_id = specialist_id
    pack.system_prompt = f"You are {specialist_id}."
    pack.finish_tool_name = "finish_task"
    pack.finish_required_fields = ["summary"]
    pack.tool_definitions = [
        {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Run shell command",
                "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finish_task",
                "description": "Complete the task",
                "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
            },
        },
    ]
    pack.aopen = AsyncMock()
    pack.aclose = AsyncMock()

    # Tool call execution: shell → result; finish_task handled by loop
    async def _execute_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name == "shell":
            return {"stdout": f"{specialist_id} output", "exit_code": 0}
        return {"error": "unknown_tool"}

    pack.execute_tool = _execute_tool
    return pack


@pytest.mark.asyncio
async def test_parallel_task_force_runs_both_packs():
    """Both specialist packs run and their payloads appear in pack_results."""
    from agentic_concierge.application.execute_task import execute_task
    from agentic_concierge.domain import Task, RunId
    from agentic_concierge.config.schema import ConciergeConfig, ModelConfig, SpecialistConfig
    from agentic_concierge.domain.models import LLMResponse, ToolCallRequest

    task = Task(
        prompt="Do two things",
        specialist_id=None,
        model_key="fast",
        network_allowed=False,
    )

    config = ConciergeConfig(
        models={"fast": ModelConfig(base_url="http://x/v1", model="m")},
        specialists={
            "engineering": SpecialistConfig(description="eng", workflow="engineering"),
            "research": SpecialistConfig(description="res", workflow="research"),
        },
        task_force_mode="parallel",
    )

    # Chat client: returns shell call first, then finish_task with summary
    call_counts: Dict[str, int] = {"engineering": 0, "research": 0}

    async def mock_chat(messages, model, tools, **kwargs):
        # Identify which pack is calling based on system message
        sys_msg = messages[0]["content"] if messages else ""
        sid = "engineering" if "engineering" in sys_msg else "research"
        call_counts[sid] = call_counts.get(sid, 0) + 1
        count = call_counts[sid]

        if count == 1:
            # First call: do shell work
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        call_id=f"c1_{sid}",
                        tool_name="shell",
                        arguments={"cmd": "echo hello"},
                    )
                ],
            )
        else:
            # Second call: finish
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        call_id=f"c2_{sid}",
                        tool_name="finish_task",
                        arguments={"summary": f"{sid} completed", "artifacts": [], "next_steps": []},
                    )
                ],
            )

    chat_client = MagicMock()
    chat_client.chat = mock_chat

    # Run repository
    run_id = RunId("test-parallel-run")
    run_repository = MagicMock()
    run_repository.create_run.return_value = (run_id, "/tmp/runs/test-parallel-run", "/tmp/workspace")
    run_repository.append_event = MagicMock()

    # Registry returns stub packs
    def _get_pack(sid, workspace_path, network_allowed):
        pack = _make_stub_pack(sid, f"{sid} completed")
        # Match system prompt to specialist id
        pack.system_prompt = f"You are {sid}."
        return pack

    registry = MagicMock()
    registry.get_pack.side_effect = _get_pack

    # Phase 12: execute_task now calls orchestrate_task (not llm_recruit_specialist directly).
    with patch("agentic_concierge.application.execute_task.orchestrate_task") as mock_orch:
        from agentic_concierge.application.orchestrator import OrchestrationPlan, SpecialistBrief
        import asyncio as _asyncio

        async def _mock_orchestrate(*args, **kwargs):
            return OrchestrationPlan(
                specialist_assignments=[
                    SpecialistBrief("engineering", ""),
                    SpecialistBrief("research", ""),
                ],
                mode="parallel",
                # synthesis_required=False so _merge_parallel_payloads result is preserved
                # (this test checks that pack_results contains both packs' outputs)
                synthesis_required=False,
                reasoning="test",
                routing_method="orchestrator",
                required_capabilities=["code_execution", "systematic_review"],
            )

        mock_orch.side_effect = _mock_orchestrate

        result = await execute_task(
            task,
            chat_client=chat_client,
            run_repository=run_repository,
            specialist_registry=registry,
            config=config,
            max_review_iterations=0,
        )

    assert "pack_results" in result.payload
    assert "engineering" in result.payload["pack_results"]
    assert "research" in result.payload["pack_results"]


@pytest.mark.asyncio
async def test_single_pack_parallel_mode_falls_to_sequential():
    """A single-specialist run with task_force_mode='parallel' runs sequentially."""
    from agentic_concierge.application.execute_task import execute_task
    from agentic_concierge.domain import Task, RunId
    from agentic_concierge.domain.models import LLMResponse, ToolCallRequest

    task = Task(
        prompt="Single pack task",
        specialist_id="engineering",
        model_key="fast",
        network_allowed=False,
    )

    config = ConciergeConfig(
        models={"fast": ModelConfig(base_url="http://x/v1", model="m")},
        specialists={
            "engineering": SpecialistConfig(description="eng", workflow="engineering"),
        },
        task_force_mode="parallel",  # parallel, but only one pack → sequential path
    )

    call_count = 0

    async def mock_chat(messages, model, tools, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(call_id="c1", tool_name="shell", arguments={"cmd": "echo hi"})],
            )
        return LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(call_id="c2", tool_name="finish_task", arguments={"summary": "done", "artifacts": [], "next_steps": []})],
        )

    chat_client = MagicMock()
    chat_client.chat = mock_chat

    run_id = RunId("test-single-parallel")
    run_repository = MagicMock()
    run_repository.create_run.return_value = (run_id, "/tmp/runs/test", "/tmp/ws")
    run_repository.append_event = MagicMock()

    registry = MagicMock()
    registry.get_pack.return_value = _make_stub_pack("engineering", "done")

    result = await execute_task(
        task,
        chat_client=chat_client,
        run_repository=run_repository,
        specialist_registry=registry,
        config=config,
        max_review_iterations=0,
    )

    # Single pack → no pack_results wrapper; just normal payload
    assert result.payload.get("action") == "final"
    assert "pack_results" not in result.payload
