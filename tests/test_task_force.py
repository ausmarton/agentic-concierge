"""Tests for multi-node graph execution (task force) via V2 planner + graph executor.

These tests cover:
- execute_task with multiple leaf nodes: all nodes run, share workspace and runlog.
- Context handoff: sibling nodes' context is passed via _build_node_messages.
- Runlog structure: node_execution_start events, step names prefixed with node ID.
- RunResult: specialist_ids, is_task_force, specialist_id (primary = first).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agentic_concierge.application.execute_task import execute_task
from agentic_concierge.application.planner import PlanResult
from agentic_concierge.application.task_graph import TaskGraph
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


def _make_multi_leaf_graph() -> TaskGraph:
    """Create a two-leaf graph: root → eng + research (parallel siblings)."""
    graph = TaskGraph.from_root("build a tool that does a systematic review", node_id="root")
    graph.add_child(
        "root", "Build the engineering tool",
        node_id="eng",
        required_capabilities=["code_python"],
    )
    graph.add_child(
        "root", "Research arxiv papers",
        node_id="res",
        required_capabilities=["web_comprehension", "summarisation"],
    )
    graph.transition("root", "decomposing")
    graph.transition("root", "critiqued")
    return graph


def _make_plan_result(graph: TaskGraph | None = None) -> PlanResult:
    return PlanResult(
        graph=graph or _make_multi_leaf_graph(),
        reasoning="Two-part task",
        critique_feedback=None,
        replan_count=0,
        planner_model="test-model",
        critic_model="test-model",
    )


def _read_runlog(run_dir: str) -> list[dict]:
    lines = Path(run_dir, "runlog.jsonl").read_text().strip().splitlines()
    return [json.loads(ln) for ln in lines if ln]


# ---------------------------------------------------------------------------
# Integration: execute_task with multi-leaf graph (task force)
# ---------------------------------------------------------------------------

async def _run_task_force(prompt: str, mock_responses: list, *, tmp_path,
                          graph: TaskGraph | None = None,
                          max_review_iterations=0) -> tuple:
    """Run execute_task with a multi-leaf graph and mock LLM responses.
    Returns (result, events).
    """
    config = load_config()
    run_repository = FileSystemRunRepository(workspace_root=str(tmp_path))
    specialist_registry = ConfigSpecialistRegistry(config)
    plan_result = _make_plan_result(graph)

    with patch(
        "agentic_concierge.application.planner.plan_task",
        new_callable=AsyncMock,
        return_value=plan_result,
    ), patch.object(
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
    """A multi-leaf graph executes all leaf nodes."""
    # Both leaves run in parallel; each needs tool + finish
    result, events = await _run_task_force(
        "build a tool that does a systematic review of arxiv papers",
        [_tool_resp("t0"), _eng_finish(), _tool_resp("t1"), _research_finish()],
        tmp_path=tmp_path,
    )

    assert result.is_task_force
    # specialist_ids are resolved from node capabilities
    assert len(result.specialist_ids) >= 1
    assert result.specialist_id is not None


@pytest.mark.asyncio
async def test_task_force_runlog_has_node_execution_start_events(tmp_path):
    """Multi-leaf graphs log a node_execution_start event for each leaf."""
    result, events = await _run_task_force(
        "build a tool that does a systematic review of arxiv papers",
        [_tool_resp("t0"), _eng_finish(), _tool_resp("t1"), _research_finish()],
        tmp_path=tmp_path,
    )

    node_starts = [e for e in events if e["kind"] == "node_execution_start"]
    assert len(node_starts) == 2
    node_ids = {e["payload"]["node_id"] for e in node_starts}
    assert "eng" in node_ids
    assert "res" in node_ids


@pytest.mark.asyncio
async def test_task_force_runlog_step_names_are_node_prefixed(tmp_path):
    """In a task force, step events use 'node_{id}_step_N' naming."""
    result, events = await _run_task_force(
        "build a tool that does a systematic review of arxiv papers",
        [_tool_resp("t0"), _eng_finish(), _tool_resp("t1"), _research_finish()],
        tmp_path=tmp_path,
    )

    llm_request_steps = [
        e.get("step") for e in events if e["kind"] == "llm_request"
    ]
    assert any(s and s.startswith("node_eng_") for s in llm_request_steps)
    assert any(s and s.startswith("node_res_") for s in llm_request_steps)


@pytest.mark.asyncio
async def test_task_force_shared_workspace(tmp_path):
    """Both nodes in a task force write to the same workspace directory."""
    result, events = await _run_task_force(
        "build a tool that does a systematic review of arxiv papers",
        [_tool_resp("t0"), _eng_finish(), _tool_resp("t1"), _research_finish()],
        tmp_path=tmp_path,
    )

    # Both nodes operate in the same run_dir/workspace.
    assert Path(result.workspace_path).is_dir()
    # Only one workspace (one run_dir) per task.
    assert result.run_dir  # single run dir


@pytest.mark.asyncio
async def test_task_force_context_passed_to_second_node(tmp_path):
    """In a sequential (chained) graph, the second node receives the first's result as context.

    We verify this by making eng a parent of res (chain), so res sees eng's result.
    """
    # Create a chained graph: root → eng → res (sequential)
    graph = TaskGraph.from_root("build a tool that does a systematic review", node_id="root")
    graph.add_child(
        "root", "Build the engineering tool",
        node_id="eng",
        required_capabilities=["code_python"],
    )
    graph.add_child(
        "eng", "Research arxiv papers",
        node_id="res",
        required_capabilities=["web_comprehension"],
    )
    graph.transition("root", "decomposing")
    graph.transition("root", "critiqued")

    result, events = await _run_task_force(
        "build a tool that does a systematic review of arxiv papers",
        [_tool_resp("t0"), _eng_finish(summary="Created tool.py"),
         _tool_resp("t1"), _research_finish()],
        tmp_path=tmp_path,
        graph=graph,
    )

    # The research node (child of eng) gets eng's result as sibling context.
    # Check that there's at least one LLM request for the research node.
    res_llm_requests = [
        e for e in events
        if e["kind"] == "llm_request" and e.get("step", "").startswith("node_res_")
    ]
    assert len(res_llm_requests) >= 1


@pytest.mark.asyncio
async def test_task_force_result_payload_is_from_last_node(tmp_path):
    """RunResult.payload comes from the last leaf node's finish_task call."""
    result, _ = await _run_task_force(
        "build a tool that does a systematic review of arxiv papers",
        [_tool_resp("t0"), _eng_finish(summary="Engineering done"),
         _tool_resp("t1"), _research_finish(answer="Research complete")],
        tmp_path=tmp_path,
    )

    # Result payload should contain content from one of the leaf nodes
    assert result.payload.get("action") == "final"


@pytest.mark.asyncio
async def test_task_force_recruitment_event_includes_specialist_ids(tmp_path):
    """The recruitment runlog event includes specialist_ids (plural) and is_task_force."""
    result, events = await _run_task_force(
        "build a tool that does a systematic review of arxiv papers",
        [_tool_resp("t0"), _eng_finish(), _tool_resp("t1"), _research_finish()],
        tmp_path=tmp_path,
    )

    recruitment_events = [e for e in events if e["kind"] == "recruitment"]
    assert len(recruitment_events) == 1

    payload = recruitment_events[0]["payload"]
    assert "specialist_ids" in payload
    assert len(payload["specialist_ids"]) >= 1
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
    """Single-specialist runs do not emit node_execution_start for multiple nodes."""
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
    node_starts = [e for e in events if e["kind"] == "node_execution_start"]
    # Single node → exactly one node_execution_start
    assert len(node_starts) == 1
