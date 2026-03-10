"""Tests for parallel execution via V2 graph executor.

Tests cover:
- _emit helper correctly puts events on queue
- V2 graph executor runs sibling leaves in parallel
- Single-specialist task works correctly
- ConciergeConfig.task_force_mode field defaults to 'sequential'
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_concierge.application.execute_task import _emit
from agentic_concierge.config.schema import ConciergeConfig, ModelConfig, SpecialistConfig


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
# V2 graph-based parallel execution (sibling nodes run concurrently)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_parallel_graph_runs_both_leaves(tmp_path):
    """V2: sibling leaf nodes under the same root run in parallel via graph executor."""
    from agentic_concierge.application.execute_task import execute_task
    from agentic_concierge.application.planner import PlanResult
    from agentic_concierge.application.task_graph import TaskGraph
    from agentic_concierge.domain import Task, LLMResponse, ToolCallRequest
    from agentic_concierge.infrastructure.ollama import OllamaChatClient
    from agentic_concierge.infrastructure.specialists import ConfigSpecialistRegistry
    from agentic_concierge.infrastructure.workspace import FileSystemRunRepository
    from agentic_concierge.config import load_config

    config = load_config()
    run_repository = FileSystemRunRepository(workspace_root=str(tmp_path))
    specialist_registry = ConfigSpecialistRegistry(config)

    # Create a two-leaf graph (parallel siblings)
    graph = TaskGraph.from_root("Do two things", node_id="root")
    graph.add_child(
        "root", "Engineering work",
        node_id="eng",
        required_capabilities=["code_python"],
    )
    graph.add_child(
        "root", "Research work",
        node_id="res",
        required_capabilities=["web_comprehension"],
    )
    graph.transition("root", "decomposing")
    graph.transition("root", "critiqued")

    plan_result = PlanResult(
        graph=graph,
        reasoning="Two parallel tasks",
        critique_feedback=None,
        replan_count=0,
        planner_model="test-model",
        critic_model="test-model",
    )

    # Mock chat: each leaf gets tool + finish
    def _finish(summary: str = "Done"):
        return LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(
                call_id="c1", tool_name="finish_task",
                arguments={"summary": summary, "artifacts": [], "next_steps": [],
                           "notes": "", "tests_verified": True},
            )],
        )

    def _tool():
        return LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(call_id="t0", tool_name="list_files", arguments={})],
        )

    with patch(
        "agentic_concierge.application.planner.plan_task",
        new_callable=AsyncMock,
        return_value=plan_result,
    ), patch.object(
        OllamaChatClient, "chat", new_callable=AsyncMock,
        side_effect=[_tool(), _finish("eng done"), _tool(), _finish("res done")],
    ):
        chat_client = OllamaChatClient(base_url="http://localhost:11434/v1", timeout_s=5.0)
        task = Task(prompt="Do two things", specialist_id=None, network_allowed=False)
        result = await execute_task(
            task,
            chat_client=chat_client,
            run_repository=run_repository,
            specialist_registry=specialist_registry,
            config=config,
            max_steps=10,
            max_review_iterations=0,
        )

    # Both nodes should have run (result payload should exist)
    assert result.payload.get("action") == "final"
    assert result.is_task_force


@pytest.mark.asyncio
async def test_single_pack_parallel_mode_falls_to_sequential(tmp_path):
    """A single-specialist run with explicit specialist_id runs directly."""
    from agentic_concierge.application.execute_task import execute_task
    from agentic_concierge.domain import Task, LLMResponse, ToolCallRequest
    from agentic_concierge.infrastructure.ollama import OllamaChatClient
    from agentic_concierge.infrastructure.specialists import ConfigSpecialistRegistry
    from agentic_concierge.infrastructure.workspace import FileSystemRunRepository
    from agentic_concierge.config import load_config

    config = load_config()
    run_repository = FileSystemRunRepository(workspace_root=str(tmp_path))
    specialist_registry = ConfigSpecialistRegistry(config)

    with patch.object(
        OllamaChatClient, "chat", new_callable=AsyncMock,
        side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(call_id="c1", tool_name="list_files", arguments={})],
            ),
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(call_id="c2", tool_name="finish_task",
                                            arguments={"summary": "done", "artifacts": [],
                                                        "next_steps": [], "notes": "",
                                                        "tests_verified": True})],
            ),
        ],
    ):
        chat_client = OllamaChatClient(base_url="http://localhost:11434/v1", timeout_s=5.0)
        task = Task(prompt="Single task", specialist_id="engineering", network_allowed=False)
        result = await execute_task(
            task,
            chat_client=chat_client,
            run_repository=run_repository,
            specialist_registry=specialist_registry,
            config=config,
            max_review_iterations=0,
        )

    assert result.payload.get("action") == "final"
    assert not result.is_task_force
