"""Tests for leaf_adapter — bridging graph executor to _execute_pack_loop."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_concierge.application.agent_roles import ModelAssignment
from agentic_concierge.application.leaf_adapter import (
    LeafExecutionContext,
    _build_node_messages,
    _node_to_specialist_id,
    build_leaf_executor,
)
from agentic_concierge.application.task_graph import TaskGraph, TaskNode
from agentic_concierge.config import ConciergeConfig, ModelConfig


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _make_config() -> ConciergeConfig:
    from agentic_concierge.config.schema import SpecialistConfig
    return ConciergeConfig(
        models={
            "quality": ModelConfig(
                base_url="http://localhost:11434",
                model="qwen2.5:7b",
                backend="ollama",
            ),
        },
        specialists={
            "research": SpecialistConfig(description="Research specialist"),
            "engineering": SpecialistConfig(description="Engineering specialist"),
        },
    )


def _make_handle(model_id: str = "qwen2.5:7b") -> MagicMock:
    """Create a mock ModelHandle."""
    handle = MagicMock()
    handle.model_id = model_id
    handle.chat_client = AsyncMock()
    handle.slot = MagicMock()
    handle.slot.model_id = model_id
    handle.slot.backend = "ollama"
    return handle


def _make_assignment(
    role: str = "researcher",
    model_id: str = "qwen2.5:7b",
) -> ModelAssignment:
    return ModelAssignment(
        role=role,
        model_id=model_id,
        handle=_make_handle(model_id),
        requirements={"instruction_following": 0.7},
    )


def _make_registry() -> MagicMock:
    """Mock SpecialistRegistry that returns a mock pack."""
    registry = MagicMock()
    pack = MagicMock()
    pack.system_prompt = "You are a helpful assistant."
    pack.specialist_id = "research"
    pack.tool_definitions = []
    pack.finish_tool_name = "finish_task"
    pack.finish_required_fields = []
    pack.aopen = AsyncMock()
    pack.aclose = AsyncMock()
    pack.execute_tool = AsyncMock(return_value={"ok": True})
    registry.get_pack.return_value = pack
    return registry


def _make_run_repository() -> MagicMock:
    repo = MagicMock()
    repo.append_event = MagicMock()
    return repo


def _make_context(**overrides) -> LeafExecutionContext:
    defaults = dict(
        specialist_registry=_make_registry(),
        run_repository=_make_run_repository(),
        run_id=MagicMock(value="test-run-001"),
        workspace_path="/tmp/workspace",
        config=_make_config(),
        network_allowed=True,
        max_steps=10,
        max_review_iterations=0,
        max_escalations=0,
    )
    defaults.update(overrides)
    return LeafExecutionContext(**defaults)


# ---------------------------------------------------------------------------
# _node_to_specialist_id
# ---------------------------------------------------------------------------


class TestNodeToSpecialistId:
    def test_code_capabilities_resolve_to_engineering(self):
        node = TaskNode(
            id="n1", description="Write Python code",
            required_capabilities=["code_python"],
        )
        sid = _node_to_specialist_id(node)
        assert sid in ("engineering", "dynamic")

    def test_web_capabilities_resolve_to_research(self):
        node = TaskNode(
            id="n2", description="Search the web",
            required_capabilities=["web_comprehension", "summarisation"],
        )
        sid = _node_to_specialist_id(node)
        assert sid in ("research", "enterprise_research", "dynamic")

    def test_no_capabilities_defaults_to_research(self):
        node = TaskNode(id="n3", description="Do something")
        sid = _node_to_specialist_id(node)
        assert sid == "research"

    def test_required_tools_only_uses_dynamic(self):
        node = TaskNode(
            id="n4", description="Run shell",
            required_tools=["shell", "read_file"],
        )
        sid = _node_to_specialist_id(node)
        assert sid == "dynamic"


# ---------------------------------------------------------------------------
# _build_node_messages
# ---------------------------------------------------------------------------


class TestBuildNodeMessages:
    def test_simple_node_produces_system_and_user_messages(self):
        graph = TaskGraph.from_root("Build an app", node_id="root")
        child = graph.add_child("root", "Write the code", node_id="c1")
        msgs = _build_node_messages(child, graph, "System prompt here")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "System prompt here"
        assert msgs[1]["role"] == "user"
        assert "Build an app" in msgs[1]["content"]
        assert "Write the code" in msgs[1]["content"]

    def test_context_from_preceding_sibling(self):
        graph = TaskGraph.from_root("Build an app", node_id="root")
        graph.add_child("root", "Step 1: plan", node_id="c1")
        c2 = graph.add_child("root", "Step 2: code", node_id="c2")
        # Mark c1 as done with a result
        graph.transition("c1", "executing")
        graph.mark_done("c1", {"plan": "use Flask"})

        msgs = _build_node_messages(c2, graph, "System")
        user_msg = msgs[1]["content"]
        assert "Step 1: plan" in user_msg
        assert "Flask" in user_msg
        assert "Context from previous steps" in user_msg

    def test_root_node_no_context(self):
        graph = TaskGraph.from_root("Simple question", node_id="root")
        root = graph.root
        msgs = _build_node_messages(root, graph, "System")
        assert len(msgs) == 2
        assert "Context from previous steps" not in msgs[1]["content"]

    def test_only_completed_siblings_included(self):
        graph = TaskGraph.from_root("Multi-step", node_id="root")
        graph.add_child("root", "Step A", node_id="a")
        graph.add_child("root", "Step B", node_id="b")
        graph.add_child("root", "Step C", node_id="c")
        # Only mark A as done
        graph.transition("a", "executing")
        graph.mark_done("a", {"result": "done_a"})
        # B is executing (not done)
        graph.transition("b", "executing")

        c_node = graph.nodes["c"]
        msgs = _build_node_messages(c_node, graph, "System")
        user_msg = msgs[1]["content"]
        # Should include A's result
        assert "done_a" in user_msg
        # Should NOT include B (not done yet)
        # B appears before C in children, but since B is not done, it shouldn't be in context


# ---------------------------------------------------------------------------
# build_leaf_executor
# ---------------------------------------------------------------------------


class TestBuildLeafExecutor:
    @pytest.mark.asyncio
    async def test_executor_calls_pack_loop(self):
        mock_loop = AsyncMock(return_value={"answer": "42", "action": "final"})
        ctx = _make_context(pack_loop_fn=mock_loop)
        executor = build_leaf_executor(ctx)

        graph = TaskGraph.from_root("Test task", node_id="root")
        child = graph.add_child(
            "root", "Sub-task",
            node_id="c1",
            required_capabilities=["instruction_following"],
        )
        assignment = _make_assignment()

        result = await executor(child, graph, assignment)

        assert result == {"answer": "42", "action": "final"}
        mock_loop.assert_called_once()

        # Verify key arguments passed to _execute_pack_loop
        call_kwargs = mock_loop.call_args[1]
        assert call_kwargs["run_id"] == ctx.run_id
        assert call_kwargs["workspace_path"] == ctx.workspace_path
        assert call_kwargs["max_steps"] == ctx.max_steps
        assert call_kwargs["max_review_iterations"] == 0
        assert call_kwargs["max_escalations"] == 0

    @pytest.mark.asyncio
    async def test_executor_stores_result_in_context(self):
        mock_payload = {"answer": "done"}
        mock_loop = AsyncMock(return_value=mock_payload)
        ctx = _make_context(pack_loop_fn=mock_loop)
        executor = build_leaf_executor(ctx)

        graph = TaskGraph.from_root("Test", node_id="root")
        child = graph.add_child("root", "Do work", node_id="c1")
        assignment = _make_assignment()

        await executor(child, graph, assignment)

        assert ctx.completed_results["c1"] == mock_payload

    @pytest.mark.asyncio
    async def test_executor_uses_handle_chat_client(self):
        mock_loop = AsyncMock(return_value={})
        ctx = _make_context(pack_loop_fn=mock_loop)
        executor = build_leaf_executor(ctx)

        graph = TaskGraph.from_root("Test", node_id="root")
        child = graph.add_child("root", "Work", node_id="c1")
        assignment = _make_assignment(model_id="phi-4:14b")

        await executor(child, graph, assignment)

        call_kwargs = mock_loop.call_args[1]
        # The chat_client should be the one from the handle
        assert call_kwargs["chat_client"] == assignment.handle.chat_client
        # The model_cfg should reference the assignment's model
        assert call_kwargs["model_cfg"].model == "phi-4:14b"

    @pytest.mark.asyncio
    async def test_executor_emits_event(self):
        queue = asyncio.Queue()
        mock_loop = AsyncMock(return_value={})
        ctx = _make_context(event_queue=queue, pack_loop_fn=mock_loop)
        executor = build_leaf_executor(ctx)

        graph = TaskGraph.from_root("Test", node_id="root")
        child = graph.add_child("root", "Work", node_id="c1")
        assignment = _make_assignment()

        await executor(child, graph, assignment)

        # Should have emitted a node_execution_start event
        event = queue.get_nowait()
        assert event["kind"] == "node_execution_start"
        assert event["data"]["node_id"] == "c1"

    @pytest.mark.asyncio
    async def test_executor_passes_step_prefix(self):
        mock_loop = AsyncMock(return_value={})
        ctx = _make_context(pack_loop_fn=mock_loop)
        executor = build_leaf_executor(ctx)

        graph = TaskGraph.from_root("Test", node_id="root")
        child = graph.add_child("root", "Work", node_id="abc123")
        assignment = _make_assignment()

        await executor(child, graph, assignment)

        call_kwargs = mock_loop.call_args[1]
        assert call_kwargs["step_prefix"] == "node_abc123_"


# ---------------------------------------------------------------------------
# LeafExecutionContext
# ---------------------------------------------------------------------------


class TestLeafExecutionContext:
    def test_defaults(self):
        ctx = _make_context()
        assert ctx.max_steps == 10
        assert ctx.max_review_iterations == 0
        assert ctx.max_escalations == 0
        assert ctx.completed_results == {}

    def test_completed_results_tracked(self):
        ctx = _make_context()
        ctx.completed_results["n1"] = {"answer": "42"}
        assert ctx.completed_results["n1"]["answer"] == "42"
