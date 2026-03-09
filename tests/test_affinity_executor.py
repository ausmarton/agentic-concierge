"""Tests for affinity-aware graph execution.

Verifies:
- Each node gets model assigned based on role
- Reviewer gets different model than doer (must_differ_from)
- Model handles released after execution
- Model handles released even on failure
- Events emitted for model assignment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from agentic_concierge.application.affinity_executor import (
    AffinityContext,
    execute_graph_with_affinity,
)
from agentic_concierge.application.agent_roles import ModelAssignment
from agentic_concierge.application.task_graph import TaskGraph, TaskNode


# ---------------------------------------------------------------------------
# Mock runtime
# ---------------------------------------------------------------------------


@dataclass
class MockHandle:
    model_id: str
    released: bool = False


class MockRuntime:
    """Runtime that assigns models based on capability requirements."""

    def __init__(self, model_map: Optional[Dict[str, str]] = None) -> None:
        # capability key → model_id
        self._model_map = model_map or {}
        self._default = "qwen2.5:7b"
        self.acquired: List[MockHandle] = []
        self.released: List[MockHandle] = []

    async def acquire(
        self,
        requirements: Dict[str, float],
        *,
        prefer_model: Optional[str] = None,
        require_tool_calling: bool = True,
        exclude_models: Optional[List[str]] = None,
        timeout_s: float = 30.0,
    ) -> MockHandle:
        # Pick model based on first matching capability
        model_id = self._default
        for cap in requirements:
            if cap in self._model_map:
                model_id = self._model_map[cap]
                break

        if exclude_models and model_id in exclude_models:
            # Try fallback
            for cap, mid in self._model_map.items():
                if mid not in exclude_models:
                    model_id = mid
                    break
            else:
                model_id = self._default

        handle = MockHandle(model_id=model_id)
        self.acquired.append(handle)
        return handle

    async def release(self, handle: Any) -> None:
        handle.released = True
        self.released.append(handle)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _simple_work(
    node: TaskNode,
    graph: TaskGraph,
    assignment: ModelAssignment,
) -> Dict[str, Any]:
    return {"answer": node.description, "model": assignment.model_id}


async def _failing_work(
    node: TaskNode,
    graph: TaskGraph,
    assignment: ModelAssignment,
) -> Dict[str, Any]:
    raise RuntimeError(f"Work failed: {node.description}")


class EventCollector:
    def __init__(self):
        self.events: List[tuple[str, Dict[str, Any]]] = []

    def __call__(self, kind: str, payload: Dict[str, Any]) -> None:
        self.events.append((kind, payload))

    def kinds(self) -> List[str]:
        return [k for k, _ in self.events]

    def get(self, kind: str) -> List[Dict[str, Any]]:
        return [p for k, p in self.events if k == kind]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestModelAssignment:

    @pytest.mark.asyncio
    async def test_each_node_gets_model(self):
        g = TaskGraph.from_root("Root", node_id="r")
        g.add_child("r", "Write Python code", node_id="code",
                     required_capabilities=["code_python"])
        g.add_child("r", "Search the web", node_id="web",
                     required_capabilities=["web_comprehension"])
        g.transition("r", "decomposing")
        g.transition("r", "critiqued")

        runtime = MockRuntime({
            "code_python": "qwen2.5-coder:7b",
            "web_comprehension": "qwen2.5:7b",
        })

        result = await execute_graph_with_affinity(
            g, runtime, _simple_work,
        )

        assert result.completed is True
        # Code task should use coder model
        assert result.results["code"]["model"] == "qwen2.5-coder:7b"
        # Web task should use general model
        assert result.results["web"]["model"] == "qwen2.5:7b"

    @pytest.mark.asyncio
    async def test_single_node_works(self):
        g = TaskGraph.from_root("Root", node_id="r")
        g.add_child("r", "Do it", node_id="t",
                     required_capabilities=["instruction_following"])
        g.transition("r", "decomposing")
        g.transition("r", "critiqued")

        runtime = MockRuntime()
        result = await execute_graph_with_affinity(g, runtime, _simple_work)

        assert result.completed is True
        assert len(runtime.acquired) == 1


class TestHandleRelease:

    @pytest.mark.asyncio
    async def test_handles_released_after_success(self):
        g = TaskGraph.from_root("Root", node_id="r")
        g.add_child("r", "Task", node_id="t")
        g.transition("r", "decomposing")
        g.transition("r", "critiqued")

        runtime = MockRuntime()
        await execute_graph_with_affinity(g, runtime, _simple_work)

        assert len(runtime.acquired) == 1
        assert len(runtime.released) == 1
        assert runtime.acquired[0].released is True

    @pytest.mark.asyncio
    async def test_handles_released_after_failure(self):
        g = TaskGraph.from_root("Root", node_id="r")
        g.add_child("r", "Task", node_id="t")
        g.transition("r", "decomposing")
        g.transition("r", "critiqued")

        runtime = MockRuntime()
        await execute_graph_with_affinity(g, runtime, _failing_work)

        # Handle should be released even though work failed
        assert len(runtime.released) == 1
        assert runtime.acquired[0].released is True

    @pytest.mark.asyncio
    async def test_all_handles_released_multi_node(self):
        g = TaskGraph.from_root("Root", node_id="r")
        g.add_child("r", "A", node_id="a")
        g.add_child("r", "B", node_id="b")
        g.add_child("r", "C", node_id="c")
        g.transition("r", "decomposing")
        g.transition("r", "critiqued")

        runtime = MockRuntime()
        await execute_graph_with_affinity(g, runtime, _simple_work)

        assert len(runtime.acquired) == 3
        assert len(runtime.released) == 3
        assert all(h.released for h in runtime.acquired)


class TestEvents:

    @pytest.mark.asyncio
    async def test_model_assigned_event(self):
        g = TaskGraph.from_root("Root", node_id="r")
        g.add_child("r", "Code task", node_id="t",
                     required_capabilities=["code_python"])
        g.transition("r", "decomposing")
        g.transition("r", "critiqued")

        events = EventCollector()
        runtime = MockRuntime({"code_python": "coder:7b"})
        await execute_graph_with_affinity(
            g, runtime, _simple_work, on_event=events,
        )

        assigned = events.get("model_assigned")
        assert len(assigned) == 1
        assert assigned[0]["node_id"] == "t"
        assert assigned[0]["role"] == "coder"
        assert assigned[0]["model_id"] == "coder:7b"


class TestAffinityContext:

    def test_context_tracks_role_model_map(self):
        ctx = AffinityContext()
        ctx.role_model_map["coder"] = "qwen2.5-coder:7b"
        ctx.role_model_map["researcher"] = "qwen2.5:7b"
        assert ctx.role_model_map["coder"] == "qwen2.5-coder:7b"

    def test_context_tracks_node_assignments(self):
        ctx = AffinityContext()
        assignment = ModelAssignment(
            role="coder",
            model_id="qwen2.5-coder:7b",
            handle=MockHandle("qwen2.5-coder:7b"),
            requirements={"code_python": 0.8},
        )
        ctx.node_assignments["node1"] = assignment
        assert ctx.node_assignments["node1"].model_id == "qwen2.5-coder:7b"
