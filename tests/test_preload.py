"""Tests for model preloading — preload_hint on LocalModelRuntime and
preload integration in affinity executor.

Verifies:
- preload_hint loads model when memory available
- preload_hint is no-op when model already loaded
- preload_hint skips when insufficient memory
- preload_hint silently ignores failures
- Affinity executor issues preload hints for upcoming siblings
- Protocol compliance (preload_hint on ModelRuntime)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from agentic_concierge.application.agent_roles import ModelAssignment
from agentic_concierge.application.affinity_executor import (
    execute_graph_with_affinity,
)
from agentic_concierge.application.task_graph import TaskGraph, TaskNode


# ---------------------------------------------------------------------------
# Mock backends for LocalModelRuntime preload tests
# ---------------------------------------------------------------------------


@dataclass
class FakeSlot:
    model_id: str
    backend: str
    memory_mb: int = 4000
    refcount: int = 0
    last_used: float = 0.0


class FakeBackend:
    """Fake backend that tracks load/estimate calls."""

    def __init__(self, name: str, models: List[str], mem_per_model: int = 4000):
        self.name = name
        self.base_url = f"http://fake-{name}"
        self._models = models
        self._mem = mem_per_model
        self.loaded: List[str] = []

    async def list_available(self) -> List[str]:
        return list(self._models)

    async def estimate_memory(self, model_id: str) -> int:
        return self._mem

    async def load_model(self, model_id: str) -> FakeSlot:
        self.loaded.append(model_id)
        return FakeSlot(model_id=model_id, backend=self.name, memory_mb=self._mem)

    async def unload_model(self, model_id: str) -> None:
        pass

    async def health_check(self) -> bool:
        return True

    def build_client(self, model_id: str) -> Any:
        return AsyncMock()


# ---------------------------------------------------------------------------
# LocalModelRuntime preload_hint tests
# ---------------------------------------------------------------------------


class TestPreloadHint:
    """Tests for LocalModelRuntime.preload_hint() directly."""

    def _make_runtime(self, models: List[str], budget_mb: int = 64000):
        from agentic_concierge.infrastructure.backends.model_runtime import LocalModelRuntime
        from agentic_concierge.infrastructure.backends.registry import BackendRegistry

        backend = FakeBackend("test", models)
        registry = BackendRegistry()
        registry.register(backend, priority=0)
        registry.mark_healthy("test")
        return LocalModelRuntime(registry, memory_budget_mb=budget_mb), backend

    @pytest.mark.asyncio
    async def test_preload_loads_model(self):
        runtime, backend = self._make_runtime(["qwen2.5:7b"])
        await runtime.preload_hint({"instruction_following": 0.5})

        # Model should be loaded
        assert "qwen2.5:7b" in backend.loaded
        status = await runtime.status()
        assert len(status.loaded_models) == 1

    @pytest.mark.asyncio
    async def test_preload_noop_when_already_loaded(self):
        runtime, backend = self._make_runtime(["qwen2.5:7b"])
        # First acquire loads the model
        await runtime.acquire({"instruction_following": 0.5})
        backend.loaded.clear()

        # Preload should be a no-op
        await runtime.preload_hint({"instruction_following": 0.5})
        assert len(backend.loaded) == 0

    @pytest.mark.asyncio
    async def test_preload_skips_when_insufficient_memory(self):
        runtime, backend = self._make_runtime(
            ["qwen2.5:7b", "phi-4-mini:14b"],
            budget_mb=5000,  # only room for one 4000 MB model
        )
        # Load one model first
        await runtime.acquire({"instruction_following": 0.5})
        backend.loaded.clear()

        # Preload should skip (not enough memory)
        await runtime.preload_hint({"reasoning": 0.5})
        assert len(backend.loaded) == 0

    @pytest.mark.asyncio
    async def test_preload_no_matching_model(self):
        runtime, backend = self._make_runtime([])
        # No models available — should silently return
        await runtime.preload_hint({"reasoning": 0.5})
        assert len(backend.loaded) == 0

    @pytest.mark.asyncio
    async def test_preload_refcount_zero(self):
        """Preloaded models have refcount 0 (not acquired yet)."""
        runtime, backend = self._make_runtime(["qwen2.5:7b"])
        await runtime.preload_hint({"instruction_following": 0.5})

        status = await runtime.status()
        assert status.loaded_models[0].refcount == 0

    @pytest.mark.asyncio
    async def test_preload_then_acquire_reuses(self):
        """acquire() reuses a preloaded model (fast path)."""
        runtime, backend = self._make_runtime(["qwen2.5:7b"])
        await runtime.preload_hint({"instruction_following": 0.5})
        backend.loaded.clear()

        # Acquire should reuse, not re-load
        handle = await runtime.acquire({"instruction_following": 0.5})
        assert len(backend.loaded) == 0
        assert handle.model_id == "qwen2.5:7b"


# ---------------------------------------------------------------------------
# Affinity executor preload integration
# ---------------------------------------------------------------------------


@dataclass
class MockHandle:
    model_id: str
    released: bool = False


class PreloadTrackingRuntime:
    """Runtime that tracks preload_hint calls."""

    def __init__(self, model_id: str = "qwen2.5:7b"):
        self._model_id = model_id
        self.preload_calls: List[Dict[str, Any]] = []

    async def acquire(self, requirements, **kwargs) -> MockHandle:
        return MockHandle(model_id=self._model_id)

    async def release(self, handle) -> None:
        handle.released = True

    async def preload_hint(
        self,
        requirements: Dict[str, float],
        *,
        require_tool_calling: bool = True,
    ) -> None:
        self.preload_calls.append({
            "requirements": requirements,
            "require_tool_calling": require_tool_calling,
        })


async def _simple_work(node, graph, assignment):
    return {"done": True}


class TestAffinityPreload:

    @pytest.mark.asyncio
    async def test_preload_issued_for_next_sibling(self):
        g = TaskGraph.from_root("Root", node_id="r")
        g.add_child("r", "Step 1", node_id="s1",
                     required_capabilities=["code_python"])
        g.add_child("r", "Step 2", node_id="s2",
                     required_capabilities=["web_comprehension"])
        g.transition("r", "decomposing")
        g.transition("r", "critiqued")

        runtime = PreloadTrackingRuntime()
        await execute_graph_with_affinity(g, runtime, _simple_work)

        # Should have issued at least one preload hint (for s2 while s1 executes)
        assert len(runtime.preload_calls) >= 1

    @pytest.mark.asyncio
    async def test_no_preload_for_single_child(self):
        g = TaskGraph.from_root("Root", node_id="r")
        g.add_child("r", "Only task", node_id="t1",
                     required_capabilities=["code_python"])
        g.transition("r", "decomposing")
        g.transition("r", "critiqued")

        runtime = PreloadTrackingRuntime()
        await execute_graph_with_affinity(g, runtime, _simple_work)

        # No subsequent siblings → no preload hints
        assert len(runtime.preload_calls) == 0

    @pytest.mark.asyncio
    async def test_preload_failure_does_not_break_execution(self):
        """Broken preload_hint doesn't affect execution."""
        g = TaskGraph.from_root("Root", node_id="r")
        g.add_child("r", "S1", node_id="s1",
                     required_capabilities=["code_python"])
        g.add_child("r", "S2", node_id="s2",
                     required_capabilities=["web_comprehension"])
        g.transition("r", "decomposing")
        g.transition("r", "critiqued")

        runtime = PreloadTrackingRuntime()
        # Make preload_hint raise

        async def failing_preload(*args, **kwargs):
            raise RuntimeError("Preload broken")

        runtime.preload_hint = failing_preload

        result = await execute_graph_with_affinity(g, runtime, _simple_work)
        assert result.completed is True
