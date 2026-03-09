"""Tests for LocalModelRuntime — acquire, release, refcounting, model selection.

Verifies:
- acquire() loads model and returns a working ModelHandle
- acquire() reuses already-loaded models (fast path)
- release() decrements refcount
- prefer_model hint selects preferred model
- exclude_models prevents same-model assignment
- require_tool_calling filters non-TC models
- RuntimeError when no model matches
- inventory() returns ModelInfo list
- status() returns RuntimeStatus
- Context manager auto-releases handle
- Multiple acquires increment refcount correctly
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pytest

from agentic_concierge.infrastructure.backends.model_runtime import LocalModelRuntime
from agentic_concierge.infrastructure.backends.protocol import (
    ModelHandle,
    ModelInfo,
    ModelSlot,
    RuntimeStatus,
)
from agentic_concierge.infrastructure.backends.registry import BackendRegistry


# ---------------------------------------------------------------------------
# Fake backend for testing
# ---------------------------------------------------------------------------


class FakeChatClient:
    """Minimal chat client returned by FakeBackend.build_client()."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    async def chat(self, messages, model, **kwargs):
        return None


class FakeBackend:
    """A configurable fake InferenceBackend for testing LocalModelRuntime."""

    def __init__(
        self,
        backend_name: str = "fake",
        models: Optional[List[str]] = None,
        memory_estimates: Optional[Dict[str, int]] = None,
    ) -> None:
        self._name = backend_name
        self._models = models or []
        self._memory_estimates = memory_estimates or {}
        self._loaded: Dict[str, ModelSlot] = {}
        self.load_calls: List[str] = []  # track load_model calls

    @property
    def name(self) -> str:
        return self._name

    @property
    def base_url(self) -> str:
        return ""

    async def health_check(self) -> bool:
        return True

    async def list_available(self) -> List[str]:
        return list(self._models)

    async def list_loaded(self) -> List[ModelSlot]:
        return list(self._loaded.values())

    async def load_model(self, model_id: str) -> ModelSlot:
        self.load_calls.append(model_id)
        mem = self._memory_estimates.get(model_id, 4000)
        slot = ModelSlot(
            model_id=model_id,
            backend=self._name,
            memory_mb=mem,
        )
        self._loaded[model_id] = slot
        return slot

    async def unload_model(self, model_id: str) -> None:
        self._loaded.pop(model_id, None)

    def build_client(self, model_id: str) -> FakeChatClient:
        return FakeChatClient(model_id)

    async def estimate_memory(self, model_id: str) -> int:
        return self._memory_estimates.get(model_id, 4000)

    async def pull_model(self, model_id: str, *, timeout_s: int = 600) -> bool:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry(*backends: tuple[FakeBackend, int]) -> BackendRegistry:
    """Create a BackendRegistry with pre-registered and pre-discovered backends."""
    registry = BackendRegistry()
    for backend, priority in backends:
        registry.register(backend, priority=priority)
    # Mark all as healthy (bypassing actual health check I/O)
    for entry in registry._entries.values():
        entry.healthy = True
    return registry


def _make_runtime(
    models: Optional[List[str]] = None,
    backend_name: str = "fake",
    memory_estimates: Optional[Dict[str, int]] = None,
) -> tuple[LocalModelRuntime, FakeBackend]:
    """Convenience: create a LocalModelRuntime with a single FakeBackend."""
    backend = FakeBackend(
        backend_name=backend_name,
        models=models if models is not None else ["qwen2.5:7b", "qwen2.5-coder:7b", "llama3.1:8b"],
        memory_estimates=memory_estimates or {},
    )
    registry = _make_registry((backend, 0))
    return LocalModelRuntime(registry), backend


# ---------------------------------------------------------------------------
# Acquire basics
# ---------------------------------------------------------------------------


class TestAcquire:

    @pytest.mark.asyncio
    async def test_acquire_loads_and_returns_handle(self):
        runtime, backend = _make_runtime()
        handle = await runtime.acquire({"reasoning": 0.5})

        assert isinstance(handle, ModelHandle)
        assert handle.model_id in ("qwen2.5:7b", "qwen2.5-coder:7b", "llama3.1:8b")
        assert handle.slot.refcount == 1
        assert handle.chat_client is not None
        assert len(backend.load_calls) == 1

    @pytest.mark.asyncio
    async def test_acquire_reuses_loaded_model(self):
        runtime, backend = _make_runtime()

        handle1 = await runtime.acquire({"reasoning": 0.5})
        model1 = handle1.model_id

        handle2 = await runtime.acquire({"reasoning": 0.5})
        model2 = handle2.model_id

        # Should reuse the loaded model, not load again
        assert model1 == model2
        assert handle1.slot is handle2.slot
        assert handle1.slot.refcount == 2
        assert len(backend.load_calls) == 1  # only one load_model call

    @pytest.mark.asyncio
    async def test_acquire_different_requirements_may_load_new(self):
        """Different requirements may pick a different model, loading a second one."""
        runtime, backend = _make_runtime()

        # First acquire: something with reasoning
        handle1 = await runtime.acquire({"reasoning": 0.7})
        model1 = handle1.model_id

        # Second acquire: code capability, excluding the first model
        handle2 = await runtime.acquire(
            {"code_python": 0.8},
            exclude_models=[model1],
        )

        # Should have loaded a second model
        assert handle2.model_id != model1
        assert len(backend.load_calls) == 2


# ---------------------------------------------------------------------------
# Release and refcounting
# ---------------------------------------------------------------------------


class TestRelease:

    @pytest.mark.asyncio
    async def test_release_decrements_refcount(self):
        runtime, _ = _make_runtime()

        handle = await runtime.acquire({"reasoning": 0.5})
        assert handle.slot.refcount == 1

        await runtime.release(handle)
        assert handle.slot.refcount == 0

    @pytest.mark.asyncio
    async def test_release_updates_last_used(self):
        runtime, _ = _make_runtime()
        handle = await runtime.acquire({"reasoning": 0.5})

        before = time.monotonic()
        await runtime.release(handle)
        after = time.monotonic()

        assert before <= handle.slot.last_used <= after

    @pytest.mark.asyncio
    async def test_multiple_acquires_and_releases(self):
        runtime, _ = _make_runtime()

        h1 = await runtime.acquire({"reasoning": 0.5})
        h2 = await runtime.acquire({"reasoning": 0.5})
        assert h1.slot.refcount == 2

        await runtime.release(h1)
        assert h1.slot.refcount == 1

        await runtime.release(h2)
        assert h1.slot.refcount == 0

    @pytest.mark.asyncio
    async def test_release_never_goes_negative(self):
        runtime, _ = _make_runtime()
        handle = await runtime.acquire({"reasoning": 0.5})

        await runtime.release(handle)
        await runtime.release(handle)  # extra release
        assert handle.slot.refcount == 0


# ---------------------------------------------------------------------------
# prefer_model hint
# ---------------------------------------------------------------------------


class TestPreferModel:

    @pytest.mark.asyncio
    async def test_prefer_model_selects_preferred(self):
        runtime, _ = _make_runtime()
        handle = await runtime.acquire(
            {"code_python": 0.5},
            prefer_model="qwen2.5-coder:7b",
        )
        assert handle.model_id == "qwen2.5-coder:7b"

    @pytest.mark.asyncio
    async def test_prefer_model_not_available_falls_back(self):
        runtime, _ = _make_runtime(models=["qwen2.5:7b", "llama3.1:8b"])
        handle = await runtime.acquire(
            {"reasoning": 0.5},
            prefer_model="nonexistent:7b",
        )
        # Should still get a valid handle
        assert handle.model_id in ("qwen2.5:7b", "llama3.1:8b")

    @pytest.mark.asyncio
    async def test_prefer_model_reuses_if_loaded(self):
        runtime, backend = _make_runtime()

        # Load qwen2.5-coder:7b first
        h1 = await runtime.acquire({}, prefer_model="qwen2.5-coder:7b")
        assert h1.model_id == "qwen2.5-coder:7b"

        # Second acquire with same preference — should reuse (fast path)
        h2 = await runtime.acquire({}, prefer_model="qwen2.5-coder:7b")
        assert h2.model_id == "qwen2.5-coder:7b"
        assert h1.slot is h2.slot
        assert len(backend.load_calls) == 1


# ---------------------------------------------------------------------------
# exclude_models constraint
# ---------------------------------------------------------------------------


class TestExcludeModels:

    @pytest.mark.asyncio
    async def test_exclude_models_prevents_selection(self):
        runtime, _ = _make_runtime(models=["qwen2.5:7b", "llama3.1:8b"])

        handle = await runtime.acquire(
            {"reasoning": 0.5},
            exclude_models=["qwen2.5:7b"],
        )
        assert handle.model_id != "qwen2.5:7b"

    @pytest.mark.asyncio
    async def test_exclude_all_models_raises(self):
        runtime, _ = _make_runtime(models=["qwen2.5:7b"])

        with pytest.raises(RuntimeError, match="No model matching"):
            await runtime.acquire(
                {"reasoning": 0.5},
                exclude_models=["qwen2.5:7b"],
            )

    @pytest.mark.asyncio
    async def test_exclude_loaded_model_loads_alternative(self):
        runtime, backend = _make_runtime()

        # Load first model
        h1 = await runtime.acquire({"reasoning": 0.5})

        # Acquire excluding the loaded model
        h2 = await runtime.acquire(
            {"reasoning": 0.5},
            exclude_models=[h1.model_id],
        )
        assert h2.model_id != h1.model_id
        assert len(backend.load_calls) == 2


# ---------------------------------------------------------------------------
# require_tool_calling filter
# ---------------------------------------------------------------------------


class TestRequireToolCalling:

    @pytest.mark.asyncio
    async def test_require_tool_calling_filters_non_tc(self):
        """Models without tool calling support are skipped when required."""
        # sqlcoder is known non-tool-calling in profiles
        runtime, _ = _make_runtime(models=["sqlcoder:7b", "qwen2.5:7b"])

        handle = await runtime.acquire(
            {"code_sql": 0.5},
            require_tool_calling=True,
        )
        # Should skip sqlcoder (no tool calling) and pick qwen2.5
        assert handle.model_id == "qwen2.5:7b"

    @pytest.mark.asyncio
    async def test_require_tool_calling_false_allows_non_tc(self):
        """Non-tool-calling models are valid when require_tool_calling=False."""
        runtime, _ = _make_runtime(models=["sqlcoder:7b"])

        handle = await runtime.acquire(
            {"code_sql": 0.5},
            require_tool_calling=False,
        )
        assert handle.model_id == "sqlcoder:7b"

    @pytest.mark.asyncio
    async def test_prefer_model_respects_tool_calling_filter(self):
        """prefer_model is ignored if it doesn't meet tool_calling requirement."""
        runtime, _ = _make_runtime(models=["sqlcoder:7b", "qwen2.5:7b"])

        handle = await runtime.acquire(
            {"code_sql": 0.5},
            prefer_model="sqlcoder:7b",
            require_tool_calling=True,
        )
        assert handle.model_id == "qwen2.5:7b"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:

    @pytest.mark.asyncio
    async def test_no_backends_raises(self):
        registry = BackendRegistry()
        runtime = LocalModelRuntime(registry)

        with pytest.raises(RuntimeError, match="No model matching"):
            await runtime.acquire({"reasoning": 0.5})

    @pytest.mark.asyncio
    async def test_empty_model_list_raises(self):
        runtime, _ = _make_runtime(models=[])

        with pytest.raises(RuntimeError, match="No model matching"):
            await runtime.acquire({"reasoning": 0.5})


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


class TestInventory:

    @pytest.mark.asyncio
    async def test_inventory_returns_model_info(self):
        runtime, _ = _make_runtime(
            models=["qwen2.5:7b", "qwen2.5-coder:7b"],
            memory_estimates={"qwen2.5:7b": 4500, "qwen2.5-coder:7b": 4200},
        )

        models = await runtime.inventory()
        assert len(models) == 2
        assert all(isinstance(m, ModelInfo) for m in models)

        ids = {m.model_id for m in models}
        assert ids == {"qwen2.5:7b", "qwen2.5-coder:7b"}

    @pytest.mark.asyncio
    async def test_inventory_marks_loaded(self):
        runtime, _ = _make_runtime(models=["qwen2.5:7b", "llama3.1:8b"])

        # Load one model
        await runtime.acquire({"reasoning": 0.5})

        models = await runtime.inventory()
        loaded = [m for m in models if m.loaded]
        unloaded = [m for m in models if not m.loaded]

        assert len(loaded) == 1
        assert len(unloaded) == 1

    @pytest.mark.asyncio
    async def test_inventory_includes_capabilities(self):
        runtime, _ = _make_runtime(models=["qwen2.5-coder:7b"])

        models = await runtime.inventory()
        assert len(models) == 1
        # qwen2.5-coder should have code_python capability from profiles
        assert "code_python" in models[0].capabilities


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class TestStatus:

    @pytest.mark.asyncio
    async def test_status_empty(self):
        runtime, _ = _make_runtime()

        st = await runtime.status()
        assert isinstance(st, RuntimeStatus)
        assert st.loaded_models == []
        assert st.used_memory_mb == 0

    @pytest.mark.asyncio
    async def test_status_after_acquire(self):
        runtime, _ = _make_runtime(
            memory_estimates={"qwen2.5:7b": 4500},
        )

        await runtime.acquire({"reasoning": 0.5})

        st = await runtime.status()
        assert len(st.loaded_models) == 1
        assert st.used_memory_mb > 0


# ---------------------------------------------------------------------------
# Context manager integration
# ---------------------------------------------------------------------------


class TestContextManager:

    @pytest.mark.asyncio
    async def test_context_manager_auto_releases(self):
        runtime, _ = _make_runtime()

        handle = await runtime.acquire({"reasoning": 0.5})
        slot = handle.slot
        assert slot.refcount == 1

        async with handle:
            assert slot.refcount == 1

        # Auto-released via __aexit__ → runtime.release()
        assert slot.refcount == 0

    @pytest.mark.asyncio
    async def test_context_manager_releases_on_exception(self):
        runtime, _ = _make_runtime()

        handle = await runtime.acquire({"reasoning": 0.5})
        slot = handle.slot

        with pytest.raises(ValueError):
            async with handle:
                raise ValueError("boom")

        assert slot.refcount == 0


# ---------------------------------------------------------------------------
# Multi-backend scenarios
# ---------------------------------------------------------------------------


class TestMultiBackend:

    @pytest.mark.asyncio
    async def test_primary_backend_preferred(self):
        """Models available on both backends — primary (higher priority) wins."""
        backend_a = FakeBackend("backend_a", models=["qwen2.5:7b"])
        backend_b = FakeBackend("backend_b", models=["qwen2.5:7b", "llama3.1:8b"])

        registry = _make_registry((backend_a, 0), (backend_b, 1))
        runtime = LocalModelRuntime(registry)

        handle = await runtime.acquire(
            {"reasoning": 0.5},
            prefer_model="qwen2.5:7b",
        )
        # qwen2.5:7b available on both; backend_a has higher priority
        assert handle.slot.backend == "backend_a"

    @pytest.mark.asyncio
    async def test_fallback_to_secondary_backend(self):
        """Model only on secondary backend — loads from there."""
        backend_a = FakeBackend("backend_a", models=["qwen2.5:7b"])
        backend_b = FakeBackend("backend_b", models=["llama3.1:8b"])

        registry = _make_registry((backend_a, 0), (backend_b, 1))
        runtime = LocalModelRuntime(registry)

        handle = await runtime.acquire(
            {"reasoning": 0.5},
            exclude_models=["qwen2.5:7b"],
        )
        assert handle.model_id == "llama3.1:8b"
        assert handle.slot.backend == "backend_b"
