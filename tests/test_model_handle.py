"""Tests for ModelHandle, ModelInfo, RuntimeStatus, and ModelRuntime protocol.

Verifies:
- ModelHandle as async context manager (auto-release)
- ModelHandle auto-releases on exception
- ModelHandle double-release is safe
- ModelHandle with runtime back-reference
- ModelSlot refcount and last_used tracking
- ModelInfo dataclass
- RuntimeStatus.free_memory_mb property
- ModelRuntime protocol shape
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from agentic_concierge.application.ports import ChatClient, ModelRuntime
from agentic_concierge.infrastructure.backends.protocol import (
    ModelHandle,
    ModelInfo,
    ModelSlot,
    RuntimeStatus,
)


# ---------------------------------------------------------------------------
# Fake chat client for testing
# ---------------------------------------------------------------------------


class FakeChatClient:
    async def chat(self, messages, model, **kwargs):
        return None


# ---------------------------------------------------------------------------
# ModelSlot
# ---------------------------------------------------------------------------


class TestModelSlot:

    def test_defaults(self):
        slot = ModelSlot(model_id="qwen2.5:7b", backend="ollama")
        assert slot.refcount == 0
        assert slot.last_used == 0.0
        assert slot.memory_mb == 0

    def test_refcount_tracking(self):
        slot = ModelSlot(model_id="test", backend="ollama", refcount=2)
        assert slot.refcount == 2
        slot.refcount -= 1
        assert slot.refcount == 1


# ---------------------------------------------------------------------------
# ModelHandle — context manager
# ---------------------------------------------------------------------------


class TestModelHandleContextManager:

    @pytest.mark.asyncio
    async def test_auto_release_on_exit(self):
        slot = ModelSlot(model_id="test", backend="ollama", refcount=1)
        handle = ModelHandle(slot=slot, chat_client=FakeChatClient())

        async with handle:
            assert not handle._released

        assert handle._released
        assert slot.refcount == 0

    @pytest.mark.asyncio
    async def test_auto_release_on_exception(self):
        slot = ModelSlot(model_id="test", backend="ollama", refcount=1)
        handle = ModelHandle(slot=slot, chat_client=FakeChatClient())

        with pytest.raises(ValueError):
            async with handle:
                raise ValueError("boom")

        assert handle._released
        assert slot.refcount == 0

    @pytest.mark.asyncio
    async def test_double_release_is_safe(self):
        slot = ModelSlot(model_id="test", backend="ollama", refcount=1)
        handle = ModelHandle(slot=slot, chat_client=FakeChatClient())

        await handle.release()
        assert slot.refcount == 0

        # Second release should be no-op
        await handle.release()
        assert slot.refcount == 0

    @pytest.mark.asyncio
    async def test_refcount_never_goes_negative(self):
        slot = ModelSlot(model_id="test", backend="ollama", refcount=0)
        handle = ModelHandle(slot=slot, chat_client=FakeChatClient())

        await handle.release()
        assert slot.refcount == 0

    @pytest.mark.asyncio
    async def test_last_used_updated_on_release(self):
        slot = ModelSlot(model_id="test", backend="ollama", refcount=1)
        handle = ModelHandle(slot=slot, chat_client=FakeChatClient())

        before = time.monotonic()
        await handle.release()
        after = time.monotonic()

        assert before <= slot.last_used <= after

    @pytest.mark.asyncio
    async def test_model_id_accessible(self):
        slot = ModelSlot(model_id="qwen2.5:7b", backend="ollama")
        handle = ModelHandle(slot=slot, chat_client=FakeChatClient())
        assert handle.model_id == "qwen2.5:7b"

    @pytest.mark.asyncio
    async def test_chat_client_accessible(self):
        client = FakeChatClient()
        slot = ModelSlot(model_id="test", backend="ollama")
        handle = ModelHandle(slot=slot, chat_client=client)
        assert handle.chat_client is client


# ---------------------------------------------------------------------------
# ModelHandle — with runtime back-reference
# ---------------------------------------------------------------------------


class TestModelHandleWithRuntime:

    @pytest.mark.asyncio
    async def test_release_calls_runtime(self):
        slot = ModelSlot(model_id="test", backend="ollama", refcount=1)
        runtime = AsyncMock()
        handle = ModelHandle(slot=slot, chat_client=FakeChatClient(), runtime=runtime)

        await handle.release()
        runtime.release.assert_awaited_once_with(handle)

    @pytest.mark.asyncio
    async def test_context_manager_calls_runtime(self):
        slot = ModelSlot(model_id="test", backend="ollama", refcount=1)
        runtime = AsyncMock()
        handle = ModelHandle(slot=slot, chat_client=FakeChatClient(), runtime=runtime)

        async with handle:
            pass

        runtime.release.assert_awaited_once()


# ---------------------------------------------------------------------------
# ModelInfo
# ---------------------------------------------------------------------------


class TestModelInfo:

    def test_defaults(self):
        info = ModelInfo(model_id="qwen2.5:7b", backend="ollama")
        assert info.available is True
        assert info.loaded is False
        assert info.capabilities == {}
        assert info.estimated_memory_mb == 0
        assert info.supports_tool_calling is True

    def test_populated(self):
        info = ModelInfo(
            model_id="qwen2.5-coder:7b",
            backend="ollama",
            loaded=True,
            capabilities={"code_python": 0.95},
            estimated_memory_mb=4500,
        )
        assert info.capabilities["code_python"] == 0.95
        assert info.estimated_memory_mb == 4500


# ---------------------------------------------------------------------------
# RuntimeStatus
# ---------------------------------------------------------------------------


class TestRuntimeStatus:

    def test_free_memory(self):
        status = RuntimeStatus(total_memory_mb=128_000, used_memory_mb=17_000)
        assert status.free_memory_mb == 111_000

    def test_free_memory_never_negative(self):
        status = RuntimeStatus(total_memory_mb=8_000, used_memory_mb=10_000)
        assert status.free_memory_mb == 0

    def test_defaults(self):
        status = RuntimeStatus()
        assert status.loaded_models == []
        assert status.total_memory_mb == 0
        assert status.free_memory_mb == 0

    def test_with_loaded_models(self):
        slots = [
            ModelSlot(model_id="a", backend="ollama", memory_mb=4000),
            ModelSlot(model_id="b", backend="ollama", memory_mb=8000),
        ]
        status = RuntimeStatus(
            loaded_models=slots,
            total_memory_mb=128_000,
            used_memory_mb=12_000,
        )
        assert len(status.loaded_models) == 2
        assert status.free_memory_mb == 116_000


# ---------------------------------------------------------------------------
# ModelRuntime protocol shape
# ---------------------------------------------------------------------------


class FakeModelRuntime:
    """Minimal implementation to verify protocol shape."""

    async def acquire(
        self,
        requirements: Dict[str, float],
        *,
        prefer_model: Optional[str] = None,
        require_tool_calling: bool = True,
        exclude_models: Optional[List[str]] = None,
        prefer_concurrent: bool = False,
        timeout_s: float = 30.0,
    ) -> ModelHandle:
        slot = ModelSlot(model_id="test", backend="fake", refcount=1)
        return ModelHandle(slot=slot, chat_client=FakeChatClient(), runtime=self)

    async def release(self, handle: Any) -> None:
        handle.slot.refcount = max(0, handle.slot.refcount - 1)

    async def inventory(self) -> List[ModelInfo]:
        return [ModelInfo(model_id="test", backend="fake")]

    async def status(self) -> RuntimeStatus:
        return RuntimeStatus(total_memory_mb=8000, used_memory_mb=0)


class TestModelRuntimeProtocol:

    @pytest.mark.asyncio
    async def test_acquire_returns_handle(self):
        runtime = FakeModelRuntime()
        handle = await runtime.acquire({"reasoning": 0.5})
        assert isinstance(handle, ModelHandle)
        assert handle.model_id == "test"

    @pytest.mark.asyncio
    async def test_acquire_context_manager_releases(self):
        runtime = FakeModelRuntime()
        async with await runtime.acquire({"reasoning": 0.5}) as handle:
            assert handle.slot.refcount == 1
        assert handle.slot.refcount == 0

    @pytest.mark.asyncio
    async def test_inventory(self):
        runtime = FakeModelRuntime()
        models = await runtime.inventory()
        assert len(models) == 1
        assert models[0].model_id == "test"

    @pytest.mark.asyncio
    async def test_status(self):
        runtime = FakeModelRuntime()
        st = await runtime.status()
        assert st.total_memory_mb == 8000
        assert st.free_memory_mb == 8000
