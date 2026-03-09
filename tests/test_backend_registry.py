"""Tests for BackendRegistry — discovery, health monitoring, and failover.

Verifies:
- Backend registration and lookup
- Discovery probes all backends concurrently
- Priority ordering (lower = higher priority)
- Failover when primary backend is unhealthy
- Health recovery re-enables backend
- mark_unhealthy / mark_healthy transitions
- from_config factory creates registry from YAML config
- Status reporting
- Error handling (no backends, all unhealthy)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from agentic_concierge.infrastructure.backends.protocol import (
    InferenceBackend,
    ModelSlot,
)
from agentic_concierge.infrastructure.backends.registry import (
    BackendEntry,
    BackendRegistry,
)


# ---------------------------------------------------------------------------
# Fake backend for testing
# ---------------------------------------------------------------------------


class FakeBackend:
    """A fake InferenceBackend for testing."""

    def __init__(self, name: str, base_url: str = "", healthy: bool = True) -> None:
        self._name = name
        self._base_url = base_url
        self._healthy = healthy
        self._models: List[str] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def base_url(self) -> str:
        return self._base_url

    async def health_check(self) -> bool:
        return self._healthy

    async def list_available(self) -> List[str]:
        return self._models

    async def list_loaded(self) -> List[ModelSlot]:
        return []

    async def load_model(self, model_id: str) -> ModelSlot:
        return ModelSlot(model_id=model_id, backend=self._name)

    async def unload_model(self, model_id: str) -> None:
        pass

    def build_client(self, model_id: str):
        return None

    async def estimate_memory(self, model_id: str) -> int:
        return 0

    async def pull_model(self, model_id: str, *, timeout_s: int = 600) -> bool:
        return False


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:

    def test_register_and_get(self):
        registry = BackendRegistry()
        backend = FakeBackend("test")
        registry.register(backend, priority=0)
        assert registry.get_backend("test") is backend

    def test_get_nonexistent_returns_none(self):
        registry = BackendRegistry()
        assert registry.get_backend("nonexistent") is None

    def test_unregister(self):
        registry = BackendRegistry()
        registry.register(FakeBackend("test"), priority=0)
        registry.unregister("test")
        assert registry.get_backend("test") is None

    def test_unregister_nonexistent_is_noop(self):
        registry = BackendRegistry()
        registry.unregister("nonexistent")  # should not raise

    def test_list_backends(self):
        registry = BackendRegistry()
        registry.register(FakeBackend("a"), priority=0)
        registry.register(FakeBackend("b"), priority=1)
        assert set(registry.list_backends()) == {"a", "b"}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscovery:

    @pytest.mark.asyncio
    async def test_discover_healthy(self):
        registry = BackendRegistry()
        registry.register(FakeBackend("ollama", healthy=True), priority=0)
        registry.register(FakeBackend("vllm", healthy=True), priority=1)

        healthy = await registry.discover()
        assert "ollama" in healthy
        assert "vllm" in healthy

    @pytest.mark.asyncio
    async def test_discover_unhealthy(self):
        registry = BackendRegistry()
        registry.register(FakeBackend("ollama", healthy=False), priority=0)

        healthy = await registry.discover()
        assert healthy == []

    @pytest.mark.asyncio
    async def test_discover_mixed(self):
        registry = BackendRegistry()
        registry.register(FakeBackend("ollama", healthy=True), priority=0)
        registry.register(FakeBackend("vllm", healthy=False), priority=1)

        healthy = await registry.discover()
        assert healthy == ["ollama"]

    @pytest.mark.asyncio
    async def test_discover_empty_registry(self):
        registry = BackendRegistry()
        healthy = await registry.discover()
        assert healthy == []

    @pytest.mark.asyncio
    async def test_discover_handles_exception(self):
        """Backend that raises during health check is marked unhealthy."""
        backend = FakeBackend("broken", healthy=True)

        async def fail():
            raise RuntimeError("connection refused")

        backend.health_check = fail  # type: ignore[assignment]
        registry = BackendRegistry()
        registry.register(backend, priority=0)

        healthy = await registry.discover()
        assert healthy == []


# ---------------------------------------------------------------------------
# Priority ordering and failover
# ---------------------------------------------------------------------------


class TestPriorityAndFailover:

    @pytest.mark.asyncio
    async def test_primary_returns_highest_priority(self):
        registry = BackendRegistry()
        registry.register(FakeBackend("vllm", healthy=True), priority=1)
        registry.register(FakeBackend("ollama", healthy=True), priority=0)

        await registry.discover()
        primary = registry.primary()
        assert primary.name == "ollama"

    @pytest.mark.asyncio
    async def test_failover_to_next_healthy(self):
        registry = BackendRegistry()
        registry.register(FakeBackend("ollama", healthy=False), priority=0)
        registry.register(FakeBackend("vllm", healthy=True), priority=1)

        await registry.discover()
        primary = registry.primary()
        assert primary.name == "vllm"

    @pytest.mark.asyncio
    async def test_no_healthy_raises(self):
        registry = BackendRegistry()
        registry.register(FakeBackend("ollama", healthy=False), priority=0)

        await registry.discover()
        with pytest.raises(RuntimeError, match="No healthy"):
            registry.primary()

    @pytest.mark.asyncio
    async def test_primary_or_none_returns_none(self):
        registry = BackendRegistry()
        registry.register(FakeBackend("ollama", healthy=False), priority=0)

        await registry.discover()
        assert registry.primary_or_none() is None

    @pytest.mark.asyncio
    async def test_healthy_backends_order(self):
        registry = BackendRegistry()
        registry.register(FakeBackend("c", healthy=True), priority=2)
        registry.register(FakeBackend("a", healthy=True), priority=0)
        registry.register(FakeBackend("b", healthy=True), priority=1)

        await registry.discover()
        backends = registry.healthy_backends()
        assert [b.name for b in backends] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Health monitoring
# ---------------------------------------------------------------------------


class TestHealthMonitoring:

    @pytest.mark.asyncio
    async def test_mark_unhealthy_triggers_failover(self):
        registry = BackendRegistry()
        registry.register(FakeBackend("ollama", healthy=True), priority=0)
        registry.register(FakeBackend("vllm", healthy=True), priority=1)

        await registry.discover()
        assert registry.primary().name == "ollama"

        registry.mark_unhealthy("ollama", error="connection lost")
        assert registry.primary().name == "vllm"

    @pytest.mark.asyncio
    async def test_mark_healthy_re_enables(self):
        registry = BackendRegistry()
        ollama = FakeBackend("ollama", healthy=True)
        registry.register(ollama, priority=0)
        registry.register(FakeBackend("vllm", healthy=True), priority=1)

        await registry.discover()
        registry.mark_unhealthy("ollama")
        assert registry.primary().name == "vllm"

        registry.mark_healthy("ollama")
        assert registry.primary().name == "ollama"

    @pytest.mark.asyncio
    async def test_health_check_all(self):
        registry = BackendRegistry()
        registry.register(FakeBackend("ollama", healthy=True), priority=0)
        registry.register(FakeBackend("vllm", healthy=False), priority=1)

        status = await registry.health_check_all()
        assert status == {"ollama": True, "vllm": False}

    @pytest.mark.asyncio
    async def test_consecutive_failures_tracked(self):
        registry = BackendRegistry()
        registry.register(FakeBackend("ollama", healthy=False), priority=0)

        await registry.discover()
        await registry.discover()

        entry = registry._entries["ollama"]
        assert entry.consecutive_failures == 2

    @pytest.mark.asyncio
    async def test_recovery_resets_failures(self):
        registry = BackendRegistry()
        backend = FakeBackend("ollama", healthy=False)
        registry.register(backend, priority=0)

        await registry.discover()
        assert registry._entries["ollama"].consecutive_failures == 1

        backend._healthy = True
        await registry.discover()
        assert registry._entries["ollama"].consecutive_failures == 0
        assert registry._entries["ollama"].healthy is True

    def test_mark_unhealthy_nonexistent_is_noop(self):
        registry = BackendRegistry()
        registry.mark_unhealthy("nonexistent")  # should not raise

    def test_mark_healthy_nonexistent_is_noop(self):
        registry = BackendRegistry()
        registry.mark_healthy("nonexistent")  # should not raise


# ---------------------------------------------------------------------------
# Status reporting
# ---------------------------------------------------------------------------


class TestStatus:

    @pytest.mark.asyncio
    async def test_status_report(self):
        registry = BackendRegistry()
        registry.register(FakeBackend("ollama", base_url="http://localhost:11434", healthy=True), priority=0)

        await registry.discover()
        status = registry.status()

        assert "ollama" in status
        assert status["ollama"]["healthy"] is True
        assert status["ollama"]["priority"] == 0
        assert status["ollama"]["base_url"] == "http://localhost:11434"
        assert status["ollama"]["consecutive_failures"] == 0

    @pytest.mark.asyncio
    async def test_status_after_failure(self):
        registry = BackendRegistry()
        registry.register(FakeBackend("ollama", healthy=True), priority=0)

        await registry.discover()
        registry.mark_unhealthy("ollama", error="timeout")

        status = registry.status()
        assert status["ollama"]["healthy"] is False
        assert status["ollama"]["last_error"] == "timeout"
        assert status["ollama"]["consecutive_failures"] == 1


# ---------------------------------------------------------------------------
# Factory: from_config
# ---------------------------------------------------------------------------


class TestFromConfig:

    def test_creates_registry_with_ollama(self):
        registry = BackendRegistry.from_config()
        assert "ollama" in registry.list_backends()

    def test_ollama_backend_registered(self):
        registry = BackendRegistry.from_config()
        backend = registry.get_backend("ollama")
        assert backend is not None
        assert backend.name == "ollama"

    def test_custom_tier(self):
        registry = BackendRegistry.from_config(tier="large")
        assert "ollama" in registry.list_backends()

    def test_unknown_tier_defaults_to_medium(self):
        registry = BackendRegistry.from_config(tier="nonexistent")
        assert "ollama" in registry.list_backends()
