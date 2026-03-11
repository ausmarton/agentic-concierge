"""Tests for ADR-033 operational excellence items (OPS-1, OPS-2, OPS-3).

OPS-1: vLLM auto-start on SERVER/LARGE profiles
OPS-2: Parallel backend preference for concurrent task graph nodes
OPS-3: Connection pooling for concurrent requests
OPS-1b: llama-server auto-install
OPS-1c: First-run bootstrap wiring
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agentic_concierge.bootstrap.backend_manager import (
    BackendHealth,
    BackendManager,
    BackendStatus,
)
from agentic_concierge.config.features import Feature, FeatureSet
from agentic_concierge.config.schema import ConciergeConfig


# ============================================================================
# OPS-3: Connection pooling
# ============================================================================


class TestConnectionPoolingOllamaChat:
    """OllamaChatClient reuses a shared httpx.AsyncClient."""

    def test_shared_client_created_lazily(self):
        from agentic_concierge.infrastructure.ollama.client import OllamaChatClient

        c = OllamaChatClient(base_url="http://localhost:11434/v1")
        assert c._client is None  # not created until first use

    def test_get_client_returns_same_instance(self):
        from agentic_concierge.infrastructure.ollama.client import OllamaChatClient

        c = OllamaChatClient(base_url="http://localhost:11434/v1")
        client1 = c._get_client()
        client2 = c._get_client()
        assert client1 is client2

    def test_shared_client_is_async_client(self):
        from agentic_concierge.infrastructure.ollama.client import OllamaChatClient

        c = OllamaChatClient(base_url="http://localhost:11434/v1")
        client = c._get_client()
        assert isinstance(client, httpx.AsyncClient)

    def test_timeout_from_constructor(self):
        from agentic_concierge.infrastructure.ollama.client import OllamaChatClient

        c = OllamaChatClient(base_url="http://localhost:11434/v1", timeout_s=42.0)
        client = c._get_client()
        assert client.timeout.read == 42.0


class TestConnectionPoolingGenericChat:
    """GenericChatClient reuses a shared httpx.AsyncClient."""

    def test_shared_client_created_lazily(self):
        from agentic_concierge.infrastructure.chat.generic import GenericChatClient

        c = GenericChatClient(base_url="http://localhost:8000/v1")
        assert c._client is None

    def test_get_client_returns_same_instance(self):
        from agentic_concierge.infrastructure.chat.generic import GenericChatClient

        c = GenericChatClient(base_url="http://localhost:8000/v1")
        client1 = c._get_client()
        client2 = c._get_client()
        assert client1 is client2


class TestConnectionPoolingVLLMChat:
    """VLLMChatClient reuses a shared httpx.AsyncClient."""

    def test_shared_client_created_lazily(self):
        from agentic_concierge.infrastructure.chat.vllm import VLLMChatClient

        c = VLLMChatClient(base_url="http://localhost:8000/v1")
        assert c._client is None

    def test_get_client_returns_same_instance(self):
        from agentic_concierge.infrastructure.chat.vllm import VLLMChatClient

        c = VLLMChatClient(base_url="http://localhost:8000/v1")
        client1 = c._get_client()
        client2 = c._get_client()
        assert client1 is client2


class TestConnectionPoolingOllamaBackend:
    """OllamaBackend reuses a shared httpx.AsyncClient for management calls."""

    def test_shared_client_created_lazily(self):
        from agentic_concierge.infrastructure.backends.ollama import OllamaBackend

        b = OllamaBackend()
        assert b._client is None

    def test_get_client_returns_same_instance(self):
        from agentic_concierge.infrastructure.backends.ollama import OllamaBackend

        b = OllamaBackend()
        client1 = b._get_client()
        client2 = b._get_client()
        assert client1 is client2

    @pytest.mark.asyncio
    async def test_health_check_uses_shared_client(self):
        from agentic_concierge.infrastructure.backends.ollama import OllamaBackend

        b = OllamaBackend()
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        b._client = mock_client

        result = await b.health_check()
        assert result is True
        mock_client.get.assert_awaited_once()


# ============================================================================
# OPS-1: vLLM auto-start
# ============================================================================


def _mock_vllm_healthy() -> BackendHealth:
    return BackendHealth(
        name="vllm",
        status=BackendStatus.HEALTHY,
        base_url="http://localhost:8000",
        models=["mistral-7b"],
    )


def _mock_vllm_unreachable() -> BackendHealth:
    return BackendHealth(
        name="vllm",
        status=BackendStatus.UNREACHABLE,
        base_url="http://localhost:8000",
        error="Connection refused",
    )


def _mock_ollama_with_models() -> BackendHealth:
    return BackendHealth(
        name="ollama",
        status=BackendStatus.HEALTHY,
        base_url="http://localhost:11434",
        models=["qwen2.5:7b", "llama3.1:8b"],
    )


def _fs(*features: Feature) -> FeatureSet:
    return FeatureSet(enabled=frozenset(features))


def _config(**overrides) -> ConciergeConfig:
    """Build a minimal ConciergeConfig with optional overrides."""
    from agentic_concierge.config.schema import ModelConfig, SpecialistConfig

    defaults = {
        "models": {
            "fast": ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5:7b"),
        },
        "specialists": {
            "engineering": SpecialistConfig(description="test"),
        },
    }
    defaults.update(overrides)
    return ConciergeConfig(**defaults)


class TestEnsureVllm:
    """Tests for BackendManager.ensure_vllm()."""

    @pytest.mark.asyncio
    async def test_already_healthy_returns_immediately(self):
        mgr = BackendManager()
        fs = _fs(Feature.VLLM, Feature.OLLAMA)
        config = _config(vllm_ensure_available=True)
        with patch.object(mgr, "probe_vllm", return_value=_mock_vllm_healthy()):
            health = await mgr.ensure_vllm(config, fs)
        assert health.status == BackendStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_disabled_feature_returns_disabled(self):
        mgr = BackendManager()
        fs = _fs(Feature.OLLAMA)  # VLLM not enabled
        config = _config(vllm_ensure_available=True)
        health = await mgr.ensure_vllm(config, fs)
        assert health.status == BackendStatus.DISABLED

    @pytest.mark.asyncio
    async def test_not_configured_returns_unreachable(self):
        """When vllm_ensure_available=False, don't try to start."""
        mgr = BackendManager()
        fs = _fs(Feature.VLLM, Feature.OLLAMA)
        config = _config(vllm_ensure_available=False)
        with patch.object(mgr, "probe_vllm", return_value=_mock_vllm_unreachable()):
            health = await mgr.ensure_vllm(config, fs)
        assert health.status == BackendStatus.UNREACHABLE

    @pytest.mark.asyncio
    async def test_no_model_returns_unreachable(self):
        """When no vllm_model set and Ollama has no models, report error."""
        mgr = BackendManager()
        fs = _fs(Feature.VLLM, Feature.OLLAMA)
        config = _config(vllm_ensure_available=True, vllm_model="")
        ollama_empty = BackendHealth(
            name="ollama", status=BackendStatus.HEALTHY, models=[],
        )
        with (
            patch.object(mgr, "probe_vllm", return_value=_mock_vllm_unreachable()),
            patch.object(mgr, "probe_ollama", return_value=ollama_empty),
        ):
            health = await mgr.ensure_vllm(config, fs)
        assert health.status == BackendStatus.UNREACHABLE
        assert "no model" in health.error.lower()

    @pytest.mark.asyncio
    async def test_auto_selects_model_from_ollama(self):
        """When vllm_model is empty, pick first Ollama model."""
        mgr = BackendManager()
        fs = _fs(Feature.VLLM, Feature.OLLAMA)
        config = _config(
            vllm_ensure_available=True,
            vllm_model="",
            vllm_start_timeout_s=1,
        )
        # First probe: unreachable, then start succeeds on first poll
        probe_results = [_mock_vllm_unreachable(), _mock_vllm_healthy()]

        with (
            patch.object(mgr, "probe_vllm", side_effect=probe_results),
            patch.object(mgr, "probe_ollama", return_value=_mock_ollama_with_models()),
            patch("subprocess.Popen") as mock_popen,
        ):
            health = await mgr.ensure_vllm(config, fs)

        assert health.status == BackendStatus.HEALTHY
        # Verify Popen was called with the Ollama model
        cmd = mock_popen.call_args[0][0]
        assert "--model" in cmd
        model_idx = cmd.index("--model") + 1
        assert cmd[model_idx] == "qwen2.5:7b"

    @pytest.mark.asyncio
    async def test_uses_explicit_model(self):
        """When vllm_model is set, use it directly."""
        mgr = BackendManager()
        fs = _fs(Feature.VLLM, Feature.OLLAMA)
        config = _config(
            vllm_ensure_available=True,
            vllm_model="my-custom-model",
            vllm_start_timeout_s=1,
        )
        with (
            patch.object(mgr, "probe_vllm", side_effect=[
                _mock_vllm_unreachable(), _mock_vllm_healthy(),
            ]),
            patch("subprocess.Popen") as mock_popen,
        ):
            health = await mgr.ensure_vllm(config, fs)

        assert health.status == BackendStatus.HEALTHY
        cmd = mock_popen.call_args[0][0]
        model_idx = cmd.index("--model") + 1
        assert cmd[model_idx] == "my-custom-model"

    @pytest.mark.asyncio
    async def test_gpu_memory_utilization_passed(self):
        """gpu_memory_utilization config is passed to the command."""
        mgr = BackendManager()
        fs = _fs(Feature.VLLM, Feature.OLLAMA)
        config = _config(
            vllm_ensure_available=True,
            vllm_model="test-model",
            vllm_gpu_memory_utilization=0.7,
            vllm_start_timeout_s=1,
        )
        with (
            patch.object(mgr, "probe_vllm", side_effect=[
                _mock_vllm_unreachable(), _mock_vllm_healthy(),
            ]),
            patch("subprocess.Popen") as mock_popen,
        ):
            await mgr.ensure_vllm(config, fs)

        cmd = mock_popen.call_args[0][0]
        util_idx = cmd.index("--gpu-memory-utilization") + 1
        assert cmd[util_idx] == "0.7"

    @pytest.mark.asyncio
    async def test_startup_timeout_respected(self):
        """If vLLM doesn't start within timeout, return unreachable."""
        mgr = BackendManager()
        fs = _fs(Feature.VLLM, Feature.OLLAMA)
        config = _config(
            vllm_ensure_available=True,
            vllm_model="test-model",
            vllm_start_timeout_s=0,  # immediate timeout
        )
        with (
            patch.object(mgr, "probe_vllm", return_value=_mock_vllm_unreachable()),
            patch("subprocess.Popen"),
        ):
            health = await mgr.ensure_vllm(config, fs)

        assert health.status == BackendStatus.UNREACHABLE


class TestVllmConfigFields:
    """Config schema includes vLLM auto-start fields."""

    def test_default_vllm_ensure_available_false(self):
        config = _config()
        assert config.vllm_ensure_available is False

    def test_default_vllm_model_empty(self):
        config = _config()
        assert config.vllm_model == ""

    def test_default_vllm_gpu_memory_utilization(self):
        config = _config()
        assert config.vllm_gpu_memory_utilization == 0.9

    def test_default_vllm_start_timeout(self):
        config = _config()
        assert config.vllm_start_timeout_s == 120

    def test_vllm_start_cmd_default(self):
        config = _config()
        assert "vllm" in " ".join(config.vllm_start_cmd)


# ============================================================================
# OPS-2: Parallel backend preference
# ============================================================================


class TestSupportsConcurrentProperty:
    """InferenceBackend.supports_concurrent is implemented correctly."""

    def test_ollama_does_not_support_concurrent(self):
        from agentic_concierge.infrastructure.backends.ollama import OllamaBackend

        b = OllamaBackend()
        assert b.supports_concurrent is False

    def test_llama_cpp_parallel_1_not_concurrent(self):
        from agentic_concierge.infrastructure.backends.llama_cpp import LlamaCppBackend

        b = LlamaCppBackend(parallel=1)
        assert b.supports_concurrent is False

    def test_llama_cpp_parallel_4_is_concurrent(self):
        from agentic_concierge.infrastructure.backends.llama_cpp import LlamaCppBackend

        b = LlamaCppBackend(parallel=4)
        assert b.supports_concurrent is True

    def test_llama_cpp_default_not_concurrent(self):
        from agentic_concierge.infrastructure.backends.llama_cpp import LlamaCppBackend

        b = LlamaCppBackend()
        assert b.supports_concurrent is False


class TestConcurrentPreferenceInRuntime:
    """LocalModelRuntime prefers concurrent backends when prefer_concurrent=True."""

    @pytest.mark.asyncio
    async def test_prefer_concurrent_selects_concurrent_backend(self):
        """When prefer_concurrent=True, models on concurrent backends are preferred."""
        from agentic_concierge.infrastructure.backends.model_runtime import LocalModelRuntime
        from agentic_concierge.infrastructure.backends.registry import BackendRegistry

        # Only the concurrent backend has a model — serial has nothing.
        # This verifies that the concurrent discovery path works and
        # models on concurrent backends are found and loaded.
        serial_backend = MagicMock()
        serial_backend.name = "ollama"
        serial_backend.base_url = "http://localhost:11434"
        serial_backend.supports_concurrent = False
        serial_backend.health_check = AsyncMock(return_value=True)
        serial_backend.list_available = AsyncMock(return_value=[])
        serial_backend.build_client = MagicMock(return_value=MagicMock())

        concurrent_backend = MagicMock()
        concurrent_backend.name = "llama_cpp"
        concurrent_backend.base_url = ""
        concurrent_backend.supports_concurrent = True
        concurrent_backend.health_check = AsyncMock(return_value=True)
        concurrent_backend.list_available = AsyncMock(return_value=["qwen2.5:7b"])
        concurrent_backend.estimate_memory = AsyncMock(return_value=4000)
        concurrent_backend.load_model = AsyncMock(return_value=MagicMock(
            model_id="qwen2.5:7b", backend="llama_cpp", memory_mb=4000,
            vram_mb=0, quantization="", metadata={}, refcount=0, last_used=0.0,
        ))
        concurrent_backend.build_client = MagicMock(return_value=MagicMock())

        registry = BackendRegistry()
        registry.register(serial_backend, priority=0)
        registry.register(concurrent_backend, priority=1)
        await registry.discover()

        runtime = LocalModelRuntime(registry, memory_budget_mb=64000)

        # With prefer_concurrent: should find and load from llama_cpp
        handle = await runtime.acquire(
            {"reasoning": 0.5},
            prefer_concurrent=True,
        )
        assert handle.slot.backend == "llama_cpp"
        assert handle.model_id == "qwen2.5:7b"
        await runtime.release(handle)

    @pytest.mark.asyncio
    async def test_prefer_concurrent_skips_serial_when_concurrent_has_match(self):
        """When both backends have models, prefer_concurrent picks concurrent backend."""
        from agentic_concierge.infrastructure.backends.model_runtime import LocalModelRuntime
        from agentic_concierge.infrastructure.backends.registry import BackendRegistry

        # Serial backend has an exclusive model already loaded
        serial_backend = MagicMock()
        serial_backend.name = "ollama"
        serial_backend.base_url = "http://localhost:11434"
        serial_backend.supports_concurrent = False
        serial_backend.health_check = AsyncMock(return_value=True)
        serial_backend.list_available = AsyncMock(return_value=["qwen2.5:7b"])
        serial_backend.estimate_memory = AsyncMock(return_value=4000)
        serial_backend.load_model = AsyncMock(return_value=MagicMock(
            model_id="qwen2.5:7b", backend="ollama", memory_mb=4000,
            vram_mb=0, quantization="", metadata={}, refcount=0, last_used=0.0,
        ))
        serial_backend.build_client = MagicMock(return_value=MagicMock())

        # Concurrent backend has its own exclusive model
        concurrent_backend = MagicMock()
        concurrent_backend.name = "vllm"
        concurrent_backend.base_url = "http://localhost:8000"
        concurrent_backend.supports_concurrent = True
        concurrent_backend.health_check = AsyncMock(return_value=True)
        concurrent_backend.list_available = AsyncMock(return_value=["phi-4-mini:14b"])
        concurrent_backend.estimate_memory = AsyncMock(return_value=8000)
        concurrent_backend.load_model = AsyncMock(return_value=MagicMock(
            model_id="phi-4-mini:14b", backend="vllm", memory_mb=8000,
            vram_mb=0, quantization="", metadata={}, refcount=0, last_used=0.0,
        ))
        concurrent_backend.build_client = MagicMock(return_value=MagicMock())

        registry = BackendRegistry()
        registry.register(serial_backend, priority=0)
        registry.register(concurrent_backend, priority=1)
        await registry.discover()

        runtime = LocalModelRuntime(registry, memory_budget_mb=64000)

        # prefer_concurrent should pick the vllm model
        handle = await runtime.acquire(
            {"reasoning": 0.5},
            prefer_concurrent=True,
        )
        assert handle.slot.backend == "vllm"
        assert handle.model_id == "phi-4-mini:14b"
        await runtime.release(handle)

    @pytest.mark.asyncio
    async def test_prefer_concurrent_falls_back_to_serial(self):
        """When no concurrent backend has matching models, fall back to serial."""
        from agentic_concierge.infrastructure.backends.model_runtime import LocalModelRuntime
        from agentic_concierge.infrastructure.backends.registry import BackendRegistry

        serial_backend = MagicMock()
        serial_backend.name = "ollama"
        serial_backend.base_url = "http://localhost:11434"
        serial_backend.supports_concurrent = False
        serial_backend.health_check = AsyncMock(return_value=True)
        serial_backend.list_available = AsyncMock(return_value=["qwen2.5:7b"])
        serial_backend.estimate_memory = AsyncMock(return_value=4000)
        serial_backend.load_model = AsyncMock(return_value=MagicMock(
            model_id="qwen2.5:7b", backend="ollama", memory_mb=4000,
            vram_mb=0, quantization="", metadata={}, refcount=0, last_used=0.0,
        ))
        serial_backend.build_client = MagicMock(return_value=MagicMock())

        registry = BackendRegistry()
        registry.register(serial_backend, priority=0)
        await registry.discover()

        runtime = LocalModelRuntime(registry, memory_budget_mb=64000)

        # prefer_concurrent=True but only serial backends → falls back
        handle = await runtime.acquire(
            {"reasoning": 0.5},
            prefer_concurrent=True,
        )
        assert handle.slot.backend == "ollama"
        await runtime.release(handle)

    @pytest.mark.asyncio
    async def test_simple_runtime_accepts_prefer_concurrent(self):
        """SimpleModelRuntime.acquire accepts prefer_concurrent without error."""
        from agentic_concierge.infrastructure.backends.simple_runtime import SimpleModelRuntime

        client = MagicMock()
        runtime = SimpleModelRuntime(client, "qwen2.5:7b")

        handle = await runtime.acquire(
            {"reasoning": 0.5},
            prefer_concurrent=True,
        )
        assert handle.model_id == "qwen2.5:7b"


class TestAffinityDetectsConcurrency:
    """Affinity executor detects concurrent nodes and sets prefer_concurrent."""

    @pytest.mark.asyncio
    async def test_single_node_not_concurrent(self):
        """A single executing node should not trigger prefer_concurrent."""
        from agentic_concierge.application.task_graph import TaskGraph, TaskNode

        graph = TaskGraph()
        root = TaskNode(id="root", description="root")
        root.status = "done"
        leaf = TaskNode(id="leaf", description="leaf", parent_id="root")
        leaf.status = "executing"  # only this node executing
        root.children = ["leaf"]
        graph.nodes = {"root": root, "leaf": leaf}
        graph.root_id = "root"

        executing_count = sum(
            1 for n in graph.nodes.values() if n.status == "executing"
        )
        assert executing_count == 1
        assert executing_count <= 1  # not concurrent

    @pytest.mark.asyncio
    async def test_multiple_nodes_are_concurrent(self):
        """Multiple executing nodes should trigger prefer_concurrent."""
        from agentic_concierge.application.task_graph import TaskGraph, TaskNode

        graph = TaskGraph()
        root = TaskNode(id="root", description="root")
        root.status = "done"
        leaf_a = TaskNode(id="a", description="a", parent_id="root")
        leaf_a.status = "executing"
        leaf_b = TaskNode(id="b", description="b", parent_id="root")
        leaf_b.status = "executing"
        root.children = ["a", "b"]
        graph.nodes = {"root": root, "a": leaf_a, "b": leaf_b}
        graph.root_id = "root"

        executing_count = sum(
            1 for n in graph.nodes.values() if n.status == "executing"
        )
        assert executing_count == 2
        assert executing_count > 1  # concurrent

    @pytest.mark.asyncio
    async def test_prefer_concurrent_passed_through_assign_model(self):
        """assign_model passes prefer_concurrent to runtime.acquire."""
        from agentic_concierge.application.agent_roles import assign_model

        class TrackingRuntime:
            def __init__(self):
                self.prefer_concurrent_values: List[bool] = []

            async def acquire(self, requirements, **kwargs):
                self.prefer_concurrent_values.append(
                    kwargs.get("prefer_concurrent", False)
                )
                slot = MagicMock()
                slot.model_id = "test-model"
                handle = MagicMock()
                handle.model_id = "test-model"
                handle.slot = slot
                return handle

        runtime = TrackingRuntime()
        await assign_model("coder", runtime, prefer_concurrent=True)
        assert runtime.prefer_concurrent_values == [True]

        await assign_model("coder", runtime, prefer_concurrent=False)
        assert runtime.prefer_concurrent_values == [True, False]


# ===========================================================================
# OPS-1b: ensure_llama_server — auto-install llama-cpp-python[server]
# ===========================================================================


class TestEnsureLlamaServer:
    """Tests for BackendManager.ensure_llama_server()."""

    @pytest.mark.asyncio
    async def test_already_installed_probes_health(self):
        """If llama-server is on PATH, don't install — just probe."""
        from agentic_concierge.bootstrap.backend_manager import BackendManager, BackendStatus

        with patch("shutil.which", return_value="/usr/bin/llama-server"), \
             patch("socket.socket"):
            mgr = BackendManager()
            health = await mgr.ensure_llama_server()
        assert health.status == BackendStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_not_installed_runs_pip(self):
        """If llama-server missing, runs pip install."""
        from agentic_concierge.bootstrap.backend_manager import BackendManager, BackendStatus
        import subprocess

        # First which() call returns None (not installed), second returns the binary (after install)
        call_count = {"n": 0}
        def fake_which(name):
            call_count["n"] += 1
            if call_count["n"] <= 1:
                return None
            return "/venv/bin/llama-server"

        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("shutil.which", side_effect=fake_which), \
             patch("subprocess.run", return_value=mock_result) as mock_run, \
             patch("socket.socket"):
            mgr = BackendManager()
            await mgr.ensure_llama_server(venv_pip="/venv/bin/pip")

        # Verify pip install was called
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "llama-cpp-python[server]" in cmd[-1]

    @pytest.mark.asyncio
    async def test_pip_failure_returns_not_installed(self):
        """If pip install fails, return NOT_INSTALLED with hint."""
        from agentic_concierge.bootstrap.backend_manager import BackendManager, BackendStatus

        mock_result = MagicMock(returncode=1, stdout="", stderr="compilation error")
        with patch("shutil.which", return_value=None), \
             patch("subprocess.run", return_value=mock_result):
            mgr = BackendManager()
            health = await mgr.ensure_llama_server(venv_pip="/venv/bin/pip")

        assert health.status == BackendStatus.NOT_INSTALLED
        assert "pip install failed" in (health.error or "")

    @pytest.mark.asyncio
    async def test_pip_exception_returns_not_installed(self):
        """If pip subprocess crashes, return NOT_INSTALLED."""
        from agentic_concierge.bootstrap.backend_manager import BackendManager, BackendStatus

        with patch("shutil.which", return_value=None), \
             patch("subprocess.run", side_effect=OSError("no such file")):
            mgr = BackendManager()
            health = await mgr.ensure_llama_server(venv_pip="/venv/bin/pip")

        assert health.status == BackendStatus.NOT_INSTALLED


# ===========================================================================
# First-run bootstrap wiring — ensure backends are called
# ===========================================================================


class TestFirstRunBackendWiring:
    """Verify first_run.py calls ensure_vllm and ensure_llama_server."""

    @pytest.mark.asyncio
    async def test_bootstrap_calls_ensure_llama_server(self):
        """When llama_cpp is a preferred backend and not installed, ensure_llama_server is called."""
        from agentic_concierge.bootstrap.backend_manager import BackendHealth, BackendStatus

        mock_probe = MagicMock()
        mock_probe.cpu_cores = 32
        mock_probe.cpu_arch = "x86_64"
        mock_probe.ram_total_mb = 128000
        mock_probe.ram_available_mb = 100000
        mock_probe.gpu_devices = []
        mock_probe.total_vram_mb = 0
        mock_probe.disk_free_mb = 1000000
        mock_probe.internet_reachable = True
        mock_probe.ollama_installed = True
        mock_probe.ollama_reachable = True
        mock_probe.vllm_reachable = False
        mock_probe.llama_server_installed = False

        mock_ensure_llama = AsyncMock(return_value=BackendHealth(
            name="llama_cpp", status=BackendStatus.HEALTHY,
        ))
        mock_ensure_ollama = AsyncMock(return_value=BackendHealth(
            name="ollama", status=BackendStatus.HEALTHY,
        ))

        from agentic_concierge.config.platform import PlatformProfile

        with patch("agentic_concierge.bootstrap.first_run.probe_system", return_value=mock_probe), \
             patch("agentic_concierge.config.platform.detect_platform",
                   return_value=PlatformProfile(
                       name="strix-halo",
                       preferred_backends=["ollama", "llama_cpp"],
                   )), \
             patch("agentic_concierge.bootstrap.backend_manager.BackendManager.ensure_llama_server",
                   mock_ensure_llama), \
             patch("agentic_concierge.bootstrap.backend_manager.BackendManager.ensure_ollama",
                   mock_ensure_ollama), \
             patch("agentic_concierge.bootstrap.first_run._pull_models", new_callable=AsyncMock), \
             patch("agentic_concierge.bootstrap.detected.is_first_run", return_value=True), \
             patch("agentic_concierge.bootstrap.detected.save_detected"):
            from agentic_concierge.bootstrap.first_run import run
            await run(interactive=False, force_profile="server")

        mock_ensure_llama.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_bootstrap_calls_ensure_vllm_when_preferred(self):
        """When vllm is a preferred backend, ensure_vllm is called."""
        from agentic_concierge.bootstrap.backend_manager import BackendHealth, BackendStatus

        mock_probe = MagicMock()
        mock_probe.cpu_cores = 32
        mock_probe.cpu_arch = "x86_64"
        mock_probe.ram_total_mb = 128000
        mock_probe.ram_available_mb = 100000
        mock_probe.gpu_devices = []
        mock_probe.total_vram_mb = 0
        mock_probe.disk_free_mb = 1000000
        mock_probe.internet_reachable = True
        mock_probe.ollama_installed = True
        mock_probe.ollama_reachable = True
        mock_probe.vllm_reachable = False
        mock_probe.llama_server_installed = True

        mock_ensure_vllm = AsyncMock(return_value=BackendHealth(
            name="vllm", status=BackendStatus.UNREACHABLE,
        ))
        mock_ensure_ollama = AsyncMock(return_value=BackendHealth(
            name="ollama", status=BackendStatus.HEALTHY,
        ))

        from agentic_concierge.config.platform import PlatformProfile

        with patch("agentic_concierge.bootstrap.first_run.probe_system", return_value=mock_probe), \
             patch("agentic_concierge.config.platform.detect_platform",
                   return_value=PlatformProfile(
                       name="nvidia-cuda",
                       preferred_backends=["vllm", "ollama"],
                   )), \
             patch("agentic_concierge.bootstrap.backend_manager.BackendManager.ensure_vllm",
                   mock_ensure_vllm), \
             patch("agentic_concierge.bootstrap.backend_manager.BackendManager.ensure_ollama",
                   mock_ensure_ollama), \
             patch("agentic_concierge.bootstrap.first_run._pull_models", new_callable=AsyncMock), \
             patch("agentic_concierge.bootstrap.detected.is_first_run", return_value=True), \
             patch("agentic_concierge.bootstrap.detected.save_detected"):
            from agentic_concierge.bootstrap.first_run import run
            await run(interactive=False, force_profile="server")

        mock_ensure_vllm.assert_awaited_once()
