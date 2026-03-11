"""Tests for LlamaCppBackend — process lifecycle, port allocation, health.

Verifies:
- health_check() based on binary availability
- list_available() scans model directory for .gguf files
- load_model() starts a process and returns a ModelSlot
- unload_model() kills the process
- build_client() returns a GenericChatClient
- Port allocation avoids conflicts
- Startup failure handling
- estimate_memory() from file size
- list_loaded() reflects running processes
- Already-loaded model returns existing slot
"""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_concierge.infrastructure.backends.llama_cpp import (
    LlamaCppBackend,
    _ManagedProcess,
    _PORT_RANGE_START,
)
from agentic_concierge.infrastructure.backends.protocol import ModelSlot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_process(pid: int = 1234, returncode: Optional[int] = None) -> MagicMock:
    """Create a mock asyncio.subprocess.Process."""
    proc = MagicMock()
    proc.pid = pid
    proc.returncode = returncode
    proc.send_signal = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=0)
    proc.stderr = None
    return proc


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealthCheck:

    @pytest.mark.asyncio
    async def test_healthy_when_binary_exists(self):
        backend = LlamaCppBackend()
        with patch("shutil.which", return_value="/usr/bin/llama-server"):
            assert await backend.health_check() is True

    @pytest.mark.asyncio
    async def test_unhealthy_when_binary_missing(self):
        backend = LlamaCppBackend()
        with patch("shutil.which", return_value=None):
            assert await backend.health_check() is False


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:

    def test_name(self):
        assert LlamaCppBackend().name == "llama_cpp"

    def test_base_url_empty(self):
        assert LlamaCppBackend().base_url == ""


# ---------------------------------------------------------------------------
# list_available
# ---------------------------------------------------------------------------


class TestListAvailable:

    @pytest.mark.asyncio
    async def test_lists_gguf_files(self, tmp_path: Path):
        (tmp_path / "model-a.gguf").touch()
        (tmp_path / "model-b.gguf").touch()
        (tmp_path / "readme.txt").touch()
        (tmp_path / "subdir").mkdir()

        backend = LlamaCppBackend(model_dir=str(tmp_path))
        models = await backend.list_available()
        assert models == ["model-a.gguf", "model-b.gguf"]

    @pytest.mark.asyncio
    async def test_empty_when_no_dir(self):
        backend = LlamaCppBackend(model_dir="/nonexistent/path")
        assert await backend.list_available() == []

    @pytest.mark.asyncio
    async def test_empty_when_no_model_dir_configured(self):
        backend = LlamaCppBackend(model_dir="")
        assert await backend.list_available() == []


# ---------------------------------------------------------------------------
# Port allocation
# ---------------------------------------------------------------------------


class TestPortAllocation:

    def test_allocates_sequential_ports(self):
        backend = LlamaCppBackend()
        p1 = backend._allocate_port()
        p2 = backend._allocate_port()
        assert p1 == _PORT_RANGE_START
        assert p2 == _PORT_RANGE_START + 1

    def test_skips_used_ports(self):
        backend = LlamaCppBackend()
        # Simulate a process already using the first port
        proc = _fake_process()
        managed = _ManagedProcess("test", "/path", _PORT_RANGE_START, proc)
        backend._processes["test"] = managed

        port = backend._allocate_port()
        assert port == _PORT_RANGE_START + 1

    def test_raises_when_ports_exhausted(self):
        backend = LlamaCppBackend()
        # Fill all ports
        from agentic_concierge.infrastructure.backends.llama_cpp import _PORT_RANGE_END
        for i in range(_PORT_RANGE_START, _PORT_RANGE_END + 1):
            proc = _fake_process(pid=i)
            managed = _ManagedProcess(f"model-{i}", "/path", i, proc)
            backend._processes[f"model-{i}"] = managed

        with pytest.raises(RuntimeError, match="No available ports"):
            backend._allocate_port()


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


class TestLoadModel:

    @pytest.mark.asyncio
    async def test_load_starts_process(self, tmp_path: Path):
        model_file = tmp_path / "test-model.gguf"
        model_file.write_bytes(b"\x00" * 1024)

        backend = LlamaCppBackend(model_dir=str(tmp_path))
        fake_proc = _fake_process()

        with patch("shutil.which", return_value="/usr/bin/llama-server"), \
             patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=fake_proc) as mock_exec, \
             patch.object(backend, "_wait_for_health", new_callable=AsyncMock):

            slot = await backend.load_model("test-model.gguf")

        assert slot.model_id == "test-model.gguf"
        assert slot.backend == "llama_cpp"
        assert slot.metadata["port"] == _PORT_RANGE_START
        assert slot.metadata["pid"] == 1234
        assert "test-model.gguf" in backend._processes

        # Verify llama-server was called with correct args
        call_args = mock_exec.call_args[0]
        assert call_args[0] == "/usr/bin/llama-server"
        assert "--model" in call_args
        assert str(tmp_path / "test-model.gguf") in call_args
        assert "--port" in call_args

    @pytest.mark.asyncio
    async def test_load_already_loaded_returns_existing(self, tmp_path: Path):
        model_file = tmp_path / "test.gguf"
        model_file.write_bytes(b"\x00" * 1024)

        backend = LlamaCppBackend(model_dir=str(tmp_path))
        fake_proc = _fake_process()
        managed = _ManagedProcess("test.gguf", str(model_file), 8100, fake_proc)
        backend._processes["test.gguf"] = managed

        slot = await backend.load_model("test.gguf")
        assert slot.model_id == "test.gguf"
        assert slot.metadata["port"] == 8100

    @pytest.mark.asyncio
    async def test_load_raises_when_binary_missing(self, tmp_path: Path):
        (tmp_path / "test.gguf").touch()
        backend = LlamaCppBackend(model_dir=str(tmp_path))

        with (
            patch("shutil.which", return_value=None),
            patch("importlib.util.find_spec", return_value=None),
        ):
            with pytest.raises(RuntimeError, match="not found.*not installed"):
                await backend.load_model("test.gguf")

    @pytest.mark.asyncio
    async def test_load_raises_when_model_not_found(self):
        backend = LlamaCppBackend(model_dir="/tmp/nonexistent")

        with patch("shutil.which", return_value="/usr/bin/llama-server"):
            with pytest.raises(FileNotFoundError, match="Model file not found"):
                await backend.load_model("missing.gguf")

    @pytest.mark.asyncio
    async def test_load_cleans_up_on_health_timeout(self, tmp_path: Path):
        model_file = tmp_path / "test.gguf"
        model_file.write_bytes(b"\x00" * 1024)

        backend = LlamaCppBackend(model_dir=str(tmp_path))
        fake_proc = _fake_process()

        async def fail_health(*args, **kwargs):
            raise RuntimeError("health check timeout")

        with patch("shutil.which", return_value="/usr/bin/llama-server"), \
             patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=fake_proc), \
             patch.object(backend, "_wait_for_health", side_effect=fail_health), \
             patch.object(backend, "_kill_process", new_callable=AsyncMock) as mock_kill:

            with pytest.raises(RuntimeError, match="health check timeout"):
                await backend.load_model("test.gguf")

        # Process should be killed and removed from tracking
        mock_kill.assert_awaited_once()
        assert "test.gguf" not in backend._processes

    @pytest.mark.asyncio
    async def test_load_includes_gpu_layers(self, tmp_path: Path):
        model_file = tmp_path / "test.gguf"
        model_file.write_bytes(b"\x00" * 1024)

        backend = LlamaCppBackend(model_dir=str(tmp_path), n_gpu_layers=32)
        fake_proc = _fake_process()

        with patch("shutil.which", return_value="/usr/bin/llama-server"), \
             patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=fake_proc) as mock_exec, \
             patch.object(backend, "_wait_for_health", new_callable=AsyncMock):

            await backend.load_model("test.gguf")

        call_args = mock_exec.call_args[0]
        idx = call_args.index("--n-gpu-layers")
        assert call_args[idx + 1] == "32"

    @pytest.mark.asyncio
    async def test_load_skips_gpu_layers_when_zero(self, tmp_path: Path):
        model_file = tmp_path / "test.gguf"
        model_file.write_bytes(b"\x00" * 1024)

        backend = LlamaCppBackend(model_dir=str(tmp_path), n_gpu_layers=0)
        fake_proc = _fake_process()

        with patch("shutil.which", return_value="/usr/bin/llama-server"), \
             patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=fake_proc) as mock_exec, \
             patch.object(backend, "_wait_for_health", new_callable=AsyncMock):

            await backend.load_model("test.gguf")

        call_args = mock_exec.call_args[0]
        assert "--n-gpu-layers" not in call_args


# ---------------------------------------------------------------------------
# unload_model
# ---------------------------------------------------------------------------


class TestUnloadModel:

    @pytest.mark.asyncio
    async def test_unload_kills_process(self):
        backend = LlamaCppBackend()
        fake_proc = _fake_process()
        managed = _ManagedProcess("test.gguf", "/path", 8100, fake_proc)
        backend._processes["test.gguf"] = managed

        with patch.object(backend, "_kill_process", new_callable=AsyncMock) as mock_kill:
            await backend.unload_model("test.gguf")

        mock_kill.assert_awaited_once()
        assert "test.gguf" not in backend._processes

    @pytest.mark.asyncio
    async def test_unload_nonexistent_is_noop(self):
        backend = LlamaCppBackend()
        await backend.unload_model("nonexistent")  # should not raise


# ---------------------------------------------------------------------------
# build_client
# ---------------------------------------------------------------------------


class TestBuildClient:

    def test_returns_generic_client(self):
        backend = LlamaCppBackend()
        fake_proc = _fake_process()
        managed = _ManagedProcess("test.gguf", "/path", 8100, fake_proc)
        backend._processes["test.gguf"] = managed

        client = backend.build_client("test.gguf")
        from agentic_concierge.infrastructure.chat.generic import GenericChatClient
        assert isinstance(client, GenericChatClient)

    def test_raises_when_model_not_loaded(self):
        backend = LlamaCppBackend()
        with pytest.raises(RuntimeError, match="not loaded"):
            backend.build_client("test.gguf")


# ---------------------------------------------------------------------------
# list_loaded
# ---------------------------------------------------------------------------


class TestListLoaded:

    @pytest.mark.asyncio
    async def test_lists_running_processes(self):
        backend = LlamaCppBackend()
        proc1 = _fake_process(pid=100)
        proc2 = _fake_process(pid=200)
        backend._processes["a.gguf"] = _ManagedProcess("a.gguf", "/a", 8100, proc1)
        backend._processes["b.gguf"] = _ManagedProcess("b.gguf", "/b", 8101, proc2)

        loaded = await backend.list_loaded()
        assert len(loaded) == 2
        ids = {s.model_id for s in loaded}
        assert ids == {"a.gguf", "b.gguf"}

    @pytest.mark.asyncio
    async def test_removes_dead_processes(self):
        backend = LlamaCppBackend()
        dead = _fake_process(pid=100, returncode=1)
        alive = _fake_process(pid=200)
        backend._processes["dead.gguf"] = _ManagedProcess("dead.gguf", "/d", 8100, dead)
        backend._processes["alive.gguf"] = _ManagedProcess("alive.gguf", "/a", 8101, alive)

        loaded = await backend.list_loaded()
        assert len(loaded) == 1
        assert loaded[0].model_id == "alive.gguf"
        assert "dead.gguf" not in backend._processes

    @pytest.mark.asyncio
    async def test_empty_when_no_processes(self):
        backend = LlamaCppBackend()
        assert await backend.list_loaded() == []


# ---------------------------------------------------------------------------
# estimate_memory
# ---------------------------------------------------------------------------


class TestEstimateMemory:

    @pytest.mark.asyncio
    async def test_estimates_from_file_size(self, tmp_path: Path):
        model_file = tmp_path / "test.gguf"
        # 1 GB file → ~1.3 GB memory estimate → 1331 MB
        model_file.write_bytes(b"\x00" * (1024 * 1024 * 1024))

        backend = LlamaCppBackend(model_dir=str(tmp_path))
        mem = await backend.estimate_memory("test.gguf")
        assert 1300 <= mem <= 1400  # ~130% of 1024 MB

    @pytest.mark.asyncio
    async def test_returns_zero_for_missing_file(self):
        backend = LlamaCppBackend(model_dir="/nonexistent")
        assert await backend.estimate_memory("missing.gguf") == 0


# ---------------------------------------------------------------------------
# pull_model
# ---------------------------------------------------------------------------


class TestPullModel:

    @pytest.mark.asyncio
    async def test_not_supported(self):
        backend = LlamaCppBackend()
        assert await backend.pull_model("test.gguf") is False


# ---------------------------------------------------------------------------
# Extra args
# ---------------------------------------------------------------------------


class TestExtraArgs:

    @pytest.mark.asyncio
    async def test_extra_args_passed_to_process(self, tmp_path: Path):
        model_file = tmp_path / "test.gguf"
        model_file.write_bytes(b"\x00" * 1024)

        backend = LlamaCppBackend(
            model_dir=str(tmp_path),
            extra_args=["--no-mmap", "--flash-attn"],
        )
        fake_proc = _fake_process()

        with patch("shutil.which", return_value="/usr/bin/llama-server"), \
             patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=fake_proc) as mock_exec, \
             patch.object(backend, "_wait_for_health", new_callable=AsyncMock):

            await backend.load_model("test.gguf")

        call_args = mock_exec.call_args[0]
        assert "--no-mmap" in call_args
        assert "--flash-attn" in call_args


# ---------------------------------------------------------------------------
# Parallel slots (--parallel N)
# ---------------------------------------------------------------------------


class TestParallelSlots:

    def test_default_parallel_is_one(self):
        backend = LlamaCppBackend()
        assert backend._parallel == 1

    def test_parallel_stored(self):
        backend = LlamaCppBackend(parallel=4)
        assert backend._parallel == 4

    def test_parallel_zero_clamped_to_one(self):
        backend = LlamaCppBackend(parallel=0)
        assert backend._parallel == 1

    def test_parallel_negative_clamped_to_one(self):
        backend = LlamaCppBackend(parallel=-3)
        assert backend._parallel == 1

    @pytest.mark.asyncio
    async def test_load_passes_parallel_flag(self, tmp_path: Path):
        """--parallel N appears in subprocess args when parallel=N."""
        model_file = tmp_path / "test.gguf"
        model_file.write_bytes(b"\x00" * 1024)

        backend = LlamaCppBackend(model_dir=str(tmp_path), parallel=4)
        fake_proc = _fake_process()

        with patch("shutil.which", return_value="/usr/bin/llama-server"), \
             patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=fake_proc) as mock_exec, \
             patch.object(backend, "_wait_for_health", new_callable=AsyncMock):

            await backend.load_model("test.gguf")

        call_args = mock_exec.call_args[0]
        idx = call_args.index("--parallel")
        assert call_args[idx + 1] == "4"

    @pytest.mark.asyncio
    async def test_load_ctx_size_scaled_by_parallel(self, tmp_path: Path):
        """--ctx-size should be ctx_size * parallel."""
        model_file = tmp_path / "test.gguf"
        model_file.write_bytes(b"\x00" * 1024)

        backend = LlamaCppBackend(model_dir=str(tmp_path), ctx_size=8192, parallel=4)
        fake_proc = _fake_process()

        with patch("shutil.which", return_value="/usr/bin/llama-server"), \
             patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=fake_proc) as mock_exec, \
             patch.object(backend, "_wait_for_health", new_callable=AsyncMock):

            await backend.load_model("test.gguf")

        call_args = mock_exec.call_args[0]
        idx = call_args.index("--ctx-size")
        assert call_args[idx + 1] == str(8192 * 4)  # 32768

    @pytest.mark.asyncio
    async def test_load_default_parallel_one_unscaled_ctx(self, tmp_path: Path):
        """parallel=1 → --ctx-size is just ctx_size (unscaled)."""
        model_file = tmp_path / "test.gguf"
        model_file.write_bytes(b"\x00" * 1024)

        backend = LlamaCppBackend(model_dir=str(tmp_path), ctx_size=4096, parallel=1)
        fake_proc = _fake_process()

        with patch("shutil.which", return_value="/usr/bin/llama-server"), \
             patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=fake_proc) as mock_exec, \
             patch.object(backend, "_wait_for_health", new_callable=AsyncMock):

            await backend.load_model("test.gguf")

        call_args = mock_exec.call_args[0]
        idx_ctx = call_args.index("--ctx-size")
        assert call_args[idx_ctx + 1] == "4096"
        idx_par = call_args.index("--parallel")
        assert call_args[idx_par + 1] == "1"

    @pytest.mark.asyncio
    async def test_parallel_and_extra_args_coexist(self, tmp_path: Path):
        """parallel and extra_args don't interfere with each other."""
        model_file = tmp_path / "test.gguf"
        model_file.write_bytes(b"\x00" * 1024)

        backend = LlamaCppBackend(
            model_dir=str(tmp_path),
            ctx_size=2048,
            parallel=3,
            extra_args=["--flash-attn"],
        )
        fake_proc = _fake_process()

        with patch("shutil.which", return_value="/usr/bin/llama-server"), \
             patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=fake_proc) as mock_exec, \
             patch.object(backend, "_wait_for_health", new_callable=AsyncMock):

            await backend.load_model("test.gguf")

        call_args = mock_exec.call_args[0]
        assert call_args[call_args.index("--parallel") + 1] == "3"
        assert call_args[call_args.index("--ctx-size") + 1] == str(2048 * 3)
        assert "--flash-attn" in call_args
