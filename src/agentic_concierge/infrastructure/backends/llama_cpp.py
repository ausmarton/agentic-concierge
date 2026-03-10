"""LlamaCppBackend — ``InferenceBackend`` for managed llama.cpp server processes.

Each model gets its own ``llama-server`` process with a dynamically
allocated port.  The backend manages process lifecycle: ``load_model()``
starts a process, ``unload_model()`` kills it.

The ``build_client()`` method returns a ``GenericChatClient`` pointed at
the process's ``/v1`` endpoint, which implements the standard OpenAI
chat completions API.

See DESIGN_V2.md §7.3 and Appendix B for rationale.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentic_concierge.application.ports import ChatClient
from agentic_concierge.infrastructure.backends.protocol import ModelSlot

logger = logging.getLogger(__name__)

# Port range for dynamically allocated llama-server processes.
_PORT_RANGE_START = 8100
_PORT_RANGE_END = 8199

# How long to wait for the server to become healthy after starting.
_STARTUP_TIMEOUT_S = 120.0

# Health check poll interval during startup.
_HEALTH_POLL_INTERVAL_S = 0.5


class _ManagedProcess:
    """Tracks a running llama-server process."""

    def __init__(
        self,
        model_id: str,
        model_path: str,
        port: int,
        process: asyncio.subprocess.Process,
    ) -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.port = port
        self.process = process


class LlamaCppBackend:
    """Managed llama.cpp server backend.

    Each loaded model runs in its own ``llama-server`` process with a
    unique port.  GPU offloading is controlled via ``--n-gpu-layers``.

    Args:
        model_dir: Directory containing ``.gguf`` model files.
        binary: Path or name of the ``llama-server`` binary.
        n_gpu_layers: Number of layers to offload to GPU (``-1`` = all).
        ctx_size: Context window size per slot.
        parallel: Number of concurrent request slots (``--parallel`` flag).
            The actual KV cache size is ``ctx_size * parallel``.
        extra_args: Additional CLI arguments for ``llama-server``.
    """

    def __init__(
        self,
        model_dir: str = "",
        binary: str = "llama-server",
        n_gpu_layers: int = -1,
        ctx_size: int = 8192,
        parallel: int = 1,
        extra_args: Optional[List[str]] = None,
    ) -> None:
        self._model_dir = model_dir
        self._binary = binary
        self._n_gpu_layers = n_gpu_layers
        self._ctx_size = ctx_size
        self._parallel = max(1, parallel)
        self._extra_args = extra_args or []
        self._processes: Dict[str, _ManagedProcess] = {}
        self._next_port = _PORT_RANGE_START

    @property
    def name(self) -> str:
        return "llama_cpp"

    @property
    def base_url(self) -> str:
        return ""  # no fixed URL — each process has its own port

    @property
    def supports_concurrent(self) -> bool:
        return self._parallel > 1

    # -------------------------------------------------------------------
    # Health check
    # -------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Check if the ``llama-server`` binary is available."""
        return shutil.which(self._binary) is not None

    # -------------------------------------------------------------------
    # Model discovery
    # -------------------------------------------------------------------

    async def list_available(self) -> List[str]:
        """List ``.gguf`` files in the model directory."""
        if not self._model_dir or not os.path.isdir(self._model_dir):
            return []
        result = []
        for entry in os.scandir(self._model_dir):
            if entry.is_file() and entry.name.endswith(".gguf"):
                result.append(entry.name)
        return sorted(result)

    # -------------------------------------------------------------------
    # Loaded models
    # -------------------------------------------------------------------

    async def list_loaded(self) -> List[ModelSlot]:
        """List currently running model processes."""
        slots = []
        for model_id, proc in list(self._processes.items()):
            if proc.process.returncode is not None:
                # Process exited unexpectedly
                del self._processes[model_id]
                continue
            slots.append(ModelSlot(
                model_id=model_id,
                backend="llama_cpp",
                metadata={"port": proc.port, "pid": proc.process.pid},
            ))
        return slots

    # -------------------------------------------------------------------
    # Model lifecycle
    # -------------------------------------------------------------------

    async def load_model(self, model_id: str) -> ModelSlot:
        """Start a ``llama-server`` process for the given model.

        Blocks until the server's ``/health`` endpoint responds (up to
        ``_STARTUP_TIMEOUT_S`` seconds).

        Raises:
            RuntimeError: If the binary is not found, the model file
                doesn't exist, or the server fails to start.
        """
        if model_id in self._processes:
            proc = self._processes[model_id]
            if proc.process.returncode is None:
                return ModelSlot(
                    model_id=model_id,
                    backend="llama_cpp",
                    metadata={"port": proc.port, "pid": proc.process.pid},
                )

        binary_path = shutil.which(self._binary)
        if binary_path is None:
            raise RuntimeError(
                f"llama-server binary {self._binary!r} not found in PATH"
            )

        model_path = self._resolve_model_path(model_id)
        port = self._allocate_port()

        # KV cache size = ctx_size_per_slot * parallel_slots
        total_ctx = self._ctx_size * self._parallel
        cmd = [
            binary_path,
            "--model", model_path,
            "--port", str(port),
            "--host", "127.0.0.1",
            "--ctx-size", str(total_ctx),
            "--parallel", str(self._parallel),
        ]
        if self._n_gpu_layers != 0:
            cmd.extend(["--n-gpu-layers", str(self._n_gpu_layers)])
        cmd.extend(self._extra_args)

        logger.info(
            "Starting llama-server for %s on port %d (pid pending)",
            model_id, port,
        )

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to start llama-server for {model_id!r}: {e}"
            ) from e

        managed = _ManagedProcess(
            model_id=model_id,
            model_path=model_path,
            port=port,
            process=process,
        )
        self._processes[model_id] = managed

        # Wait for server to become healthy
        try:
            await self._wait_for_health(port, process)
        except Exception:
            await self._kill_process(managed)
            del self._processes[model_id]
            raise

        logger.info(
            "llama-server for %s ready on port %d (pid %d)",
            model_id, port, process.pid,
        )

        mem = await self.estimate_memory(model_id)
        return ModelSlot(
            model_id=model_id,
            backend="llama_cpp",
            memory_mb=mem,
            metadata={"port": port, "pid": process.pid},
        )

    async def unload_model(self, model_id: str) -> None:
        """Kill the ``llama-server`` process for the given model."""
        proc = self._processes.pop(model_id, None)
        if proc is not None:
            await self._kill_process(proc)
            logger.info("Stopped llama-server for %s (pid %d)", model_id, proc.process.pid)

    # -------------------------------------------------------------------
    # Chat client
    # -------------------------------------------------------------------

    def build_client(self, model_id: str) -> ChatClient:
        """Return a ``GenericChatClient`` pointed at the model's server port."""
        proc = self._processes.get(model_id)
        if proc is None:
            raise RuntimeError(
                f"Model {model_id!r} is not loaded on llama_cpp backend"
            )
        from agentic_concierge.infrastructure.chat.generic import GenericChatClient
        return GenericChatClient(base_url=f"http://127.0.0.1:{proc.port}/v1")

    # -------------------------------------------------------------------
    # Memory estimation
    # -------------------------------------------------------------------

    async def estimate_memory(self, model_id: str) -> int:
        """Estimate memory in MB from ``.gguf`` file size.

        Rough heuristic: file size ≈ model weight size.  Actual memory
        usage is ~120-130% of file size due to KV cache and buffers.
        Returns 0 if the file is not found.
        """
        try:
            model_path = self._resolve_model_path(model_id)
            size_bytes = os.path.getsize(model_path)
            # ~130% of file size for runtime overhead
            return int(size_bytes * 1.3 / (1024 * 1024))
        except (FileNotFoundError, OSError):
            return 0

    # -------------------------------------------------------------------
    # Model pulling (not supported)
    # -------------------------------------------------------------------

    async def pull_model(self, model_id: str, *, timeout_s: int = 600) -> bool:
        """Not supported — models must be pre-downloaded as ``.gguf`` files."""
        return False

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _resolve_model_path(self, model_id: str) -> str:
        """Resolve a model ID to a file path.

        If ``model_id`` is an absolute path or already exists, use it
        directly.  Otherwise, look in ``_model_dir``.
        """
        if os.path.isabs(model_id) and os.path.isfile(model_id):
            return model_id
        if os.path.isfile(model_id):
            return os.path.abspath(model_id)
        if self._model_dir:
            path = os.path.join(self._model_dir, model_id)
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(
            f"Model file not found: {model_id!r} "
            f"(model_dir={self._model_dir!r})"
        )

    def _allocate_port(self) -> int:
        """Allocate the next available port."""
        used = {p.port for p in self._processes.values()}
        port = self._next_port
        while port in used and port <= _PORT_RANGE_END:
            port += 1
        if port > _PORT_RANGE_END:
            raise RuntimeError(
                f"No available ports in range {_PORT_RANGE_START}-{_PORT_RANGE_END}"
            )
        self._next_port = port + 1
        return port

    async def _wait_for_health(
        self,
        port: int,
        process: asyncio.subprocess.Process,
    ) -> None:
        """Poll the server's ``/health`` endpoint until it responds.

        Raises ``RuntimeError`` on timeout or if the process exits.
        """
        import httpx

        url = f"http://127.0.0.1:{port}/health"
        deadline = asyncio.get_event_loop().time() + _STARTUP_TIMEOUT_S

        while asyncio.get_event_loop().time() < deadline:
            # Check if process died
            if process.returncode is not None:
                stderr = ""
                if process.stderr:
                    stderr = (await process.stderr.read()).decode(errors="replace")
                raise RuntimeError(
                    f"llama-server exited with code {process.returncode}: {stderr[:500]}"
                )
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url, timeout=2.0)
                if resp.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(_HEALTH_POLL_INTERVAL_S)

        raise RuntimeError(
            f"llama-server on port {port} did not become healthy "
            f"within {_STARTUP_TIMEOUT_S}s"
        )

    async def _kill_process(self, proc: _ManagedProcess) -> None:
        """Terminate a managed process gracefully, then force-kill if needed."""
        try:
            proc.process.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                proc.process.kill()
                await proc.process.wait()
        except ProcessLookupError:
            pass  # already dead
        except Exception:
            logger.warning(
                "Error killing llama-server pid %d", proc.process.pid,
                exc_info=True,
            )
