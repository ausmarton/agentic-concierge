"""InferenceBackend protocol and model lifecycle types.

Each backend (Ollama, llama.cpp, MLX, vLLM, …) implements the
``InferenceBackend`` protocol.  Application code never imports a concrete
backend directly; it depends only on the protocol shape.

``ModelHandle`` is the async context manager returned by
``ModelRuntime.acquire()``.  ``ModelInfo`` and ``RuntimeStatus`` are
inspection types.

See DESIGN_V2.md §7.1 and §7.3 for rationale and lifecycle semantics.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from agentic_concierge.application.ports import ChatClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model slot (loaded model)
# ---------------------------------------------------------------------------


@dataclass
class ModelSlot:
    """A loaded model slot returned by ``load_model()``.

    Represents a model currently occupying memory on the backend.
    """

    model_id: str
    backend: str                     # "ollama", "llama_cpp", "mlx", "vllm"
    memory_mb: int = 0               # estimated memory consumption in MB
    vram_mb: int = 0                 # VRAM portion (0 if CPU-only)
    quantization: str = ""           # e.g. "Q4_K_M", "Q5_K_S"
    metadata: Dict[str, Any] = field(default_factory=dict)
    refcount: int = 0               # number of active users
    last_used: float = 0.0          # time.monotonic() of last release


# ---------------------------------------------------------------------------
# Model handle (acquire/release context manager)
# ---------------------------------------------------------------------------


class ModelHandle:
    """Opaque handle returned by ``ModelRuntime.acquire()``.

    Holds a reference to a loaded model and its chat client.  Must be
    released (via ``release()`` or the async context manager) when no
    longer needed — this decrements the model's refcount, making it
    eligible for eviction.

    Usage::

        async with await runtime.acquire(requirements) as handle:
            response = await handle.chat_client.chat(messages, handle.model_id)
        # auto-released here
    """

    def __init__(
        self,
        slot: ModelSlot,
        chat_client: ChatClient,
        runtime: Any = None,  # Optional back-reference to ModelRuntime for auto-release
    ) -> None:
        self.slot = slot
        self.chat_client = chat_client
        self.model_id = slot.model_id
        self._runtime = runtime
        self._released = False

    async def release(self) -> None:
        """Release this handle, decrementing the model's refcount."""
        if self._released:
            return
        self._released = True
        if self._runtime is not None:
            await self._runtime.release(self)
        else:
            # No runtime reference — just decrement directly
            self.slot.refcount = max(0, self.slot.refcount - 1)
            self.slot.last_used = time.monotonic()

    async def __aenter__(self) -> "ModelHandle":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.release()


# ---------------------------------------------------------------------------
# Inspection types
# ---------------------------------------------------------------------------


@dataclass
class ModelInfo:
    """Information about a model known to the runtime (loaded or available)."""

    model_id: str
    backend: str                           # "ollama", "llama_cpp", "mlx"
    available: bool = True                 # available to load (downloaded/cached)
    loaded: bool = False                   # currently in memory
    capabilities: Dict[str, float] = field(default_factory=dict)
    estimated_memory_mb: int = 0
    supports_tool_calling: bool = True


@dataclass
class RuntimeStatus:
    """Snapshot of model runtime resource usage."""

    loaded_models: List[ModelSlot] = field(default_factory=list)
    total_memory_mb: int = 0
    used_memory_mb: int = 0

    @property
    def free_memory_mb(self) -> int:
        return max(0, self.total_memory_mb - self.used_memory_mb)


@runtime_checkable
class InferenceBackend(Protocol):
    """Abstraction over inference serving backends.

    Each method is async because backend operations involve I/O (HTTP calls,
    subprocess management, file system).  All methods are best-effort —
    implementations should raise clear exceptions on failure rather than
    returning ambiguous results.
    """

    @property
    def name(self) -> str:
        """Backend identifier, e.g. ``"ollama"``, ``"llama_cpp"``."""
        ...

    @property
    def base_url(self) -> str:
        """Base URL for the backend's API (empty string for in-process)."""
        ...

    async def health_check(self) -> bool:
        """Return ``True`` if the backend is healthy and responsive.

        Must complete within a few seconds; never raises.
        """
        ...

    async def list_available(self) -> List[str]:
        """List model IDs available to load (downloaded/cached).

        For Ollama this is ``GET /api/tags``; for llama.cpp it would scan
        the model directory.
        """
        ...

    async def list_loaded(self) -> List[ModelSlot]:
        """List currently loaded models with memory info.

        For Ollama this is ``GET /api/ps``.
        """
        ...

    async def load_model(self, model_id: str) -> ModelSlot:
        """Load a model into memory and return its slot info.

        For Ollama: ``POST /api/generate`` with empty prompt and
        ``keep_alive: "-1"`` to pin the model in memory.
        """
        ...

    async def unload_model(self, model_id: str) -> None:
        """Unload a model to free memory.

        For Ollama: ``POST /api/generate`` with ``keep_alive: "0"``.
        """
        ...

    def build_client(self, model_id: str) -> ChatClient:
        """Create a ``ChatClient`` bound to the given model.

        The returned client is ready to use for chat completions.
        The model should already be loaded (or the backend must handle
        lazy loading transparently).
        """
        ...

    async def estimate_memory(self, model_id: str) -> int:
        """Estimate memory in MB required to load this model.

        Returns 0 if the estimate is unavailable.
        """
        ...

    async def pull_model(self, model_id: str, *, timeout_s: int = 600) -> bool:
        """Download a model.  Returns ``True`` on success.

        Not all backends support pulling (e.g. llama.cpp expects models
        to be pre-downloaded).  Backends that don't support this should
        return ``False``.
        """
        ...
