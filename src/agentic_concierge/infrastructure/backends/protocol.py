"""InferenceBackend protocol — the contract every backend must satisfy.

Each backend (Ollama, llama.cpp, MLX, vLLM, …) implements this protocol.
Application code never imports a concrete backend directly; it depends only
on this protocol shape.

See DESIGN_V2.md §7.3 for rationale and lifecycle semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, runtime_checkable

from agentic_concierge.application.ports import ChatClient


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
