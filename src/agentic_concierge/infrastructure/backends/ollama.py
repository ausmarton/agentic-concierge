"""OllamaBackend — ``InferenceBackend`` implementation for Ollama.

Consolidates all Ollama-specific API interactions:

- ``GET  /api/tags``     → ``list_available()``
- ``GET  /api/ps``       → ``list_loaded()``
- ``POST /api/generate`` → ``load_model()`` / ``unload_model()``
- ``POST /api/pull``     → ``pull_model()``
- ``GET  /api/tags``     → ``health_check()``
- ``POST /api/show``     → ``estimate_memory()``

The chat completions endpoint (``POST /v1/chat/completions``) is handled by
``OllamaChatClient`` returned from ``build_client()``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

import httpx

from agentic_concierge.application.ports import ChatClient
from agentic_concierge.infrastructure.backends.protocol import ModelSlot

logger = logging.getLogger(__name__)

# Embedding-only model families (should be excluded from chat model lists)
_EMBEDDING_FAMILIES = frozenset(
    s.lower()
    for s in (
        "bge", "bge-m3", "nomic-embed", "mxbai-embed",
        "snowflake-arctic-embed", "all-minilm", "nomic-embed-text",
        "multilingual-e5",
    )
)


def _is_chat_capable(model: Dict[str, Any]) -> bool:
    """Return ``True`` if this Ollama model entry is suitable for chat."""
    name = (model.get("name") or model.get("model") or "").lower()
    details = model.get("details") or {}
    family = (details.get("family") or "").lower()
    families = [f.lower() for f in details.get("families") or []]
    for prefix in _EMBEDDING_FAMILIES:
        if name.startswith(prefix) or family.startswith(prefix) or any(f.startswith(prefix) for f in families):
            return False
    if "embed" in name and "chat" not in name and "instruct" not in name:
        return False
    return True


def _model_name(model: Dict[str, Any]) -> str:
    """Preferred name for an Ollama model entry."""
    return model.get("name") or model.get("model") or ""


def _parse_size_bytes(size_str: str) -> int:
    """Parse a human-readable size string to bytes.

    Handles formats like ``"4.7 GB"``, ``"512 MB"``, ``"3840000000"``.
    Returns 0 if unparseable.
    """
    if not size_str:
        return 0
    # Try as plain integer (bytes)
    try:
        return int(size_str)
    except ValueError:
        pass
    m = re.match(r"([\d.]+)\s*([KMGT]?B)", size_str.strip(), re.IGNORECASE)
    if not m:
        return 0
    val = float(m.group(1))
    unit = m.group(2).upper()
    multipliers = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    return int(val * multipliers.get(unit, 1))


class OllamaBackend:
    """Ollama inference backend (v0.14+).

    All HTTP calls use ``httpx.AsyncClient`` with short timeouts.
    Methods are best-effort and return safe defaults on failure.
    """

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        # Store the API root (no /v1 suffix)
        self._base_url = base_url.rstrip("/")
        if self._base_url.endswith("/v1"):
            self._base_url = self._base_url[:-3]
        self._api_key = ""
        self._timeout_s = 120.0

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def chat_url(self) -> str:
        """The ``/v1`` endpoint used by ``OllamaChatClient``."""
        return f"{self._base_url}/v1"

    # -------------------------------------------------------------------
    # Health check
    # -------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Check if Ollama is healthy by querying ``GET /api/tags``."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self._base_url}/api/tags", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    # -------------------------------------------------------------------
    # Model discovery
    # -------------------------------------------------------------------

    async def list_available(self) -> List[str]:
        """List all chat-capable model IDs via ``GET /api/tags``."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self._base_url}/api/tags", timeout=10.0)
            if resp.status_code != 200:
                return []
            data = resp.json()
            models = data.get("models") or []
            return [_model_name(m) for m in models if _is_chat_capable(m) and _model_name(m)]
        except Exception:
            logger.debug("Failed to list Ollama models at %s", self._base_url, exc_info=True)
            return []

    async def list_available_raw(self) -> List[Dict[str, Any]]:
        """List all model entries (raw dicts) from ``GET /api/tags``.

        Returns the full Ollama model metadata including ``details``,
        ``parameter_size``, etc.  Useful for model selection logic that
        needs size/family information.
        """
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self._base_url}/api/tags", timeout=10.0)
            if resp.status_code != 200:
                return []
            data = resp.json()
            return data.get("models") or []
        except Exception:
            return []

    # -------------------------------------------------------------------
    # Loaded models
    # -------------------------------------------------------------------

    async def list_loaded(self) -> List[ModelSlot]:
        """List currently loaded models via ``GET /api/ps``."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self._base_url}/api/ps", timeout=5.0)
            if resp.status_code != 200:
                return []
            data = resp.json()
            models = data.get("models") or []
            slots = []
            for m in models:
                name = m.get("name") or m.get("model") or ""
                size_bytes = m.get("size") or 0
                size_vram_bytes = m.get("size_vram") or 0
                slots.append(ModelSlot(
                    model_id=name,
                    backend="ollama",
                    memory_mb=size_bytes // (1024 * 1024),
                    vram_mb=size_vram_bytes // (1024 * 1024),
                    quantization=m.get("details", {}).get("quantization_level", ""),
                    metadata=m,
                ))
            return slots
        except Exception:
            logger.debug("Failed to list loaded Ollama models", exc_info=True)
            return []

    # -------------------------------------------------------------------
    # Model lifecycle
    # -------------------------------------------------------------------

    async def load_model(self, model_id: str) -> ModelSlot:
        """Pin a model in memory via ``POST /api/generate``.

        Uses ``keep_alive: "-1"`` to keep the model loaded indefinitely.
        """
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self._base_url}/api/generate",
                    json={"model": model_id, "prompt": "", "keep_alive": "-1"},
                    timeout=120.0,
                )
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_id!r} on Ollama: {e}") from e

        # Query loaded state for memory info
        loaded = await self.list_loaded()
        for slot in loaded:
            if slot.model_id == model_id:
                return slot

        # Model loaded but not found in ps (possible race); return minimal slot
        return ModelSlot(model_id=model_id, backend="ollama")

    async def unload_model(self, model_id: str) -> None:
        """Unload a model via ``POST /api/generate`` with ``keep_alive: "0"``."""
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self._base_url}/api/generate",
                    json={"model": model_id, "prompt": "", "keep_alive": "0"},
                    timeout=30.0,
                )
        except Exception:
            logger.warning("Failed to unload model %s from Ollama", model_id, exc_info=True)

    # -------------------------------------------------------------------
    # Chat client
    # -------------------------------------------------------------------

    def build_client(self, model_id: str) -> ChatClient:
        """Return an ``OllamaChatClient`` configured for this backend."""
        from agentic_concierge.infrastructure.ollama.client import OllamaChatClient
        return OllamaChatClient(
            base_url=self.chat_url,
            api_key=self._api_key,
            timeout_s=self._timeout_s,
        )

    # -------------------------------------------------------------------
    # Memory estimation
    # -------------------------------------------------------------------

    async def estimate_memory(self, model_id: str) -> int:
        """Estimate memory in MB via ``POST /api/show``.

        Parses the ``size`` field from the model info response.
        Returns 0 if the estimate is unavailable.
        """
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self._base_url}/api/show",
                    json={"name": model_id},
                    timeout=10.0,
                )
            if resp.status_code != 200:
                return 0
            data = resp.json()
            # Ollama /api/show returns "size" in bytes for the model file
            size_bytes = data.get("size") or 0
            if isinstance(size_bytes, int) and size_bytes > 0:
                return size_bytes // (1024 * 1024)
            # Some versions use "modelfile_size" or "parameter_size"
            param_size = (data.get("details") or {}).get("parameter_size", "")
            if param_size:
                # rough: 1B params ≈ 500MB at Q4
                m = re.match(r"([\d.]+)\s*([BbMm])", param_size)
                if m:
                    val = float(m.group(1))
                    unit = m.group(2).upper()
                    if unit == "M":
                        val /= 1000.0
                    return int(val * 500)  # rough Q4 estimate
            return 0
        except Exception:
            return 0

    # -------------------------------------------------------------------
    # Model pulling
    # -------------------------------------------------------------------

    async def pull_model(self, model_id: str, *, timeout_s: int = 600) -> bool:
        """Download a model via ``POST /api/pull``.

        Streams the response and waits for completion.  Returns ``True``
        on success.
        """
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/api/pull",
                    json={"name": model_id, "stream": True},
                    timeout=httpx.Timeout(timeout_s, connect=30.0),
                ) as resp:
                    if resp.status_code != 200:
                        return False
                    last_status = ""
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            import json
                            event = json.loads(line)
                            last_status = event.get("status", "")
                            if event.get("error"):
                                logger.error("Ollama pull error: %s", event["error"])
                                return False
                        except Exception:
                            pass
                    return last_status == "success" or "success" in last_status.lower()
        except Exception:
            logger.warning("Failed to pull model %s via Ollama API", model_id, exc_info=True)
            return False
