"""LocalModelRuntime — concrete ModelRuntime for single-machine model lifecycle.

Wraps ``BackendRegistry`` and model capability profiles to manage model
loading, capability matching, and refcounting.  Application code acquires a
``ModelHandle`` for each model it needs; the runtime handles backend
selection, capability matching, and reference tracking.

Usage::

    runtime = LocalModelRuntime(registry)
    async with await runtime.acquire({"reasoning": 0.7}) as handle:
        response = await handle.chat_client.chat(messages, handle.model_id)
    # auto-released here

See DESIGN_V2.md §7.1–§7.4 for rationale.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from agentic_concierge.infrastructure.backends.protocol import (
    ModelHandle,
    ModelInfo,
    ModelSlot,
    RuntimeStatus,
)
from agentic_concierge.infrastructure.backends.registry import BackendRegistry
from agentic_concierge.infrastructure.model_profiles import get_profile, match_models

logger = logging.getLogger(__name__)


class LocalModelRuntime:
    """Concrete ModelRuntime for single-machine model lifecycle management.

    Manages model loading, capability matching, and refcounting across
    all healthy backends in the ``BackendRegistry``.  Serialises
    ``acquire()`` / ``release()`` calls with an ``asyncio.Lock`` to
    prevent races on slot state.
    """

    def __init__(self, registry: BackendRegistry) -> None:
        self._registry = registry
        self._slots: Dict[str, ModelSlot] = {}  # model_id → loaded slot
        self._lock = asyncio.Lock()

    # -------------------------------------------------------------------
    # Acquire / Release
    # -------------------------------------------------------------------

    async def acquire(
        self,
        requirements: Dict[str, float],
        *,
        prefer_model: Optional[str] = None,
        require_tool_calling: bool = True,
        exclude_models: Optional[List[str]] = None,
        timeout_s: float = 30.0,
    ) -> ModelHandle:
        """Ensure a model matching *requirements* is loaded; return a handle.

        Prefer already-loaded models to avoid loading latency.  If no loaded
        model matches, the best available model is loaded via its backend.

        Args:
            requirements: Capability name → minimum score (0.0–1.0).
            prefer_model: Hint to prefer a specific model ID.
            require_tool_calling: Only consider tool-calling-capable models.
            exclude_models: Model IDs to exclude (e.g. for reviewer
                must-differ-from constraint).
            timeout_s: Reserved for future use (eviction queue timeout).

        Raises:
            RuntimeError: If no model matching requirements is available.
        """
        async with self._lock:
            return await self._acquire_locked(
                requirements,
                prefer_model=prefer_model,
                require_tool_calling=require_tool_calling,
                exclude_models=exclude_models,
            )

    async def _acquire_locked(
        self,
        requirements: Dict[str, float],
        *,
        prefer_model: Optional[str],
        require_tool_calling: bool,
        exclude_models: Optional[List[str]],
    ) -> ModelHandle:
        excluded = set(exclude_models or [])

        # --- Fast path: reuse already-loaded model (no I/O) ---
        loaded_ids = [mid for mid in self._slots if mid not in excluded]
        selected = self._select_model(
            loaded_ids, requirements, require_tool_calling, prefer_model,
        )
        if selected is not None:
            return self._handle_for_loaded(selected)

        # --- Slow path: discover available models from backends ---
        model_backends = await self._discover_available()
        all_ids = [mid for mid in model_backends if mid not in excluded]
        selected = self._select_model(
            all_ids, requirements, require_tool_calling, prefer_model,
        )
        if selected is None:
            raise RuntimeError(
                f"No model matching requirements {requirements} is available. "
                f"Available models: {all_ids}"
            )

        # Already loaded? (available list includes loaded models)
        if selected in self._slots:
            return self._handle_for_loaded(selected)

        # Load model via backend
        backend = model_backends[selected]
        slot = await backend.load_model(selected)
        slot.refcount = 1
        self._slots[selected] = slot

        client = backend.build_client(selected)
        return ModelHandle(slot=slot, chat_client=client, runtime=self)

    async def release(self, handle: ModelHandle) -> None:
        """Decrement refcount; model stays loaded but becomes eligible for eviction."""
        handle.slot.refcount = max(0, handle.slot.refcount - 1)
        handle.slot.last_used = time.monotonic()

    # -------------------------------------------------------------------
    # Inspection
    # -------------------------------------------------------------------

    async def inventory(self) -> List[ModelInfo]:
        """List all known models (loaded and available-to-load) with capabilities."""
        result: List[ModelInfo] = []
        seen: set = set()
        for backend in self._registry.healthy_backends():
            try:
                available = await backend.list_available()
            except Exception:
                logger.warning(
                    "Backend %s failed to list available models",
                    backend.name, exc_info=True,
                )
                continue
            for model_id in available:
                if model_id in seen:
                    continue
                seen.add(model_id)
                profile = get_profile(model_id)
                loaded = model_id in self._slots
                if loaded:
                    mem = self._slots[model_id].memory_mb
                else:
                    try:
                        mem = await backend.estimate_memory(model_id)
                    except Exception:
                        mem = 0
                result.append(ModelInfo(
                    model_id=model_id,
                    backend=backend.name,
                    available=True,
                    loaded=loaded,
                    capabilities=dict(profile.capabilities),
                    estimated_memory_mb=mem,
                    supports_tool_calling=profile.supports_tool_calling,
                ))
        return result

    async def status(self) -> RuntimeStatus:
        """Current resource usage snapshot."""
        loaded = list(self._slots.values())
        used = sum(s.memory_mb for s in loaded)
        return RuntimeStatus(
            loaded_models=loaded,
            used_memory_mb=used,
        )

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _select_model(
        self,
        candidates: List[str],
        requirements: Dict[str, float],
        require_tool_calling: bool,
        prefer_model: Optional[str],
    ) -> Optional[str]:
        """Select best model from candidates, honoring ``prefer_model`` hint.

        If ``prefer_model`` is among the candidates and passes the
        tool-calling filter, it is selected immediately.  Otherwise,
        ``match_models()`` picks the best capability match.
        """
        if not candidates:
            return None

        # prefer_model hint: if available and passes filters, use immediately
        if prefer_model and prefer_model in candidates:
            if not require_tool_calling:
                return prefer_model
            profile = get_profile(prefer_model)
            if profile.supports_tool_calling:
                return prefer_model

        # Regular capability-based selection
        return match_models(
            candidates, requirements,
            require_tool_calling=require_tool_calling,
        )

    def _handle_for_loaded(self, model_id: str) -> ModelHandle:
        """Create a ``ModelHandle`` for an already-loaded model."""
        slot = self._slots[model_id]
        slot.refcount += 1
        backend = self._registry.get_backend(slot.backend)
        if backend is None:
            raise RuntimeError(
                f"Backend {slot.backend!r} for loaded model {model_id!r} "
                f"not found in registry"
            )
        client = backend.build_client(model_id)
        return ModelHandle(slot=slot, chat_client=client, runtime=self)

    async def _discover_available(self) -> Dict[str, Any]:
        """Return ``model_id → backend`` for all available models.

        Queries all healthy backends in priority order.  First backend wins
        for models available on multiple backends.
        """
        result: Dict[str, Any] = {}
        for backend in self._registry.healthy_backends():
            try:
                models = await backend.list_available()
            except Exception:
                logger.warning(
                    "Backend %s failed to list available models",
                    backend.name, exc_info=True,
                )
                continue
            for model_id in models:
                if model_id not in result:
                    result[model_id] = backend
        return result
