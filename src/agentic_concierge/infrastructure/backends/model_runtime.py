"""LocalModelRuntime — concrete ModelRuntime for single-machine model lifecycle.

Wraps ``BackendRegistry`` and model capability profiles to manage model
loading, capability matching, refcounting, and LRU eviction.  Application
code acquires a ``ModelHandle`` for each model it needs; the runtime handles
backend selection, capability matching, memory management, and reference
tracking.

Usage::

    runtime = LocalModelRuntime(registry, memory_budget_mb=64_000)
    async with await runtime.acquire({"reasoning": 0.7}) as handle:
        response = await handle.chat_client.chat(messages, handle.model_id)
    # auto-released here

See DESIGN_V2.md §7.1–§7.4 for rationale.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
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

# Reserve 15% of total RAM for OS and other processes.
_OS_RESERVE_FRACTION = 0.15

# APU VRAM correction threshold: when a GPU reports less than this
# amount of VRAM, it's likely an APU sharing system RAM.
_APU_VRAM_THRESHOLD_MB = 4096

# Minimum system RAM (MB) to apply APU correction.
_APU_RAM_THRESHOLD_MB = 32_768


def detect_memory_budget_mb() -> int:
    """Detect total system RAM and return a conservative memory budget in MB.

    Returns 85% of total system RAM (leaving room for OS and other
    processes).  Returns 0 if detection fails.
    """
    try:
        system = platform.system()
        if system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        # "MemTotal:   131935860 kB"
                        parts = line.split()
                        kb = int(parts[1])
                        total_mb = kb // 1024
                        return int(total_mb * (1 - _OS_RESERVE_FRACTION))
        elif system == "Darwin":
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                total_bytes = int(result.stdout.strip())
                total_mb = total_bytes // (1024 * 1024)
                return int(total_mb * (1 - _OS_RESERVE_FRACTION))
    except Exception:
        logger.debug("Failed to detect system memory", exc_info=True)
    return 0


class LocalModelRuntime:
    """Concrete ModelRuntime for single-machine model lifecycle management.

    Manages model loading, capability matching, refcounting, and LRU
    eviction across all healthy backends in the ``BackendRegistry``.
    Serialises ``acquire()`` / ``release()`` calls with an
    ``asyncio.Lock`` to prevent races on slot state.

    When ``acquire()`` needs to load a new model and the memory budget
    would be exceeded, it evicts the least-recently-used models with
    ``refcount == 0`` until enough memory is free.
    """

    def __init__(
        self,
        registry: BackendRegistry,
        memory_budget_mb: Optional[int] = None,
    ) -> None:
        self._registry = registry
        self._slots: Dict[str, ModelSlot] = {}  # model_id → loaded slot
        self._lock = asyncio.Lock()
        self._memory_budget_mb = (
            memory_budget_mb
            if memory_budget_mb is not None
            else detect_memory_budget_mb()
        )

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
        prefer_concurrent: bool = False,
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
            prefer_concurrent: Prefer models on backends that support
                concurrent requests (e.g. vLLM, llama_cpp with parallel>1).
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
                prefer_concurrent=prefer_concurrent,
            )

    async def _acquire_locked(
        self,
        requirements: Dict[str, float],
        *,
        prefer_model: Optional[str],
        require_tool_calling: bool,
        exclude_models: Optional[List[str]],
        prefer_concurrent: bool = False,
    ) -> ModelHandle:
        excluded = set(exclude_models or [])

        # --- Fast path: reuse already-loaded model (no I/O) ---
        # Skip the fast path when prefer_model is specified but not loaded;
        # this allows the slow path to discover and load the preferred model.
        loaded_ids = [mid for mid in self._slots if mid not in excluded]

        # When prefer_concurrent, only consider loaded models on concurrent
        # backends for the fast path; if none qualify, fall through to slow
        # path where concurrent backends are tried first.
        if prefer_concurrent:
            loaded_ids = [
                mid for mid in loaded_ids
                if self._is_on_concurrent_backend(mid)
            ]

        skip_fast = bool(prefer_model and prefer_model not in loaded_ids)
        if not skip_fast:
            selected = self._select_model(
                loaded_ids, requirements, require_tool_calling, prefer_model,
            )
            if selected is not None:
                return self._handle_for_loaded(selected)

        # --- Slow path: discover available models from backends ---
        selected: Optional[str] = None

        # When prefer_concurrent, try concurrent-capable backends first
        if prefer_concurrent:
            concurrent_map = await self._discover_available(concurrent_only=True)
            concurrent_ids = [mid for mid in concurrent_map if mid not in excluded]
            if concurrent_ids:
                selected = self._select_model(
                    concurrent_ids, requirements, require_tool_calling, prefer_model,
                )
                if selected is not None:
                    logger.info(
                        "Concurrent preference: selected %s from %s backend",
                        selected, concurrent_map[selected].name,
                    )

        # Fallback to all backends
        model_backends = await self._discover_available()
        if selected is not None:
            # Merge concurrent mapping so the selected model uses the
            # concurrent backend even if a serial backend has higher priority
            model_backends.update(concurrent_map)  # type: ignore[possibly-undefined]
        all_ids = [mid for mid in model_backends if mid not in excluded]

        if selected is None:
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

        # Estimate memory and evict if necessary
        backend = model_backends[selected]
        try:
            needed_mb = await backend.estimate_memory(selected)
        except Exception:
            needed_mb = 0
        if needed_mb > 0 and self._memory_budget_mb > 0:
            await self._ensure_memory(needed_mb)

        # Load model via backend
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
            total_memory_mb=self._memory_budget_mb,
            used_memory_mb=used,
        )

    # -------------------------------------------------------------------
    # Eviction
    # -------------------------------------------------------------------

    @property
    def _used_memory_mb(self) -> int:
        """Total memory consumed by all loaded models."""
        return sum(s.memory_mb for s in self._slots.values())

    async def _ensure_memory(self, needed_mb: int) -> None:
        """Evict LRU models with ``refcount == 0`` until enough memory is free.

        Raises ``RuntimeError`` if there is not enough reclaimable memory
        (all models are in use).
        """
        free = self._memory_budget_mb - self._used_memory_mb
        if free >= needed_mb:
            return

        # Sort evictable models by last_used (oldest first = LRU)
        evictable = sorted(
            [s for s in self._slots.values() if s.refcount == 0],
            key=lambda s: s.last_used,
        )

        for slot in evictable:
            if free >= needed_mb:
                break
            backend = self._registry.get_backend(slot.backend)
            if backend is not None:
                try:
                    await backend.unload_model(slot.model_id)
                except Exception:
                    logger.warning(
                        "Failed to unload model %s from %s",
                        slot.model_id, slot.backend, exc_info=True,
                    )
            free += slot.memory_mb
            del self._slots[slot.model_id]
            logger.info(
                "Evicted model %s (%d MB) — free: %d MB",
                slot.model_id, slot.memory_mb, free,
            )

        if free < needed_mb:
            in_use = [
                f"{s.model_id} (refcount={s.refcount}, {s.memory_mb} MB)"
                for s in self._slots.values()
                if s.refcount > 0
            ]
            raise RuntimeError(
                f"Insufficient memory: need {needed_mb} MB but only {free} MB "
                f"available after evicting all unused models. "
                f"In-use models: {in_use}"
            )

    # -------------------------------------------------------------------
    # Preloading
    # -------------------------------------------------------------------

    async def preload_hint(
        self,
        requirements: Dict[str, float],
        *,
        require_tool_calling: bool = True,
    ) -> None:
        """Hint that a model matching *requirements* will be needed soon.

        Non-blocking background loading.  If a matching model is already
        loaded, this is a no-op.  If memory is available, the model is
        loaded in the background.  If memory is insufficient, the hint
        is silently ignored (will evict when ``acquire()`` is called).
        """
        async with self._lock:
            await self._preload_locked(requirements, require_tool_calling)

    async def _preload_locked(
        self,
        requirements: Dict[str, float],
        require_tool_calling: bool,
    ) -> None:
        # Check if a matching model is already loaded
        loaded_ids = list(self._slots.keys())
        selected = self._select_model(
            loaded_ids, requirements, require_tool_calling, None,
        )
        if selected is not None:
            logger.debug("preload_hint: model %s already loaded", selected)
            return

        # Discover available models
        model_backends = await self._discover_available()
        all_ids = list(model_backends.keys())
        selected = self._select_model(
            all_ids, requirements, require_tool_calling, None,
        )
        if selected is None:
            logger.debug("preload_hint: no model matches requirements %s", requirements)
            return

        if selected in self._slots:
            return  # Already loaded (found via a different capability match)

        # Check if we have memory
        backend = model_backends[selected]
        try:
            needed_mb = await backend.estimate_memory(selected)
        except Exception:
            needed_mb = 0

        if needed_mb > 0 and self._memory_budget_mb > 0:
            free = self._memory_budget_mb - self._used_memory_mb
            if free < needed_mb:
                logger.debug(
                    "preload_hint: insufficient memory for %s "
                    "(%d MB needed, %d MB free); skipping",
                    selected, needed_mb, free,
                )
                return

        # Load in background (non-blocking from caller's perspective)
        try:
            slot = await backend.load_model(selected)
            slot.refcount = 0  # Not acquired yet
            self._slots[selected] = slot
            logger.info("preload_hint: preloaded %s (%d MB)", selected, slot.memory_mb)
        except Exception:
            logger.debug("preload_hint: failed to preload %s", selected, exc_info=True)

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

    def _is_on_concurrent_backend(self, model_id: str) -> bool:
        """Check if a loaded model's backend supports concurrent requests."""
        slot = self._slots.get(model_id)
        if slot is None:
            return False
        backend = self._registry.get_backend(slot.backend)
        return bool(backend and getattr(backend, "supports_concurrent", False))

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

    async def _discover_available(
        self,
        *,
        concurrent_only: bool = False,
    ) -> Dict[str, Any]:
        """Return ``model_id → backend`` for all available models.

        Queries all healthy backends in priority order.  First backend wins
        for models available on multiple backends.

        When *concurrent_only* is True, only backends whose
        ``supports_concurrent`` property is True are queried.
        """
        result: Dict[str, Any] = {}
        for backend in self._registry.healthy_backends():
            if concurrent_only and not getattr(backend, "supports_concurrent", False):
                continue
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
