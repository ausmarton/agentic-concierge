"""Backend registry — discovers, monitors, and provides failover for backends.

The registry probes all configured backends at startup, tracks their health,
and provides the highest-priority healthy backend on demand.  If a backend
goes down, subsequent ``primary()`` calls automatically fail over to the
next healthy backend.

Usage::

    registry = BackendRegistry.from_config()
    await registry.discover()
    backend = registry.primary()           # highest-priority healthy backend
    client = backend.build_client("qwen2.5:7b")

See DESIGN_V2.md §11.5 and §11.10 for rationale.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agentic_concierge.infrastructure.backends.protocol import InferenceBackend

logger = logging.getLogger(__name__)


@dataclass
class BackendEntry:
    """A registered backend with health tracking."""

    backend: InferenceBackend
    priority: int = 0                  # lower = higher priority
    healthy: bool = False
    last_check: float = 0.0           # time.monotonic() of last health check
    consecutive_failures: int = 0
    last_error: str = ""


class BackendRegistry:
    """Registry of inference backends with discovery, health monitoring, and failover.

    Backends are ordered by priority (lower number = tried first).
    Health checks mark backends as healthy/unhealthy; ``primary()`` returns
    the first healthy backend in priority order.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, BackendEntry] = {}

    # -------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------

    def register(self, backend: InferenceBackend, *, priority: int = 0) -> None:
        """Register a backend with a given priority (lower = higher priority)."""
        self._entries[backend.name] = BackendEntry(
            backend=backend,
            priority=priority,
        )

    def unregister(self, name: str) -> None:
        """Remove a backend from the registry."""
        self._entries.pop(name, None)

    # -------------------------------------------------------------------
    # Discovery
    # -------------------------------------------------------------------

    async def discover(self) -> List[str]:
        """Probe all registered backends concurrently; return names of healthy ones.

        Updates the ``healthy`` flag on each entry based on the health check result.
        """
        if not self._entries:
            return []

        entries = list(self._entries.values())
        results = await asyncio.gather(
            *[self._check_entry(e) for e in entries],
            return_exceptions=True,
        )

        healthy_names = []
        for entry, result in zip(entries, results):
            if isinstance(result, Exception):
                entry.healthy = False
                entry.consecutive_failures += 1
                entry.last_error = str(result)
                logger.warning("Backend %s health check failed: %s", entry.backend.name, result)
            elif result:
                entry.healthy = True
                entry.consecutive_failures = 0
                entry.last_error = ""
                healthy_names.append(entry.backend.name)
            else:
                entry.healthy = False
                entry.consecutive_failures += 1
            entry.last_check = time.monotonic()

        return healthy_names

    async def _check_entry(self, entry: BackendEntry) -> bool:
        """Run health check on a single backend entry."""
        try:
            return await entry.backend.health_check()
        except Exception as e:
            entry.last_error = str(e)
            return False

    # -------------------------------------------------------------------
    # Backend access
    # -------------------------------------------------------------------

    def get_backend(self, name: str) -> Optional[InferenceBackend]:
        """Get a specific backend by name, or ``None`` if not registered."""
        entry = self._entries.get(name)
        return entry.backend if entry else None

    def primary(self) -> InferenceBackend:
        """Return the highest-priority healthy backend.

        Raises ``RuntimeError`` if no healthy backend is available.
        """
        healthy = self._sorted_healthy()
        if not healthy:
            names = list(self._entries.keys())
            raise RuntimeError(
                f"No healthy inference backend available. "
                f"Registered backends: {names}"
            )
        return healthy[0].backend

    def primary_or_none(self) -> Optional[InferenceBackend]:
        """Return the highest-priority healthy backend, or ``None``."""
        healthy = self._sorted_healthy()
        return healthy[0].backend if healthy else None

    def healthy_backends(self) -> List[InferenceBackend]:
        """Return all healthy backends in priority order."""
        return [e.backend for e in self._sorted_healthy()]

    def _sorted_healthy(self) -> List[BackendEntry]:
        """Healthy entries sorted by priority (lowest first)."""
        return sorted(
            [e for e in self._entries.values() if e.healthy],
            key=lambda e: e.priority,
        )

    # -------------------------------------------------------------------
    # Health monitoring
    # -------------------------------------------------------------------

    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all registered backends.

        Returns a dict mapping backend name → health status.
        Also updates internal state for ``primary()`` and ``healthy_backends()``.
        """
        await self.discover()
        return {name: entry.healthy for name, entry in self._entries.items()}

    def mark_unhealthy(self, name: str, error: str = "") -> None:
        """Mark a backend as unhealthy (e.g. after a request failure).

        Call this when a backend fails during use — the registry will
        fail over to the next healthy backend on the next ``primary()`` call.
        """
        entry = self._entries.get(name)
        if entry:
            entry.healthy = False
            entry.consecutive_failures += 1
            entry.last_error = error
            logger.warning("Backend %s marked unhealthy: %s", name, error)

    def mark_healthy(self, name: str) -> None:
        """Mark a backend as healthy (e.g. after a successful recovery)."""
        entry = self._entries.get(name)
        if entry:
            entry.healthy = True
            entry.consecutive_failures = 0
            entry.last_error = ""

    # -------------------------------------------------------------------
    # Inspection
    # -------------------------------------------------------------------

    def list_backends(self) -> List[str]:
        """Return names of all registered backends."""
        return list(self._entries.keys())

    def status(self) -> Dict[str, Dict[str, Any]]:
        """Return a status summary for all registered backends."""
        return {
            name: {
                "healthy": entry.healthy,
                "priority": entry.priority,
                "consecutive_failures": entry.consecutive_failures,
                "last_error": entry.last_error,
                "base_url": entry.backend.base_url,
            }
            for name, entry in self._entries.items()
        }

    # -------------------------------------------------------------------
    # Factory
    # -------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        *,
        tier: Optional[str] = None,
    ) -> "BackendRegistry":
        """Create a registry with backends configured from ``backends.yaml``.

        Registers an ``OllamaBackend`` and optionally a ``LlamaCppBackend``
        (when ``model_dir`` is configured).  Backend priority is determined
        by the tier priority list from ``backends.yaml``.

        Args:
            tier: Profile tier name (e.g. "medium").  If ``None``, defaults
                  to "medium" priority ordering.
        """
        from agentic_concierge.config.constants import default_backend_urls
        from agentic_concierge.config.features import ProfileTier, backend_priority
        from agentic_concierge.infrastructure.backends.ollama import OllamaBackend

        urls = default_backend_urls()
        bp = backend_priority()

        # Determine priority order
        tier_enum = ProfileTier.MEDIUM
        if tier:
            try:
                tier_enum = ProfileTier(tier.lower())
            except ValueError:
                pass
        priority_list = bp.get(tier_enum, ["ollama", "llama_cpp", "vllm", "cloud"])

        registry = cls()

        # Register Ollama backend
        ollama_url = urls.get("ollama", "http://localhost:11434/v1")
        ollama_priority = priority_list.index("ollama") if "ollama" in priority_list else 99
        registry.register(OllamaBackend(base_url=ollama_url), priority=ollama_priority)

        # Register LlamaCppBackend if model_dir is configured
        try:
            from agentic_concierge.config.external import load_yaml_config
            backends_cfg = load_yaml_config("backends.yaml")
            llama_cfg = backends_cfg.get("backends", {}).get("llama_cpp", {})
            model_dir = llama_cfg.get("model_dir", "")
            if model_dir:
                from agentic_concierge.infrastructure.backends.llama_cpp import LlamaCppBackend
                llama_priority = (
                    priority_list.index("llama_cpp")
                    if "llama_cpp" in priority_list
                    else len(priority_list)
                )
                registry.register(
                    LlamaCppBackend(
                        model_dir=model_dir,
                        binary=llama_cfg.get("binary", "llama-server"),
                        n_gpu_layers=int(llama_cfg.get("n_gpu_layers", -1)),
                        ctx_size=int(llama_cfg.get("ctx_size", 8192)),
                    ),
                    priority=llama_priority,
                )
        except Exception:
            logger.debug("LlamaCppBackend not registered", exc_info=True)

        return registry
