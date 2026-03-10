"""SimpleModelRuntime — passthrough runtime wrapping a single ChatClient.

When no real ``LocalModelRuntime`` is available (e.g. tests, or when the
user hasn't configured ``backends.yaml``), this runtime wraps an existing
``ChatClient`` and model name.  Every ``acquire()`` returns the same
client regardless of requirements; ``release()`` is a no-op.

This allows the V2 graph execution path (``execute_graph_with_affinity``)
to work with the existing V1 infrastructure (``resolve_llm`` +
``build_chat_client``) without requiring a full backend registry.

See GH issue #28 for context on the V2 integration wiring.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agentic_concierge.application.ports import ChatClient
from agentic_concierge.infrastructure.backends.protocol import (
    ModelHandle,
    ModelInfo,
    ModelSlot,
    RuntimeStatus,
)

logger = logging.getLogger(__name__)


class SimpleModelRuntime:
    """Passthrough ModelRuntime that always returns the same ChatClient.

    Implements the ``ModelRuntime`` protocol from ``application/ports.py``.
    All capability requirements are ignored — the wrapped client is
    always returned.  This is suitable for single-model setups and
    testing scenarios.

    Usage::

        runtime = SimpleModelRuntime(chat_client, "qwen2.5:7b")
        handle = await runtime.acquire({"reasoning": 0.7})
        # handle.chat_client is the original chat_client
        # handle.model_id is "qwen2.5:7b"
    """

    def __init__(
        self,
        chat_client: ChatClient,
        model_id: str,
        *,
        available_models: Optional[List[str]] = None,
        backend: str = "ollama",
    ) -> None:
        self._client = chat_client
        self._model_id = model_id
        self._available = available_models or [model_id]
        self._slot = ModelSlot(
            model_id=model_id,
            backend=backend,
            memory_mb=0,
        )

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
        """Always returns the wrapped ChatClient, ignoring requirements."""
        excluded = set(exclude_models or [])
        if self._model_id in excluded:
            # If the only model is excluded (e.g. must_differ_from),
            # still return it (fail-open, like V1 reviewer behavior).
            logger.debug(
                "SimpleModelRuntime: model %s is excluded but no alternative; "
                "returning anyway (fail-open)",
                self._model_id,
            )
        return ModelHandle(slot=self._slot, chat_client=self._client, runtime=self)

    async def release(self, handle: Any) -> None:
        """No-op — single client is always available."""
        pass

    async def inventory(self) -> List[ModelInfo]:
        """Return the single wrapped model."""
        return [
            ModelInfo(
                model_id=self._model_id,
                backend="passthrough",
                available=True,
                loaded=True,
                capabilities={},
                estimated_memory_mb=0,
                supports_tool_calling=True,
            )
        ]

    async def status(self) -> RuntimeStatus:
        """Return a minimal status."""
        return RuntimeStatus(
            loaded_models=[self._slot],
            total_memory_mb=0,
            used_memory_mb=0,
        )

    async def preload_hint(
        self,
        requirements: Dict[str, float],
        *,
        require_tool_calling: bool = True,
    ) -> None:
        """No-op — model is always loaded."""
        pass
