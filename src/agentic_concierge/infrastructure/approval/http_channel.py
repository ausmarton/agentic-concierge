"""HTTP approval channel: emits SSE event, awaits REST callback.

The SSE streaming endpoint injects approval requests as events. A separate
``POST /runs/{run_id}/approve`` endpoint resolves the pending request.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict

from agentic_concierge.application.approval import ApprovalDecision, ApprovalRequest

logger = logging.getLogger(__name__)

# TTL for cleaning up abandoned channels (1 hour).
_CHANNEL_TTL_S = 3600.0


class HttpApprovalChannel:
    """Approval channel backed by asyncio.Event for HTTP streaming runs.

    The loop calls ``request_approval()`` which blocks on an ``asyncio.Event``.
    The HTTP endpoint calls ``resolve()`` to unblock the waiter.
    """

    def __init__(self, timeout_s: float = 600.0):
        self._timeout_s = timeout_s
        self._pending: Dict[str, asyncio.Event] = {}
        self._decisions: Dict[str, ApprovalDecision] = {}
        self._created_at = time.monotonic()

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        event = asyncio.Event()
        self._pending[request.request_id] = event
        try:
            await asyncio.wait_for(event.wait(), timeout=self._timeout_s)
        except asyncio.TimeoutError:
            logger.warning(
                "HTTP approval timed out for %s after %.0fs; denying",
                request.request_id, self._timeout_s,
            )
            return ApprovalDecision(
                approved=False,
                comment=f"Timed out after {self._timeout_s}s (fail-closed)",
            )
        finally:
            self._pending.pop(request.request_id, None)
        return self._decisions.pop(request.request_id, ApprovalDecision(approved=False))

    def resolve(self, request_id: str, decision: ApprovalDecision) -> bool:
        """Resolve a pending approval request. Returns True if found."""
        event = self._pending.get(request_id)
        if event is None:
            return False
        self._decisions[request_id] = decision
        event.set()
        return True

    @property
    def is_stale(self) -> bool:
        """True if this channel was created more than TTL ago."""
        return (time.monotonic() - self._created_at) > _CHANNEL_TTL_S
