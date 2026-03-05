"""CLI approval channel: prompts via stdin.

Uses ``asyncio.to_thread`` to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging

from agentic_concierge.application.approval import ApprovalDecision, ApprovalRequest

logger = logging.getLogger(__name__)


def _prompt_user(request: ApprovalRequest) -> ApprovalDecision:
    """Blocking stdin prompt — runs in a thread via ``asyncio.to_thread``."""
    print(
        f"\n{'=' * 60}\n"
        f"APPROVAL REQUESTED\n"
        f"  Action:  {request.action}\n"
        f"  Reason:  {request.reason}\n"
        f"  Command: {request.command}\n"
        f"{'=' * 60}"
    )
    while True:
        answer = input("Approve? [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            comment = input("Comment (optional): ").strip()
            return ApprovalDecision(approved=True, comment=comment)
        if answer in ("n", "no"):
            comment = input("Reason for denial (optional): ").strip()
            return ApprovalDecision(approved=False, comment=comment)
        print("Please enter 'y' or 'n'.")


class CliApprovalChannel:
    """Interactive approval via terminal stdin."""

    def __init__(self, timeout_s: float = 600.0):
        self._timeout_s = timeout_s

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_prompt_user, request),
                timeout=self._timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("Approval timed out after %.0fs; denying", self._timeout_s)
            return ApprovalDecision(
                approved=False,
                comment=f"Timed out after {self._timeout_s}s (fail-closed)",
            )
