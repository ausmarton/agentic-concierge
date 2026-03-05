"""Auto-approval channel: always approves.

Used for tests, ``--auto-approve`` CLI flag, and non-streaming HTTP runs.
"""

from __future__ import annotations

from agentic_concierge.application.approval import ApprovalDecision, ApprovalRequest


class AutoApprovalChannel:
    """Always approves immediately."""

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(approved=True, comment="auto-approved")
