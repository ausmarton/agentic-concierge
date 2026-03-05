"""Approval channel protocol and data types for human-in-the-loop approval.

The ``ApprovalChannel`` protocol defines the interface that the pack tool loop
uses to pause and wait for human approval before a specialist performs a
dangerous action (deploy, push, etc.).  Concrete implementations live in
``infrastructure/approval/``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ApprovalRequest:
    """A request for human approval of a dangerous action."""

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    action: str = ""
    reason: str = ""
    command: str = ""
    specialist_id: str = ""


@dataclass
class ApprovalDecision:
    """Human decision on an approval request."""

    approved: bool = True
    comment: str = ""


class ApprovalChannel(Protocol):
    """Interface for requesting human approval during a specialist run."""

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision: ...
