"""Shell tool: run allowlisted commands in sandbox."""

from __future__ import annotations

import shlex
from typing import List

from .sandbox import SandboxPolicy, run_cmd


def run_shell(policy: SandboxPolicy, cmd: List[str], timeout_s: int = 120) -> dict:
    # LLMs occasionally pass timeout_s as a string despite the schema type hint.
    # LLMs also sometimes pass cmd as a string instead of a list (e.g.
    # "python -m pytest" instead of ["python", "-m", "pytest"]).
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    return run_cmd(policy, cmd, timeout_s=int(timeout_s))
