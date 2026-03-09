"""Tests for sandbox path safety and command allowlist."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from agentic_concierge.infrastructure.tools.sandbox import SandboxPolicy, run_cmd, safe_path


def test_safe_path_within_root():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "a").mkdir()
        policy = SandboxPolicy(root=root)
        p = safe_path(policy, "a/file.txt")
        assert p == root / "a" / "file.txt"


@pytest.mark.parametrize("escape_path", [
    "../etc/passwd",
    "..",
    "a/../../etc/passwd",
])
def test_safe_path_escape_fails(tmp_path, escape_path):
    policy = SandboxPolicy(root=tmp_path)
    with pytest.raises(PermissionError):
        safe_path(policy, escape_path)


def test_safe_path_rejects_absolute_path(tmp_path):
    """safe_path must reject absolute paths with a PermissionError."""
    policy = SandboxPolicy(root=tmp_path)
    with pytest.raises(PermissionError, match="relative"):
        safe_path(policy, "/etc/passwd")


def test_run_cmd_rejects_disallowed_command(tmp_path):
    """run_cmd must raise PermissionError for commands not in the allowlist."""
    policy = SandboxPolicy(root=tmp_path)
    with pytest.raises(PermissionError, match="not allowed"):
        run_cmd(policy, ["rm", "-rf", "/"])


def test_run_cmd_allows_listed_command(tmp_path):
    """python3 --version is in the allowlist and should succeed."""
    policy = SandboxPolicy(root=tmp_path)
    result = run_cmd(policy, ["python3", "--version"])
    assert result["returncode"] == 0
    assert "Python" in result["stdout"] or "Python" in result["stderr"]


# ---------------------------------------------------------------------------
# Issue #6: LLM sends cmd as string instead of list
# ---------------------------------------------------------------------------


def test_run_shell_coerces_string_cmd_to_list(tmp_path):
    """LLMs sometimes send cmd as a string; run_shell should split it."""
    from agentic_concierge.infrastructure.tools.shell_tools import run_shell
    policy = SandboxPolicy(root=tmp_path)
    result = run_shell(policy, "python3 --version")
    assert result["returncode"] == 0
    assert "Python" in result["stdout"] or "Python" in result["stderr"]


def test_run_shell_string_cmd_with_args(tmp_path):
    """String cmd with arguments should be split correctly."""
    from agentic_concierge.infrastructure.tools.shell_tools import run_shell
    policy = SandboxPolicy(root=tmp_path)
    result = run_shell(policy, "python3 -c 'print(42)'")
    assert result["returncode"] == 0
    assert "42" in result["stdout"]


def test_run_shell_list_cmd_still_works(tmp_path):
    """List cmd (correct usage) should still work unchanged."""
    from agentic_concierge.infrastructure.tools.shell_tools import run_shell
    policy = SandboxPolicy(root=tmp_path)
    result = run_shell(policy, ["python3", "--version"])
    assert result["returncode"] == 0


def test_run_shell_string_cmd_blocked_command(tmp_path):
    """String cmd with disallowed command should still be blocked."""
    from agentic_concierge.infrastructure.tools.shell_tools import run_shell
    policy = SandboxPolicy(root=tmp_path)
    with pytest.raises(PermissionError, match="not allowed"):
        run_shell(policy, "rm -rf /")
