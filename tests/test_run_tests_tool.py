"""Tests for infrastructure/tools/test_runner.py.

Covers:
- _detect_framework: Cargo.toml, package.json+test, pytest.ini, pyproject.toml,
  setup.cfg, test_*.py glob, fallback to pytest.
- _parse_pytest_output: passed/failed/error counts, summary string.
- _parse_cargo_output: ok/FAILED status, counts.
- run_tests: framework detection, command building, timeout, sandbox passthrough.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentic_concierge.infrastructure.tools.test_runner import (
    _detect_framework,
    _parse_cargo_output,
    _parse_pytest_output,
    _parse_unittest_output,
    run_tests,
)
from agentic_concierge.infrastructure.tools.sandbox import SandboxPolicy


# ---------------------------------------------------------------------------
# _detect_framework tests
# ---------------------------------------------------------------------------

def test_detect_cargo(tmp_path):
    (tmp_path / "Cargo.toml").write_text("[package]\nname = \"mylib\"\n")
    assert _detect_framework(tmp_path) == "cargo"


def test_detect_npm_with_test_script(tmp_path):
    pkg = {"name": "myapp", "scripts": {"test": "jest"}}
    (tmp_path / "package.json").write_text(json.dumps(pkg))
    assert _detect_framework(tmp_path) == "npm"


def test_detect_npm_without_test_script_falls_through(tmp_path):
    pkg = {"name": "myapp", "scripts": {"build": "webpack"}}
    (tmp_path / "package.json").write_text(json.dumps(pkg))
    # No test script → should fall through to pytest heuristics or default
    result = _detect_framework(tmp_path)
    assert result == "pytest"  # default


def test_detect_pytest_ini(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    assert _detect_framework(tmp_path) == "pytest"


def test_detect_pyproject_with_pytest_section(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    assert _detect_framework(tmp_path) == "pytest"


def test_detect_setup_cfg_with_pytest(tmp_path):
    (tmp_path / "setup.cfg").write_text("[tool:pytest]\n")
    assert _detect_framework(tmp_path) == "pytest"


def test_detect_test_file_glob(tmp_path):
    (tmp_path / "test_mymodule.py").write_text("# test\n")
    assert _detect_framework(tmp_path) == "pytest"


def test_detect_fallback_is_pytest(tmp_path):
    # No markers → default to pytest
    assert _detect_framework(tmp_path) == "pytest"


# ---------------------------------------------------------------------------
# _parse_pytest_output tests
# ---------------------------------------------------------------------------

def test_parse_pytest_all_passed():
    out = "5 passed in 1.2s"
    passed, failed, errors, summary = _parse_pytest_output(out)
    assert passed is True
    assert failed == 0
    assert errors == 0
    assert "5 passed" in summary


def test_parse_pytest_with_failures():
    out = "3 passed, 2 failed in 0.8s"
    passed, failed, errors, summary = _parse_pytest_output(out)
    assert passed is False
    assert failed == 2
    assert "2 failed" in summary


def test_parse_pytest_with_errors():
    out = "1 passed, 1 error in 0.3s"
    passed, failed, errors, summary = _parse_pytest_output(out)
    assert passed is False
    assert errors == 1
    assert "1 error" in summary


def test_parse_pytest_no_results():
    out = "no tests ran"
    passed, failed, errors, summary = _parse_pytest_output(out)
    assert passed is True  # no failures
    assert "no test results detected" in summary


# ---------------------------------------------------------------------------
# _parse_cargo_output tests
# ---------------------------------------------------------------------------

def test_parse_cargo_ok():
    out = "test result: ok. 10 passed; 0 failed; 0 ignored;"
    passed, failed, errors, summary = _parse_cargo_output(out)
    assert passed is True
    assert failed == 0
    assert "10 passed" in summary


def test_parse_cargo_failed():
    out = "test result: FAILED. 8 passed; 2 failed; 0 ignored;"
    passed, failed, errors, summary = _parse_cargo_output(out)
    assert passed is False
    assert failed == 2
    assert "2 failed" in summary


def test_parse_cargo_no_match():
    out = "compilation error"
    passed, failed, errors, summary = _parse_cargo_output(out)
    assert passed is False
    assert "no test results detected" in summary


# ---------------------------------------------------------------------------
# run_tests tests
# ---------------------------------------------------------------------------

def _make_policy(root: Path) -> SandboxPolicy:
    return SandboxPolicy(root=root)


def test_run_tests_auto_detects_and_runs_pytest(tmp_path):
    """run_tests with framework='auto' on a pytest project runs pytest."""
    (tmp_path / "pytest.ini").write_text("[pytest]\n")

    mock_result = {"stdout": "2 passed in 0.1s", "stderr": "", "returncode": 0}
    with patch(
        "agentic_concierge.infrastructure.tools.test_runner.run_cmd",
        return_value=mock_result,
    ) as mock_run_cmd:
        policy = _make_policy(tmp_path)
        result = run_tests(policy, framework="auto")

    mock_run_cmd.assert_called_once()
    cmd_arg = mock_run_cmd.call_args[0][1]  # second positional arg is the command list
    assert cmd_arg[:3] == ["python", "-m", "pytest"]
    assert result["passed"] is True
    assert result["framework"] == "pytest"


def test_run_tests_explicit_cargo(tmp_path):
    """run_tests with framework='cargo' runs cargo test."""
    mock_result = {
        "stdout": "test result: ok. 5 passed; 0 failed; 0 ignored;",
        "stderr": "",
        "returncode": 0,
    }
    with patch(
        "agentic_concierge.infrastructure.tools.test_runner.run_cmd",
        return_value=mock_result,
    ) as mock_run_cmd:
        policy = _make_policy(tmp_path)
        result = run_tests(policy, framework="cargo")

    cmd_arg = mock_run_cmd.call_args[0][1]
    assert cmd_arg[0] == "cargo"
    assert result["passed"] is True
    assert result["framework"] == "cargo"


def test_run_tests_explicit_npm(tmp_path):
    """run_tests with framework='npm' runs npm test; trusts exit code."""
    mock_result = {"stdout": "All tests pass!", "stderr": "", "returncode": 0}
    with patch(
        "agentic_concierge.infrastructure.tools.test_runner.run_cmd",
        return_value=mock_result,
    ) as mock_run_cmd:
        policy = _make_policy(tmp_path)
        result = run_tests(policy, framework="npm")

    cmd_arg = mock_run_cmd.call_args[0][1]
    assert cmd_arg[0] == "npm"
    assert result["passed"] is True
    assert result["framework"] == "npm"


def test_run_tests_timeout_returns_error_dict(tmp_path):
    """When the test command times out, run_tests returns an error dict."""
    with patch(
        "agentic_concierge.infrastructure.tools.test_runner.run_cmd",
        side_effect=subprocess.TimeoutExpired(cmd="pytest", timeout=5),
    ):
        policy = _make_policy(tmp_path)
        result = run_tests(policy, framework="pytest", timeout_s=5)

    assert result["passed"] is False
    assert "timed out" in result["summary"]
    assert result["framework"] == "pytest"


def test_run_tests_failed_exit_code_sets_passed_false(tmp_path):
    """A non-zero exit code with no parsed failures still marks passed=False."""
    mock_result = {"stdout": "import error", "stderr": "", "returncode": 1}
    with patch(
        "agentic_concierge.infrastructure.tools.test_runner.run_cmd",
        return_value=mock_result,
    ):
        policy = _make_policy(tmp_path)
        result = run_tests(policy, framework="pytest")

    assert result["passed"] is False


def test_run_tests_output_truncated(tmp_path):
    """Output longer than 3000 chars is truncated."""
    long_output = "x" * 5000
    mock_result = {"stdout": long_output, "stderr": "", "returncode": 0}
    with patch(
        "agentic_concierge.infrastructure.tools.test_runner.run_cmd",
        return_value=mock_result,
    ):
        policy = _make_policy(tmp_path)
        result = run_tests(policy, framework="pytest")

    assert len(result["output"]) <= 3000


# ---------------------------------------------------------------------------
# _parse_unittest_output tests
# ---------------------------------------------------------------------------

def test_parse_unittest_all_passed():
    output = "....\n----------------------------------------------------------------------\nRan 4 tests in 0.001s\n\nOK"
    passed, failed, errors, summary = _parse_unittest_output(output)
    assert passed is True
    assert failed == 0
    assert errors == 0
    assert "4 ran" in summary


def test_parse_unittest_with_failures():
    output = "...F\n======\nFAIL: test_foo\n------\nRan 4 tests in 0.001s\n\nFAILED (failures=1)"
    passed, failed, errors, summary = _parse_unittest_output(output)
    assert passed is False
    assert failed == 1
    assert "failed" in summary


def test_parse_unittest_with_errors():
    output = "...E\nRan 4 tests in 0.001s\n\nFAILED (errors=1)"
    passed, failed, errors, summary = _parse_unittest_output(output)
    assert passed is False
    assert errors == 1


def test_parse_unittest_empty_output():
    passed, failed, errors, summary = _parse_unittest_output("")
    assert passed is False
    assert "no tests" in summary


# ---------------------------------------------------------------------------
# run_tests with framework='unittest'
# ---------------------------------------------------------------------------

def test_run_tests_explicit_unittest_uses_python_m_unittest(tmp_path):
    """run_tests with framework='unittest' runs python -m unittest discover."""
    mock_result = {
        "stdout": "....\n------\nRan 4 tests in 0.001s\n\nOK",
        "stderr": "",
        "returncode": 0,
    }
    with patch(
        "agentic_concierge.infrastructure.tools.test_runner.run_cmd",
        return_value=mock_result,
    ) as mock_run_cmd:
        policy = _make_policy(tmp_path)
        result = run_tests(policy, framework="unittest")

    cmd_arg = mock_run_cmd.call_args[0][1]
    assert cmd_arg[:4] == ["python", "-m", "unittest", "discover"]
    assert result["framework"] == "unittest"
    assert result["passed"] is True


def test_run_tests_pytest_uses_python_m_pytest(tmp_path):
    """run_tests with framework='pytest' runs python -m pytest, not bare pytest."""
    mock_result = {"stdout": "1 passed in 0.1s", "stderr": "", "returncode": 0}
    with patch(
        "agentic_concierge.infrastructure.tools.test_runner.run_cmd",
        return_value=mock_result,
    ) as mock_run_cmd:
        policy = _make_policy(tmp_path)
        run_tests(policy, framework="pytest")

    cmd_arg = mock_run_cmd.call_args[0][1]
    assert cmd_arg[0] == "python"
    assert cmd_arg[1] == "-m"
    assert cmd_arg[2] == "pytest"


def test_run_tests_unknown_framework_falls_back_to_pytest(tmp_path):
    """An unrecognised framework name silently falls back to pytest."""
    mock_result = {"stdout": "1 passed", "stderr": "", "returncode": 0}
    with patch(
        "agentic_concierge.infrastructure.tools.test_runner.run_cmd",
        return_value=mock_result,
    ) as mock_run_cmd:
        policy = _make_policy(tmp_path)
        result = run_tests(policy, framework="mocha")  # not a known framework

    cmd_arg = mock_run_cmd.call_args[0][1]
    assert cmd_arg[:3] == ["python", "-m", "pytest"]
    assert result["framework"] == "pytest"


# ---------------------------------------------------------------------------
# FR7.4: String-typed numeric coercion — LLMs send "120" not 120
# ---------------------------------------------------------------------------

def test_run_tests_string_timeout_coerced_to_int(tmp_path):
    """run_tests coerces timeout_s from str to int (FR7.4 defence)."""
    mock_result = {"stdout": "1 passed in 0.1s", "stderr": "", "returncode": 0}
    with patch(
        "agentic_concierge.infrastructure.tools.test_runner.run_cmd",
        return_value=mock_result,
    ) as mock_run_cmd:
        policy = _make_policy(tmp_path)
        # Pass timeout_s as a string — this is what small LLMs do
        result = run_tests(policy, framework="pytest", timeout_s="120")  # type: ignore[arg-type]

    assert result["passed"] is True
    # The value passed to run_cmd must be an int, not a string
    call_kwargs = mock_run_cmd.call_args
    assert isinstance(call_kwargs.kwargs.get("timeout_s", call_kwargs[1].get("timeout_s")), int)
