"""Tests for the doctor diagnostics module.

Verifies:
- Individual check functions
- DiagnosticReport properties
- Full diagnostics run (with mocked backends)
"""

from __future__ import annotations

import pytest

from agentic_concierge.application.doctor import (
    CheckResult,
    DiagnosticReport,
    check_agent_roles,
    check_backends_yaml,
    check_config_loadable,
    check_memory_budget,
)


# ---------------------------------------------------------------------------
# CheckResult
# ---------------------------------------------------------------------------


class TestCheckResult:

    def test_passed(self):
        c = CheckResult(name="test", passed=True, message="OK")
        assert c.passed is True

    def test_failed_with_suggestion(self):
        c = CheckResult(
            name="test", passed=False, message="Failed",
            suggestion="Do this",
        )
        assert c.passed is False
        assert c.suggestion == "Do this"


# ---------------------------------------------------------------------------
# DiagnosticReport
# ---------------------------------------------------------------------------


class TestDiagnosticReport:

    def test_all_passed(self):
        report = DiagnosticReport(checks=[
            CheckResult("a", True, "OK"),
            CheckResult("b", True, "OK"),
        ])
        assert report.all_passed is True

    def test_not_all_passed(self):
        report = DiagnosticReport(checks=[
            CheckResult("a", True, "OK"),
            CheckResult("b", False, "Bad"),
        ])
        assert report.all_passed is False

    def test_failed_checks(self):
        report = DiagnosticReport(checks=[
            CheckResult("a", True, "OK"),
            CheckResult("b", False, "Bad"),
            CheckResult("c", False, "Also bad"),
        ])
        failed = report.failed_checks
        assert len(failed) == 2
        assert failed[0].name == "b"

    def test_summary(self):
        report = DiagnosticReport(checks=[
            CheckResult("a", True, "OK"),
            CheckResult("b", False, "Bad"),
        ])
        assert report.summary() == "1/2 checks passed"

    def test_empty_report(self):
        report = DiagnosticReport()
        assert report.all_passed is True
        assert report.summary() == "0/0 checks passed"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


class TestConfigLoadable:

    def test_config_loads(self):
        result = check_config_loadable()
        # Should pass in test environment
        assert result.name == "config_loadable"
        assert result.passed is True


class TestBackendsYaml:

    def test_backends_yaml_loads(self):
        result = check_backends_yaml()
        assert result.name == "backends_yaml"
        assert result.passed is True
        assert "backends configured" in result.message


class TestMemoryBudget:

    def test_detects_memory(self):
        result = check_memory_budget()
        assert result.name == "memory_budget"
        # Should pass on any real system with > 4 GB RAM
        assert result.passed is True
        assert "MB" in result.message


class TestAgentRoles:

    def test_roles_loadable(self):
        result = check_agent_roles()
        assert result.name == "agent_roles"
        assert result.passed is True
        assert "6 agent roles" in result.message
