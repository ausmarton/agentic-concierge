"""Tests for engineering pack quality gate.

Covers:
- tests_verified is in finish_required_fields.
- validate_finish_payload rejects tests_verified=False.
- validate_finish_payload accepts tests_verified=True.
- run_tests tool is in the engineering pack's tool list.
"""
from __future__ import annotations

import tempfile

from agentic_concierge.infrastructure.specialists.dynamic_pack import build_template_pack


def test_tests_verified_in_required_fields():
    """tests_verified must be in the engineering pack's finish_required_fields."""
    with tempfile.TemporaryDirectory() as d:
        pack = build_template_pack("engineering", d, network_allowed=False)
        assert "tests_verified" in pack.finish_required_fields


def test_validate_finish_payload_rejects_tests_verified_false():
    """Quality gate returns an error string when tests_verified=False."""
    with tempfile.TemporaryDirectory() as d:
        pack = build_template_pack("engineering", d, network_allowed=False)
        error = pack.validate_finish_payload({
            "summary": "all done",
            "artifacts": [],
            "next_steps": [],
            "notes": "",
            "tests_verified": False,
        })
        assert isinstance(error, str)
        assert len(error) > 0
        assert "tests_verified" in error.lower() or "run_tests" in error.lower()


def test_validate_finish_payload_accepts_tests_verified_true():
    """Quality gate returns None (no error) when tests_verified=True."""
    with tempfile.TemporaryDirectory() as d:
        pack = build_template_pack("engineering", d, network_allowed=False)
        error = pack.validate_finish_payload({
            "summary": "all done",
            "artifacts": [],
            "next_steps": [],
            "notes": "",
            "tests_verified": True,
        })
        assert error is None


def test_validate_finish_payload_passes_when_tests_verified_missing():
    """Quality gate does not fire when tests_verified is absent (missing field is caught by gate 2)."""
    with tempfile.TemporaryDirectory() as d:
        pack = build_template_pack("engineering", d, network_allowed=False)
        # Missing tests_verified key — gate 2 (required fields) handles this, not gate 3.
        error = pack.validate_finish_payload({"summary": "done"})
        assert error is None


def test_run_tests_tool_in_engineering_pack():
    """run_tests must appear in the engineering pack's tool list."""
    with tempfile.TemporaryDirectory() as d:
        pack = build_template_pack("engineering", d, network_allowed=False)
        assert "run_tests" in pack.tool_names
