"""Tests for concierge doctor and bootstrap CLI commands."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from agentic_concierge.interfaces.cli import app
from agentic_concierge.bootstrap.backend_manager import BackendHealth, BackendStatus
from agentic_concierge.bootstrap.model_advisor import SystemProfile
from agentic_concierge.bootstrap.system_probe import SystemProbe
from agentic_concierge.config.features import ProfileTier


runner = CliRunner()


def _sample_probe() -> SystemProbe:
    return SystemProbe(
        cpu_cores=8,
        cpu_arch="x86_64",
        ram_total_mb=16 * 1024,
        ram_available_mb=8 * 1024,
        internet_reachable=True,
        ollama_installed=True,
        ollama_reachable=True,
    )


def _sample_profile() -> SystemProfile:
    return SystemProfile(
        tier=ProfileTier.MEDIUM,
        routing_model="qwen2.5:0.5b",
        fast_model="qwen2.5:7b",
        quality_model="qwen2.5:14b",
        max_concurrent_agents=3,
        ram_total_mb=16384,
        ram_available_mb=8192,
        total_vram_mb=0,
        cpu_cores=8,
        cpu_arch="x86_64",
        gpu_count=0,
    )


def _healthy_backends() -> dict:
    return {
        "ollama": BackendHealth(
            name="ollama", status=BackendStatus.HEALTHY, models=["qwen2.5:7b"]
        ),
        "llama_cpp": BackendHealth(name="llama_cpp", status=BackendStatus.HEALTHY,
                                    hint="llama-server installed; models started on demand."),
        "vllm": BackendHealth(name="vllm", status=BackendStatus.DISABLED),
    }


# ---------------------------------------------------------------------------
# doctor command
# ---------------------------------------------------------------------------

def test_doctor_shows_profile(monkeypatch):
    with (
        patch("agentic_concierge.bootstrap.system_probe.probe_system", return_value=_sample_probe()),
        patch("agentic_concierge.bootstrap.detected.load_detected", return_value=_sample_profile()),
        patch(
            "agentic_concierge.bootstrap.backend_manager.BackendManager.probe_all",
            return_value=_healthy_backends(),
        ),
        patch("agentic_concierge.bootstrap.system_probe._check_internet", return_value=True),
        patch("agentic_concierge.bootstrap.system_probe._check_ollama", return_value=True),
        patch("agentic_concierge.bootstrap.system_probe._check_vllm", return_value=False),
        patch("subprocess.run", side_effect=FileNotFoundError),
    ):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "MEDIUM" in result.output or "medium" in result.output.lower()


def test_doctor_shows_backends(monkeypatch):
    with (
        patch("agentic_concierge.bootstrap.system_probe.probe_system", return_value=_sample_probe()),
        patch("agentic_concierge.bootstrap.detected.load_detected", return_value=_sample_profile()),
        patch(
            "agentic_concierge.bootstrap.backend_manager.BackendManager.probe_all",
            return_value=_healthy_backends(),
        ),
        patch("agentic_concierge.bootstrap.system_probe._check_internet", return_value=True),
        patch("agentic_concierge.bootstrap.system_probe._check_ollama", return_value=True),
        patch("agentic_concierge.bootstrap.system_probe._check_vllm", return_value=False),
        patch("subprocess.run", side_effect=FileNotFoundError),
    ):
        result = runner.invoke(app, ["doctor"])
    assert "ollama" in result.output


def test_doctor_unhealthy_backend_shows_hint(monkeypatch):
    backends = {
        "ollama": BackendHealth(name="ollama", status=BackendStatus.NOT_INSTALLED,
                                hint="Install from ollama.com"),
        "llama_cpp": BackendHealth(name="llama_cpp", status=BackendStatus.NOT_INSTALLED,
                                    hint="Install llama.cpp"),
        "vllm": BackendHealth(name="vllm", status=BackendStatus.DISABLED),
    }
    with (
        patch("agentic_concierge.bootstrap.system_probe.probe_system", return_value=_sample_probe()),
        patch("agentic_concierge.bootstrap.detected.load_detected", return_value=_sample_profile()),
        patch(
            "agentic_concierge.bootstrap.backend_manager.BackendManager.probe_all",
            return_value=backends,
        ),
        patch("agentic_concierge.bootstrap.system_probe._check_internet", return_value=False),
        patch("agentic_concierge.bootstrap.system_probe._check_ollama", return_value=False),
        patch("agentic_concierge.bootstrap.system_probe._check_vllm", return_value=False),
        patch("subprocess.run", side_effect=FileNotFoundError),
    ):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "ollama.com" in result.output or "Install" in result.output


# ---------------------------------------------------------------------------
# bootstrap command
# ---------------------------------------------------------------------------

def test_bootstrap_command_runs(tmp_path):
    from agentic_concierge.bootstrap.system_probe import SystemProbe
    fake_probe = SystemProbe(
        cpu_cores=4, cpu_arch="x86_64",
        ram_total_mb=8 * 1024, ram_available_mb=4 * 1024,
    )
    with (
        patch("agentic_concierge.bootstrap.first_run.probe_system", return_value=fake_probe),
        patch("agentic_concierge.bootstrap.first_run._pull_models", return_value=None),
        patch(
            "agentic_concierge.bootstrap.detected.detected_path",
            return_value=tmp_path / "detected.json",
        ),
    ):
        result = runner.invoke(app, ["bootstrap", "--non-interactive"])
    assert result.exit_code == 0
    assert "Bootstrap complete" in result.output


def test_bootstrap_invalid_profile_exits_nonzero():
    result = runner.invoke(app, ["bootstrap", "--profile", "ultraextreme"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# P11-8: doctor shows browser and chromadb rows
# ---------------------------------------------------------------------------

def _invoke_doctor():
    """Helper: invoke doctor with all external calls mocked."""
    with (
        patch("agentic_concierge.bootstrap.system_probe.probe_system", return_value=_sample_probe()),
        patch("agentic_concierge.bootstrap.detected.load_detected", return_value=_sample_profile()),
        patch(
            "agentic_concierge.bootstrap.backend_manager.BackendManager.probe_all",
            return_value=_healthy_backends(),
        ),
        patch("agentic_concierge.bootstrap.system_probe._check_internet", return_value=True),
        patch("agentic_concierge.bootstrap.system_probe._check_ollama", return_value=True),
        patch("agentic_concierge.bootstrap.system_probe._check_vllm", return_value=False),
        patch("subprocess.run", side_effect=FileNotFoundError),
    ):
        return runner.invoke(app, ["doctor"])


def test_doctor_shows_browser_row():
    result = _invoke_doctor()
    assert result.exit_code == 0
    assert "browser" in result.output


def test_doctor_shows_chromadb_row():
    result = _invoke_doctor()
    assert result.exit_code == 0
    assert "chromadb" in result.output


# ---------------------------------------------------------------------------
# P15: doctor uses launcher venv pip path in install hints
# ---------------------------------------------------------------------------

def test_doctor_launcher_venv_pip_hint(tmp_path):
    """When the launcher venv pip exists, doctor install hints use its full path."""
    # Create a fake launcher pip binary
    fake_pip = tmp_path / ".local/share/agentic-concierge/venv/bin/pip"
    fake_pip.parent.mkdir(parents=True)
    fake_pip.write_text("#!/bin/sh\n")

    with (
        patch("agentic_concierge.bootstrap.system_probe.probe_system", return_value=_sample_probe()),
        patch("agentic_concierge.bootstrap.detected.load_detected", return_value=_sample_profile()),
        patch(
            "agentic_concierge.bootstrap.backend_manager.BackendManager.probe_all",
            return_value=_healthy_backends(),
        ),
        patch("agentic_concierge.bootstrap.system_probe._check_internet", return_value=True),
        patch("agentic_concierge.bootstrap.system_probe._check_ollama", return_value=True),
        patch("agentic_concierge.bootstrap.system_probe._check_vllm", return_value=False),
        patch("subprocess.run", side_effect=FileNotFoundError),
        # Patch Path.home() to point at our tmp_path so the launcher pip check finds it
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    # The install hints should reference the launcher venv pip path (may be
    # wrapped across multiple lines by the Rich table, so check for a
    # distinctive substring that will appear on a single line).
    assert "venv/bin/pip" in result.output


def test_doctor_regular_pip_hint_when_no_launcher():
    """Without launcher venv, doctor install hints use plain 'pip'."""
    result = _invoke_doctor()
    assert result.exit_code == 0
    # At minimum, hints should mention 'pip install' (not a launcher path)
    assert "pip" in result.output
