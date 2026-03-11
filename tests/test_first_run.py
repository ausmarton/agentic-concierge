"""Tests for bootstrap/first_run.py, bootstrap/detected.py, and the bootstrap CLI command."""

from __future__ import annotations

import json
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_concierge.bootstrap.detected import (
    detected_path,
    is_first_run,
    load_detected,
    save_detected,
)
from agentic_concierge.bootstrap.model_advisor import SystemProfile
from agentic_concierge.config.features import ProfileTier


def _sample_profile(tier: ProfileTier = ProfileTier.SMALL) -> SystemProfile:
    return SystemProfile(
        tier=tier,
        routing_model="qwen2.5:0.5b",
        fast_model="qwen2.5:7b",
        quality_model="qwen2.5:7b",
        max_concurrent_agents=2,
        ram_total_mb=16384,
        ram_available_mb=8192,
        total_vram_mb=0,
        cpu_cores=8,
        cpu_arch="x86_64",
        gpu_count=0,
    )


# ---------------------------------------------------------------------------
# detected.py helpers
# ---------------------------------------------------------------------------

def test_save_and_load_detected(tmp_path):
    profile = _sample_profile()
    p = tmp_path / "detected.json"
    save_detected(profile, path=p)
    assert p.exists()
    loaded = load_detected(path=p)
    assert loaded is not None
    assert loaded.tier == ProfileTier.SMALL
    assert loaded.fast_model == "qwen2.5:7b"
    assert loaded.cpu_cores == 8


def test_load_detected_missing_returns_none(tmp_path):
    result = load_detected(path=tmp_path / "nonexistent.json")
    assert result is None


def test_load_detected_corrupt_returns_none(tmp_path):
    p = tmp_path / "detected.json"
    p.write_text("NOT JSON", encoding="utf-8")
    result = load_detected(path=p)
    assert result is None


def test_load_detected_wrong_tier_returns_none(tmp_path):
    p = tmp_path / "detected.json"
    p.write_text(json.dumps({"tier": "invalid_tier"}), encoding="utf-8")
    result = load_detected(path=p)
    assert result is None


def test_is_first_run_true_when_absent(tmp_path):
    assert is_first_run(path=tmp_path / "detected.json") is True


def test_is_first_run_false_after_save(tmp_path):
    p = tmp_path / "detected.json"
    save_detected(_sample_profile(), path=p)
    assert is_first_run(path=p) is False


def test_round_trip_all_tiers(tmp_path):
    for tier in ProfileTier:
        p = tmp_path / f"detected_{tier.value}.json"
        profile = _sample_profile(tier)
        save_detected(profile, path=p)
        loaded = load_detected(path=p)
        assert loaded.tier == tier


# ---------------------------------------------------------------------------
# first_run.run() — happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_first_run_returns_cached_profile(tmp_path):
    """If detected.json exists and no force, return cached profile (still probes for backend ensure)."""
    saved = _sample_profile(ProfileTier.MEDIUM)
    p = tmp_path / "detected.json"
    save_detected(saved, path=p)

    from agentic_concierge.bootstrap import first_run
    from agentic_concierge.bootstrap.system_probe import SystemProbe

    fake_probe = SystemProbe(
        cpu_cores=8, cpu_arch="x86_64",
        ram_total_mb=8 * 1024, ram_available_mb=4 * 1024,
    )
    with (
        patch("agentic_concierge.bootstrap.first_run.probe_system", return_value=fake_probe) as mock_probe,
        patch("subprocess.run", side_effect=FileNotFoundError),
    ):
        result = await first_run.run(interactive=False, detected_override=p)

    # Probe is called (needed for backend ensure steps) but cached profile is used
    mock_probe.assert_called_once()
    assert result.tier == ProfileTier.MEDIUM


@pytest.mark.asyncio
async def test_first_run_probes_on_first_run(tmp_path):
    p = tmp_path / "detected.json"

    from agentic_concierge.bootstrap import first_run
    from agentic_concierge.bootstrap.system_probe import SystemProbe

    fake_probe = SystemProbe(
        cpu_cores=8, cpu_arch="x86_64",
        ram_total_mb=8 * 1024, ram_available_mb=4 * 1024,
    )
    with (
        patch("agentic_concierge.bootstrap.first_run.probe_system", return_value=fake_probe),
        patch("agentic_concierge.bootstrap.first_run._pull_models", return_value=None),
    ):
        result = await first_run.run(interactive=False, detected_override=p)

    assert p.exists()
    assert result.tier == ProfileTier.SMALL


@pytest.mark.asyncio
async def test_first_run_force_profile(tmp_path):
    p = tmp_path / "detected.json"

    from agentic_concierge.bootstrap import first_run
    from agentic_concierge.bootstrap.system_probe import SystemProbe

    fake_probe = SystemProbe(
        cpu_cores=4, cpu_arch="x86_64",
        ram_total_mb=4 * 1024, ram_available_mb=2 * 1024,
    )
    with (
        patch("agentic_concierge.bootstrap.first_run.probe_system", return_value=fake_probe),
        patch("agentic_concierge.bootstrap.first_run._pull_models", return_value=None),
    ):
        result = await first_run.run(
            interactive=False, force_profile="medium", detected_override=p
        )

    assert result.tier == ProfileTier.MEDIUM


@pytest.mark.asyncio
async def test_first_run_invalid_force_profile_raises(tmp_path):
    p = tmp_path / "detected.json"

    from agentic_concierge.bootstrap import first_run
    from agentic_concierge.bootstrap.system_probe import SystemProbe

    fake_probe = SystemProbe(
        cpu_cores=4, cpu_arch="x86_64",
        ram_total_mb=4 * 1024, ram_available_mb=2 * 1024,
    )
    with patch("agentic_concierge.bootstrap.first_run.probe_system", return_value=fake_probe):
        with pytest.raises(ValueError, match="Unknown profile"):
            await first_run.run(
                interactive=False, force_profile="bogus", detected_override=p
            )


@pytest.mark.asyncio
async def test_first_run_non_interactive_skips_model_pull(tmp_path):
    p = tmp_path / "detected.json"

    from agentic_concierge.bootstrap import first_run
    from agentic_concierge.bootstrap.system_probe import SystemProbe

    fake_probe = SystemProbe(
        cpu_cores=8, cpu_arch="x86_64",
        ram_total_mb=16 * 1024, ram_available_mb=8 * 1024,
        ollama_reachable=True,  # reachable but non-interactive → no pull
    )
    with (
        patch("agentic_concierge.bootstrap.first_run.probe_system", return_value=fake_probe),
        patch("agentic_concierge.bootstrap.first_run._pull_models") as mock_pull,
    ):
        await first_run.run(interactive=False, detected_override=p)

    mock_pull.assert_not_called()
