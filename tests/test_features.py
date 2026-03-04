"""Tests for config/features.py: ProfileTier, Feature, FeatureSet, FeatureDisabledError."""

from __future__ import annotations

import pytest

from agentic_concierge.config.features import (
    BACKEND_PRIORITY,
    Feature,
    FeatureDisabledError,
    FeatureSet,
    PROFILE_FEATURES,
    ProfileTier,
)


# ---------------------------------------------------------------------------
# ProfileTier
# ---------------------------------------------------------------------------

def test_profile_tier_values():
    assert ProfileTier.NANO.value == "nano"
    assert ProfileTier.SMALL.value == "small"
    assert ProfileTier.MEDIUM.value == "medium"
    assert ProfileTier.LARGE.value == "large"
    assert ProfileTier.SERVER.value == "server"


# ---------------------------------------------------------------------------
# PROFILE_FEATURES defaults
# ---------------------------------------------------------------------------

def test_nano_features():
    fs = PROFILE_FEATURES[ProfileTier.NANO]
    assert Feature.INPROCESS in fs
    assert Feature.CLOUD in fs
    assert Feature.OLLAMA not in fs
    assert Feature.VLLM not in fs
    assert Feature.MCP not in fs


def test_small_features():
    fs = PROFILE_FEATURES[ProfileTier.SMALL]
    assert Feature.OLLAMA in fs
    assert Feature.MCP in fs
    assert Feature.VLLM not in fs
    assert Feature.EMBEDDING not in fs


def test_medium_features():
    fs = PROFILE_FEATURES[ProfileTier.MEDIUM]
    assert Feature.VLLM in fs
    assert Feature.EMBEDDING in fs
    assert Feature.CONTAINER not in fs


def test_large_features():
    fs = PROFILE_FEATURES[ProfileTier.LARGE]
    assert Feature.CONTAINER in fs
    assert Feature.TELEMETRY not in fs


def test_server_features():
    fs = PROFILE_FEATURES[ProfileTier.SERVER]
    assert Feature.TELEMETRY in fs
    # Server includes Ollama as fallback (adaptive backend resolution)
    assert Feature.OLLAMA in fs
    assert Feature.VLLM in fs


# ---------------------------------------------------------------------------
# FeatureSet.from_profile
# ---------------------------------------------------------------------------

class _Overrides:
    """Helper: accepts keyword args as Optional[bool] feature overrides."""
    def __init__(self, **kwargs):
        self._d = kwargs

    def __getattr__(self, name: str):
        return self._d.get(name, None)


def test_from_profile_uses_defaults():
    overrides = _Overrides()
    fs = FeatureSet.from_profile(ProfileTier.NANO, overrides)
    assert fs.is_enabled(Feature.INPROCESS)
    assert fs.is_enabled(Feature.CLOUD)
    assert not fs.is_enabled(Feature.OLLAMA)


def test_from_profile_override_enables():
    overrides = _Overrides(ollama=True)
    fs = FeatureSet.from_profile(ProfileTier.NANO, overrides)
    assert fs.is_enabled(Feature.OLLAMA)  # forced on despite nano default


def test_from_profile_override_disables():
    overrides = _Overrides(inprocess=False)
    fs = FeatureSet.from_profile(ProfileTier.SMALL, overrides)
    assert not fs.is_enabled(Feature.INPROCESS)  # forced off


# ---------------------------------------------------------------------------
# FeatureSet.require
# ---------------------------------------------------------------------------

def test_require_passes_when_enabled():
    fs = FeatureSet(enabled=frozenset({Feature.OLLAMA}))
    fs.require(Feature.OLLAMA)  # should not raise


def test_require_raises_when_disabled():
    fs = FeatureSet(enabled=frozenset())
    with pytest.raises(FeatureDisabledError) as exc_info:
        fs.require(Feature.VLLM, "Enable vllm in your config.")
    assert exc_info.value.feature == Feature.VLLM
    assert "vllm" in str(exc_info.value)
    assert "Enable vllm" in str(exc_info.value)


def test_require_error_has_feature_attribute():
    fs = FeatureSet(enabled=frozenset())
    with pytest.raises(FeatureDisabledError) as exc_info:
        fs.require(Feature.BROWSER)
    assert exc_info.value.feature == Feature.BROWSER


# ---------------------------------------------------------------------------
# FeatureSet.all_enabled
# ---------------------------------------------------------------------------

def test_all_enabled_contains_every_feature():
    fs = FeatureSet.all_enabled()
    for f in Feature:
        assert fs.is_enabled(f), f"{f.value} should be enabled"


# ---------------------------------------------------------------------------
# P11-2: BROWSER in PROFILE_FEATURES (small, medium, large, server; NOT nano)
# ---------------------------------------------------------------------------

def test_browser_not_in_nano_features():
    assert Feature.BROWSER not in PROFILE_FEATURES[ProfileTier.NANO]


def test_browser_in_small_features():
    assert Feature.BROWSER in PROFILE_FEATURES[ProfileTier.SMALL]


def test_browser_in_medium_features():
    assert Feature.BROWSER in PROFILE_FEATURES[ProfileTier.MEDIUM]


def test_browser_in_large_and_server_features():
    assert Feature.BROWSER in PROFILE_FEATURES[ProfileTier.LARGE]
    assert Feature.BROWSER in PROFILE_FEATURES[ProfileTier.SERVER]


# ---------------------------------------------------------------------------
# Adaptive backend resolution: BACKEND_PRIORITY and SERVER profile
# ---------------------------------------------------------------------------

def test_server_profile_includes_ollama():
    """Ollama is enabled in SERVER profile for fallback."""
    assert Feature.OLLAMA in PROFILE_FEATURES[ProfileTier.SERVER]


def test_backend_priority_all_tiers():
    """BACKEND_PRIORITY has entries for every ProfileTier."""
    for tier in ProfileTier:
        assert tier in BACKEND_PRIORITY, f"Missing BACKEND_PRIORITY for {tier.value}"
        assert len(BACKEND_PRIORITY[tier]) >= 2, f"BACKEND_PRIORITY[{tier.value}] too short"
