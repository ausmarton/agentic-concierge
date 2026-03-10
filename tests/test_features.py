"""Tests for config/features.py: ProfileTier, Feature, FeatureSet, FeatureDisabledError."""

from __future__ import annotations

import pytest

from agentic_concierge.config.features import (
    Feature,
    FeatureDisabledError,
    FeatureSet,
    PROFILE_FEATURES,
    ProfileTier,
    backend_priority,
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
    assert Feature.OLLAMA in fs
    assert Feature.CLOUD in fs
    assert Feature.VLLM not in fs
    assert Feature.MCP not in fs


def test_small_features():
    fs = PROFILE_FEATURES[ProfileTier.SMALL]
    assert Feature.OLLAMA in fs
    assert Feature.LLAMA_CPP in fs
    assert Feature.MCP in fs
    assert Feature.VLLM not in fs
    assert Feature.EMBEDDING not in fs


def test_medium_features():
    fs = PROFILE_FEATURES[ProfileTier.MEDIUM]
    assert Feature.OLLAMA in fs
    assert Feature.LLAMA_CPP in fs
    assert Feature.EMBEDDING in fs
    assert Feature.VLLM not in fs   # vLLM only on LARGE/SERVER
    assert Feature.CONTAINER not in fs


def test_large_features():
    fs = PROFILE_FEATURES[ProfileTier.LARGE]
    assert Feature.LLAMA_CPP in fs
    assert Feature.VLLM in fs
    assert Feature.CONTAINER in fs
    assert Feature.TELEMETRY not in fs


def test_server_features():
    fs = PROFILE_FEATURES[ProfileTier.SERVER]
    assert Feature.TELEMETRY in fs
    assert Feature.OLLAMA in fs
    assert Feature.LLAMA_CPP in fs
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
    assert fs.is_enabled(Feature.OLLAMA)
    assert fs.is_enabled(Feature.CLOUD)
    assert not fs.is_enabled(Feature.VLLM)


def test_from_profile_override_enables():
    overrides = _Overrides(vllm=True)
    fs = FeatureSet.from_profile(ProfileTier.NANO, overrides)
    assert fs.is_enabled(Feature.VLLM)  # forced on despite nano default


def test_from_profile_override_disables():
    overrides = _Overrides(ollama=False)
    fs = FeatureSet.from_profile(ProfileTier.SMALL, overrides)
    assert not fs.is_enabled(Feature.OLLAMA)  # forced off


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
# Adaptive backend resolution: backend_priority() and SERVER profile
# ---------------------------------------------------------------------------

def test_server_profile_includes_ollama():
    """Ollama is enabled in SERVER profile for fallback."""
    assert Feature.OLLAMA in PROFILE_FEATURES[ProfileTier.SERVER]


def test_backend_priority_all_tiers():
    """backend_priority() has entries for every ProfileTier."""
    bp = backend_priority()
    for tier in ProfileTier:
        assert tier in bp, f"Missing backend_priority for {tier.value}"
        assert len(bp[tier]) >= 2, f"backend_priority[{tier.value}] too short"


# ---------------------------------------------------------------------------
# Backend consolidation: inprocess removed, llama_cpp first-class
# ---------------------------------------------------------------------------

def test_no_inprocess_feature():
    """Feature enum must NOT contain INPROCESS after backend consolidation."""
    feature_values = {f.value for f in Feature}
    assert "inprocess" not in feature_values


def test_no_inprocess_in_any_profile():
    """No profile tier should reference an inprocess feature."""
    for tier in ProfileTier:
        feature_values = {f.value for f in PROFILE_FEATURES[tier]}
        assert "inprocess" not in feature_values, f"inprocess found in {tier.value}"


def test_no_inprocess_in_backend_priority():
    """inprocess must not appear in any tier's backend priority list."""
    bp = backend_priority()
    for tier in ProfileTier:
        assert "inprocess" not in bp[tier], f"inprocess in priority for {tier.value}"


def test_llama_cpp_in_small_plus_profiles():
    """llama_cpp should be enabled on SMALL and above (not NANO)."""
    assert Feature.LLAMA_CPP not in PROFILE_FEATURES[ProfileTier.NANO]
    assert Feature.LLAMA_CPP in PROFILE_FEATURES[ProfileTier.SMALL]
    assert Feature.LLAMA_CPP in PROFILE_FEATURES[ProfileTier.MEDIUM]
    assert Feature.LLAMA_CPP in PROFILE_FEATURES[ProfileTier.LARGE]
    assert Feature.LLAMA_CPP in PROFILE_FEATURES[ProfileTier.SERVER]


def test_llama_cpp_in_all_backend_priorities():
    """llama_cpp should appear in every tier's backend priority list."""
    bp = backend_priority()
    for tier in ProfileTier:
        assert "llama_cpp" in bp[tier], f"llama_cpp missing from {tier.value} priority"


def test_vllm_first_in_server_priority():
    """SERVER tier should have vllm as first backend."""
    bp = backend_priority()
    assert bp[ProfileTier.SERVER][0] == "vllm"


def test_vllm_not_in_nano_small_medium_priority():
    """vllm should not appear in NANO, SMALL, or MEDIUM priority."""
    bp = backend_priority()
    assert "vllm" not in bp[ProfileTier.NANO]
    assert "vllm" not in bp[ProfileTier.SMALL]
    assert "vllm" not in bp[ProfileTier.MEDIUM]


def test_vllm_in_large_and_server_priority():
    """vllm should appear in LARGE and SERVER priority."""
    bp = backend_priority()
    assert "vllm" in bp[ProfileTier.LARGE]
    assert "vllm" in bp[ProfileTier.SERVER]


def test_cloud_in_all_priorities():
    """cloud should be in every tier's priority as a fallback."""
    bp = backend_priority()
    for tier in ProfileTier:
        assert "cloud" in bp[tier], f"cloud missing from {tier.value} priority"
