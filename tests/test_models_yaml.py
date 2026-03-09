"""Tests for externalized models.yaml — tier model recommendations and nano config.

Verifies:
- Shipped default YAML loads correctly and matches v1 hardcoded values.
- User YAML override changes tier model recommendations.
- Nano config is overridable.
- Cache invalidation via reload functions.
- Backward compatibility: existing model_advisor behaviour unchanged.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
import yaml

from agentic_concierge.bootstrap.model_advisor import (
    _FALLBACK_MODEL_CTX_MB,
    _FALLBACK_MODEL_TABLE,
    _loaded_model_ctx_mb,
    _loaded_model_table,
    advise_profile,
    reload_models_config,
)
from agentic_concierge.bootstrap.system_probe import SystemProbe
from agentic_concierge.config.constants import (
    _FALLBACK_NANO_GGUF_FILENAME,
    _FALLBACK_NANO_GGUF_URL,
    nano_gguf_filename,
    nano_gguf_url,
    reload_nano_config,
)
from agentic_concierge.config.features import ProfileTier


def _probe(ram_mb: int, available_mb: int | None = None, cpu_cores: int = 8) -> SystemProbe:
    return SystemProbe(
        cpu_cores=cpu_cores,
        cpu_arch="x86_64",
        ram_total_mb=ram_mb,
        ram_available_mb=available_mb if available_mb is not None else ram_mb // 2,
        gpu_devices=[],
    )


# ---------------------------------------------------------------------------
# Shipped YAML loads and matches v1 hardcoded values
# ---------------------------------------------------------------------------


class TestShippedDefaults:
    """Verify shipped models.yaml matches the v1 hardcoded values exactly."""

    def setup_method(self):
        reload_models_config()
        reload_nano_config()

    def teardown_method(self):
        reload_models_config()
        reload_nano_config()

    def test_yaml_file_exists(self):
        from agentic_concierge.config.external import shipped_defaults_dir
        yaml_path = shipped_defaults_dir() / "models.yaml"
        assert yaml_path.is_file(), f"Shipped models.yaml not found at {yaml_path}"

    def test_yaml_has_catalog_version(self):
        from agentic_concierge.config.external import shipped_defaults_dir
        with open(shipped_defaults_dir() / "models.yaml") as f:
            data = yaml.safe_load(f)
        assert data["catalog_version"] == 1

    def test_tier_models_match_v1_fallback(self):
        """Loaded YAML tier_models must match the hardcoded fallback exactly."""
        table = _loaded_model_table()
        for tier in ProfileTier:
            assert tier in table, f"Missing tier {tier.name}"
            assert table[tier] == _FALLBACK_MODEL_TABLE[tier], (
                f"Tier {tier.name} mismatch: YAML={table[tier]}, fallback={_FALLBACK_MODEL_TABLE[tier]}"
            )

    def test_nano_gguf_filename_matches_v1(self):
        assert nano_gguf_filename() == _FALLBACK_NANO_GGUF_FILENAME

    def test_nano_gguf_url_matches_v1(self):
        assert nano_gguf_url() == _FALLBACK_NANO_GGUF_URL

    def test_model_ctx_mb_matches_v1(self):
        assert _loaded_model_ctx_mb() == _FALLBACK_MODEL_CTX_MB

    def test_all_tiers_present_in_yaml(self):
        from agentic_concierge.config.external import shipped_defaults_dir
        with open(shipped_defaults_dir() / "models.yaml") as f:
            data = yaml.safe_load(f)
        tier_models = data["tier_models"]
        for tier in ProfileTier:
            assert tier.name.lower() in tier_models, f"Tier {tier.name} missing from YAML"

    def test_each_tier_has_all_roles(self):
        table = _loaded_model_table()
        for tier in ProfileTier:
            for role in ("routing", "fast", "quality"):
                assert role in table[tier], f"Tier {tier.name} missing role {role}"
                assert table[tier][role], f"Tier {tier.name} role {role} is empty"


# ---------------------------------------------------------------------------
# User YAML override
# ---------------------------------------------------------------------------


class TestUserOverride:
    """Verify user YAML can override tier models and nano config."""

    def setup_method(self):
        reload_models_config()
        reload_nano_config()

    def teardown_method(self):
        # Clean up env vars
        os.environ.pop("CONCIERGE_MODELS_PATH", None)
        reload_models_config()
        reload_nano_config()

    def test_user_override_changes_tier_quality_model(self, tmp_path):
        """User YAML overriding medium tier quality model propagates to advise_profile."""
        override = {
            "tier_models": {
                "medium": {
                    "routing": "qwen2.5:0.5b",
                    "fast": "qwen2.5:7b",
                    "quality": "qwen3.5:9b",
                }
            }
        }
        override_file = tmp_path / "models.yaml"
        override_file.write_text(yaml.dump(override))
        os.environ["CONCIERGE_MODELS_PATH"] = str(override_file)
        reload_models_config()

        profile = advise_profile(_probe(16 * 1024))  # MEDIUM tier
        assert profile.quality_model == "qwen3.5:9b"
        # Other tiers remain at default
        nano_profile = advise_profile(_probe(4 * 1024))
        assert nano_profile.fast_model == "qwen2.5:3b"

    def test_user_override_changes_nano_filename(self, tmp_path):
        override = {
            "nano": {
                "gguf_filename": "qwen3.5-3b-q4_k_m.gguf",
                "gguf_url": "https://example.com/qwen3.5-3b-q4_k_m.gguf",
            }
        }
        override_file = tmp_path / "models.yaml"
        override_file.write_text(yaml.dump(override))
        os.environ["CONCIERGE_MODELS_PATH"] = str(override_file)
        reload_nano_config()

        assert nano_gguf_filename() == "qwen3.5-3b-q4_k_m.gguf"
        assert nano_gguf_url() == "https://example.com/qwen3.5-3b-q4_k_m.gguf"

    def test_user_override_partial_tier_preserves_other_tiers(self, tmp_path):
        """Only the overridden tier changes; others stay at defaults."""
        override = {
            "tier_models": {
                "server": {
                    "routing": "xlam-2:8b",
                    "fast": "qwen3.5:32b",
                    "quality": "qwen3.5:72b",
                }
            }
        }
        override_file = tmp_path / "models.yaml"
        override_file.write_text(yaml.dump(override))
        os.environ["CONCIERGE_MODELS_PATH"] = str(override_file)
        reload_models_config()

        table = _loaded_model_table()
        assert table[ProfileTier.SERVER]["routing"] == "xlam-2:8b"
        assert table[ProfileTier.SERVER]["quality"] == "qwen3.5:72b"
        # SMALL tier unchanged
        assert table[ProfileTier.SMALL] == _FALLBACK_MODEL_TABLE[ProfileTier.SMALL]

    def test_override_model_ctx_mb(self, tmp_path):
        override = {"model_ctx_mb": 4096}
        override_file = tmp_path / "models.yaml"
        override_file.write_text(yaml.dump(override))
        os.environ["CONCIERGE_MODELS_PATH"] = str(override_file)
        reload_models_config()

        assert _loaded_model_ctx_mb() == 4096


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------


class TestCacheInvalidation:
    """Verify reload functions clear caches properly."""

    def setup_method(self):
        reload_models_config()
        reload_nano_config()

    def teardown_method(self):
        os.environ.pop("CONCIERGE_MODELS_PATH", None)
        reload_models_config()
        reload_nano_config()

    def test_reload_models_config_clears_cache(self, tmp_path):
        # Load defaults
        table1 = _loaded_model_table()
        assert table1[ProfileTier.NANO]["quality"] == "phi3:mini"

        # Apply override
        override = {
            "tier_models": {
                "nano": {"routing": "qwen2.5:0.5b", "fast": "qwen2.5:3b", "quality": "qwen3.5:3b"}
            }
        }
        override_file = tmp_path / "models.yaml"
        override_file.write_text(yaml.dump(override))
        os.environ["CONCIERGE_MODELS_PATH"] = str(override_file)
        reload_models_config()

        table2 = _loaded_model_table()
        assert table2[ProfileTier.NANO]["quality"] == "qwen3.5:3b"

    def test_reload_nano_config_clears_cache(self, tmp_path):
        # Load defaults
        fn1 = nano_gguf_filename()
        assert fn1 == _FALLBACK_NANO_GGUF_FILENAME

        # Apply override
        override = {"nano": {"gguf_filename": "new-model.gguf"}}
        override_file = tmp_path / "models.yaml"
        override_file.write_text(yaml.dump(override))
        os.environ["CONCIERGE_MODELS_PATH"] = str(override_file)
        reload_nano_config()

        assert nano_gguf_filename() == "new-model.gguf"


# ---------------------------------------------------------------------------
# Backward compatibility with advise_profile
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """Verify advise_profile produces identical results to v1."""

    def setup_method(self):
        reload_models_config()

    def teardown_method(self):
        reload_models_config()

    def test_nano_tier_models(self):
        profile = advise_profile(_probe(4 * 1024))
        assert profile.routing_model == "qwen2.5:0.5b"
        assert profile.fast_model == "qwen2.5:3b"
        assert profile.quality_model == "phi3:mini"

    def test_small_tier_models(self):
        profile = advise_profile(_probe(8 * 1024))
        assert profile.routing_model == "qwen2.5:0.5b"
        assert profile.fast_model == "qwen2.5:7b"
        assert profile.quality_model == "qwen2.5:7b"

    def test_medium_tier_models(self):
        profile = advise_profile(_probe(16 * 1024))
        assert profile.routing_model == "qwen2.5:0.5b"
        assert profile.fast_model == "qwen2.5:7b"
        assert profile.quality_model == "qwen2.5:14b"

    def test_large_tier_models(self):
        profile = advise_profile(_probe(32 * 1024))
        assert profile.routing_model == "qwen2.5:0.5b"
        assert profile.fast_model == "qwen2.5:14b"
        assert profile.quality_model == "qwen2.5:32b"

    def test_server_tier_models(self):
        profile = advise_profile(_probe(128 * 1024))
        assert profile.routing_model == "qwen2.5:0.5b"
        assert profile.fast_model == "qwen2.5:32b"
        assert profile.quality_model == "qwen2.5:72b"


# ---------------------------------------------------------------------------
# Error resilience
# ---------------------------------------------------------------------------


class TestErrorResilience:
    """Verify graceful fallback when YAML is malformed or missing."""

    def setup_method(self):
        reload_models_config()
        reload_nano_config()

    def teardown_method(self):
        os.environ.pop("CONCIERGE_MODELS_PATH", None)
        reload_models_config()
        reload_nano_config()

    def test_malformed_yaml_falls_back(self, tmp_path):
        bad_file = tmp_path / "models.yaml"
        bad_file.write_text("not: [valid: yaml: structure")
        os.environ["CONCIERGE_MODELS_PATH"] = str(bad_file)
        reload_models_config()
        reload_nano_config()

        # Should get fallback values, not crash
        table = _loaded_model_table()
        assert table[ProfileTier.NANO] == _FALLBACK_MODEL_TABLE[ProfileTier.NANO]
        assert nano_gguf_filename() == _FALLBACK_NANO_GGUF_FILENAME

    def test_empty_yaml_falls_back(self, tmp_path):
        empty_file = tmp_path / "models.yaml"
        empty_file.write_text("")
        os.environ["CONCIERGE_MODELS_PATH"] = str(empty_file)
        reload_models_config()

        table = _loaded_model_table()
        for tier in ProfileTier:
            assert table[tier] == _FALLBACK_MODEL_TABLE[tier]

    def test_missing_tier_filled_from_fallback(self, tmp_path):
        """YAML with only one tier defined still has all tiers."""
        override = {
            "tier_models": {
                "nano": {"routing": "x:1b", "fast": "x:3b", "quality": "x:3b"}
            }
        }
        override_file = tmp_path / "models.yaml"
        override_file.write_text(yaml.dump(override))
        os.environ["CONCIERGE_MODELS_PATH"] = str(override_file)
        reload_models_config()

        table = _loaded_model_table()
        assert table[ProfileTier.NANO]["routing"] == "x:1b"
        # SERVER tier filled from fallback
        assert table[ProfileTier.SERVER] == _FALLBACK_MODEL_TABLE[ProfileTier.SERVER]

    def test_unknown_tier_name_ignored(self, tmp_path):
        override = {
            "tier_models": {
                "nano": {"routing": "qwen2.5:0.5b", "fast": "qwen2.5:3b", "quality": "phi3:mini"},
                "imaginary_tier": {"routing": "a", "fast": "b", "quality": "c"},
            }
        }
        override_file = tmp_path / "models.yaml"
        override_file.write_text(yaml.dump(override))
        os.environ["CONCIERGE_MODELS_PATH"] = str(override_file)
        reload_models_config()

        table = _loaded_model_table()
        assert len(table) == len(ProfileTier)  # No extra tier added
