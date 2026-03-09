"""Tests for externalized backends.yaml — backend URLs and tier priority.

Verifies:
- Shipped default YAML loads correctly and matches v1 hardcoded values.
- User YAML override adds custom backend URLs.
- User YAML override changes tier priority order.
- Cache invalidation via reload functions.
- Backward compatibility: existing module-level aliases still work.
- Error resilience: malformed/missing YAML falls back gracefully.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from agentic_concierge.config.constants import (
    _FALLBACK_BACKEND_URLS,
    default_backend_urls,
    reload_backend_config,
)
from agentic_concierge.config.features import (
    _FALLBACK_BACKEND_PRIORITY,
    ProfileTier,
    backend_priority,
    reload_backend_priority,
)


@pytest.fixture(autouse=True)
def _clean_caches():
    """Ensure each test starts and ends with clean caches."""
    reload_backend_config()
    reload_backend_priority()
    yield
    os.environ.pop("CONCIERGE_BACKENDS_PATH", None)
    reload_backend_config()
    reload_backend_priority()


# ---------------------------------------------------------------------------
# Shipped YAML loads and matches v1 hardcoded values
# ---------------------------------------------------------------------------


class TestShippedDefaults:
    """Verify shipped backends.yaml matches the v1 hardcoded values exactly."""

    def test_yaml_file_exists(self):
        from agentic_concierge.config.external import shipped_defaults_dir
        yaml_path = shipped_defaults_dir() / "backends.yaml"
        assert yaml_path.is_file(), f"Shipped backends.yaml not found at {yaml_path}"

    def test_yaml_has_version(self):
        from agentic_concierge.config.external import shipped_defaults_dir
        with open(shipped_defaults_dir() / "backends.yaml") as f:
            data = yaml.safe_load(f)
        assert data["backends_version"] == 1

    def test_backend_urls_match_v1_fallback(self):
        urls = default_backend_urls()
        for name, url in _FALLBACK_BACKEND_URLS.items():
            assert name in urls, f"Missing backend {name}"
            assert urls[name] == url, (
                f"Backend {name} URL mismatch: YAML={urls[name]}, fallback={url}"
            )

    def test_backend_urls_contain_ollama_and_vllm(self):
        urls = default_backend_urls()
        assert "ollama" in urls
        assert "vllm" in urls

    def test_tier_priority_match_v1_fallback(self):
        bp = backend_priority()
        for tier in ProfileTier:
            assert tier in bp, f"Missing tier {tier.name}"
            assert bp[tier] == _FALLBACK_BACKEND_PRIORITY[tier], (
                f"Tier {tier.name} priority mismatch: "
                f"YAML={bp[tier]}, fallback={_FALLBACK_BACKEND_PRIORITY[tier]}"
            )

    def test_all_tiers_present_in_priority(self):
        bp = backend_priority()
        for tier in ProfileTier:
            assert tier in bp

    def test_every_priority_list_is_nonempty(self):
        bp = backend_priority()
        for tier in ProfileTier:
            assert len(bp[tier]) >= 2, f"Tier {tier.name} priority list too short"

    def test_yaml_has_all_backends(self):
        from agentic_concierge.config.external import shipped_defaults_dir
        with open(shipped_defaults_dir() / "backends.yaml") as f:
            data = yaml.safe_load(f)
        backends = data["backends"]
        assert "ollama" in backends
        assert "vllm" in backends

    def test_yaml_has_all_tier_priorities(self):
        from agentic_concierge.config.external import shipped_defaults_dir
        with open(shipped_defaults_dir() / "backends.yaml") as f:
            data = yaml.safe_load(f)
        tier_priority = data["tier_priority"]
        for tier in ProfileTier:
            assert tier.name.lower() in tier_priority, (
                f"Tier {tier.name} missing from YAML tier_priority"
            )


# ---------------------------------------------------------------------------
# User YAML override — backend URLs
# ---------------------------------------------------------------------------


class TestUserOverrideUrls:
    """Verify user YAML can add/change backend URLs."""

    def test_override_changes_ollama_url(self, tmp_path):
        override = {
            "backends": {
                "ollama": {"url": "http://remote-host:11434/v1"},
            }
        }
        override_file = tmp_path / "backends.yaml"
        override_file.write_text(yaml.dump(override))
        os.environ["CONCIERGE_BACKENDS_PATH"] = str(override_file)
        reload_backend_config()

        urls = default_backend_urls()
        assert urls["ollama"] == "http://remote-host:11434/v1"
        # vllm should still have its default
        assert urls["vllm"] == _FALLBACK_BACKEND_URLS["vllm"]

    def test_override_adds_new_backend(self, tmp_path):
        override = {
            "backends": {
                "llama_cpp": {"url": "http://localhost:8080/v1"},
            }
        }
        override_file = tmp_path / "backends.yaml"
        override_file.write_text(yaml.dump(override))
        os.environ["CONCIERGE_BACKENDS_PATH"] = str(override_file)
        reload_backend_config()

        urls = default_backend_urls()
        assert urls["llama_cpp"] == "http://localhost:8080/v1"
        # Existing backends preserved
        assert "ollama" in urls
        assert "vllm" in urls

    def test_override_multiple_backends(self, tmp_path):
        override = {
            "backends": {
                "ollama": {"url": "http://nas:11434/v1"},
                "vllm": {"url": "http://gpu-server:8000/v1"},
                "mlx": {"url": "http://localhost:8080/v1"},
            }
        }
        override_file = tmp_path / "backends.yaml"
        override_file.write_text(yaml.dump(override))
        os.environ["CONCIERGE_BACKENDS_PATH"] = str(override_file)
        reload_backend_config()

        urls = default_backend_urls()
        assert urls["ollama"] == "http://nas:11434/v1"
        assert urls["vllm"] == "http://gpu-server:8000/v1"
        assert urls["mlx"] == "http://localhost:8080/v1"

    def test_returned_dict_is_a_copy(self):
        """Mutating returned dict must not affect the cached value."""
        urls1 = default_backend_urls()
        urls1["ollama"] = "http://mutated:1234/v1"
        urls2 = default_backend_urls()
        assert urls2["ollama"] == _FALLBACK_BACKEND_URLS["ollama"]


# ---------------------------------------------------------------------------
# User YAML override — tier priority
# ---------------------------------------------------------------------------


class TestUserOverridePriority:
    """Verify user YAML can change tier backend priority."""

    def test_override_changes_nano_priority(self, tmp_path):
        override = {
            "tier_priority": {
                "nano": ["ollama", "cloud"],
            }
        }
        override_file = tmp_path / "backends.yaml"
        override_file.write_text(yaml.dump(override))
        os.environ["CONCIERGE_BACKENDS_PATH"] = str(override_file)
        reload_backend_priority()

        bp = backend_priority()
        assert bp[ProfileTier.NANO] == ["ollama", "cloud"]
        # Other tiers unchanged
        assert bp[ProfileTier.SERVER] == _FALLBACK_BACKEND_PRIORITY[ProfileTier.SERVER]

    def test_override_adds_custom_backend_to_priority(self, tmp_path):
        override = {
            "tier_priority": {
                "medium": ["ollama", "llama_cpp", "vllm", "cloud"],
            }
        }
        override_file = tmp_path / "backends.yaml"
        override_file.write_text(yaml.dump(override))
        os.environ["CONCIERGE_BACKENDS_PATH"] = str(override_file)
        reload_backend_priority()

        bp = backend_priority()
        assert bp[ProfileTier.MEDIUM] == ["ollama", "llama_cpp", "vllm", "cloud"]

    def test_override_partial_tiers_preserves_others(self, tmp_path):
        override = {
            "tier_priority": {
                "server": ["vllm", "cloud"],
            }
        }
        override_file = tmp_path / "backends.yaml"
        override_file.write_text(yaml.dump(override))
        os.environ["CONCIERGE_BACKENDS_PATH"] = str(override_file)
        reload_backend_priority()

        bp = backend_priority()
        assert bp[ProfileTier.SERVER] == ["vllm", "cloud"]
        assert bp[ProfileTier.NANO] == _FALLBACK_BACKEND_PRIORITY[ProfileTier.NANO]
        assert bp[ProfileTier.LARGE] == _FALLBACK_BACKEND_PRIORITY[ProfileTier.LARGE]


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------


class TestCacheInvalidation:

    def test_reload_backend_config_clears_url_cache(self, tmp_path):
        urls1 = default_backend_urls()
        assert urls1["ollama"] == _FALLBACK_BACKEND_URLS["ollama"]

        override = {"backends": {"ollama": {"url": "http://changed:11434/v1"}}}
        override_file = tmp_path / "backends.yaml"
        override_file.write_text(yaml.dump(override))
        os.environ["CONCIERGE_BACKENDS_PATH"] = str(override_file)
        reload_backend_config()

        urls2 = default_backend_urls()
        assert urls2["ollama"] == "http://changed:11434/v1"

    def test_reload_backend_priority_clears_priority_cache(self, tmp_path):
        bp1 = backend_priority()
        assert bp1[ProfileTier.NANO] == _FALLBACK_BACKEND_PRIORITY[ProfileTier.NANO]

        override = {"tier_priority": {"nano": ["cloud", "ollama"]}}
        override_file = tmp_path / "backends.yaml"
        override_file.write_text(yaml.dump(override))
        os.environ["CONCIERGE_BACKENDS_PATH"] = str(override_file)
        reload_backend_priority()

        bp2 = backend_priority()
        assert bp2[ProfileTier.NANO] == ["cloud", "ollama"]


# ---------------------------------------------------------------------------
# Error resilience
# ---------------------------------------------------------------------------


class TestErrorResilience:
    """Verify graceful fallback when YAML is malformed or missing."""

    def test_malformed_yaml_urls_falls_back(self, tmp_path):
        bad_file = tmp_path / "backends.yaml"
        bad_file.write_text("backends: [invalid, list, not, mapping]")
        os.environ["CONCIERGE_BACKENDS_PATH"] = str(bad_file)
        reload_backend_config()

        urls = default_backend_urls()
        assert urls == _FALLBACK_BACKEND_URLS

    def test_malformed_yaml_priority_falls_back(self, tmp_path):
        bad_file = tmp_path / "backends.yaml"
        bad_file.write_text("tier_priority: not_a_mapping")
        os.environ["CONCIERGE_BACKENDS_PATH"] = str(bad_file)
        reload_backend_priority()

        bp = backend_priority()
        for tier in ProfileTier:
            assert bp[tier] == _FALLBACK_BACKEND_PRIORITY[tier]

    def test_empty_yaml_falls_back(self, tmp_path):
        empty_file = tmp_path / "backends.yaml"
        empty_file.write_text("")
        os.environ["CONCIERGE_BACKENDS_PATH"] = str(empty_file)
        reload_backend_config()
        reload_backend_priority()

        urls = default_backend_urls()
        assert urls == _FALLBACK_BACKEND_URLS
        bp = backend_priority()
        for tier in ProfileTier:
            assert bp[tier] == _FALLBACK_BACKEND_PRIORITY[tier]

    def test_missing_url_field_in_backend_skipped(self, tmp_path):
        """A backend entry without 'url' is skipped; fallback fills it."""
        override = {
            "backends": {
                "ollama": {"health_endpoint": "/api/tags"},  # no url
            }
        }
        override_file = tmp_path / "backends.yaml"
        override_file.write_text(yaml.dump(override))
        os.environ["CONCIERGE_BACKENDS_PATH"] = str(override_file)
        reload_backend_config()

        urls = default_backend_urls()
        # Ollama URL filled from fallback since override had no 'url'
        assert urls["ollama"] == _FALLBACK_BACKEND_URLS["ollama"]

    def test_unknown_tier_in_priority_ignored(self, tmp_path):
        override = {
            "tier_priority": {
                "nano": ["ollama", "cloud"],
                "imaginary": ["magic_backend"],
            }
        }
        override_file = tmp_path / "backends.yaml"
        override_file.write_text(yaml.dump(override))
        os.environ["CONCIERGE_BACKENDS_PATH"] = str(override_file)
        reload_backend_priority()

        bp = backend_priority()
        assert len(bp) == len(ProfileTier)  # No extra tier
        assert bp[ProfileTier.NANO] == ["ollama", "cloud"]

    def test_non_list_priority_value_ignored(self, tmp_path):
        override = {
            "tier_priority": {
                "nano": "not_a_list",
            }
        }
        override_file = tmp_path / "backends.yaml"
        override_file.write_text(yaml.dump(override))
        os.environ["CONCIERGE_BACKENDS_PATH"] = str(override_file)
        reload_backend_priority()

        bp = backend_priority()
        # Nano filled from fallback since override had invalid type
        assert bp[ProfileTier.NANO] == _FALLBACK_BACKEND_PRIORITY[ProfileTier.NANO]
