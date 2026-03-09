"""Tests for externalized YAML configuration loader (config/external.py).

Tests cover:
- config_dir() and shipped_defaults_dir() resolution
- deep_merge() behavior (scalars, nested dicts, lists)
- load_yaml_config() hierarchy (shipped → user → env)
- load_all_yaml_configs() directory loading
- Error handling (missing files, invalid YAML, non-dict YAML)
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
import yaml

from agentic_concierge.config.external import (
    config_dir,
    deep_merge,
    load_all_yaml_configs,
    load_yaml_config,
    shipped_defaults_dir,
)


# ---------------------------------------------------------------------------
# config_dir()
# ---------------------------------------------------------------------------


class TestConfigDir:
    def test_respects_xdg_config_home(self, monkeypatch, tmp_path):
        """config_dir() uses $XDG_CONFIG_HOME when set."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        assert config_dir() == tmp_path / "concierge"

    def test_default_is_dot_config(self, monkeypatch):
        """config_dir() falls back to ~/.config/concierge/ when XDG unset."""
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        assert config_dir() == Path.home() / ".config" / "concierge"


# ---------------------------------------------------------------------------
# shipped_defaults_dir()
# ---------------------------------------------------------------------------


class TestShippedDefaultsDir:
    def test_exists(self):
        """Shipped defaults directory must exist in the installed package."""
        d = shipped_defaults_dir()
        assert d.is_dir(), f"Shipped defaults dir does not exist: {d}"

    def test_contains_readme(self):
        """Shipped defaults dir has the README marker file."""
        d = shipped_defaults_dir()
        readme = d / "README.yaml"
        assert readme.is_file(), f"README.yaml not found in {d}"


# ---------------------------------------------------------------------------
# deep_merge()
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_scalar_override(self):
        """Override replaces scalar values."""
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        assert deep_merge(base, override) == {"a": 1, "b": 99}

    def test_nested_dict_merge(self):
        """Nested dicts are merged recursively."""
        base = {"x": {"a": 1, "b": 2}}
        override = {"x": {"b": 99, "c": 3}}
        result = deep_merge(base, override)
        assert result == {"x": {"a": 1, "b": 99, "c": 3}}

    def test_list_replacement(self):
        """Lists are replaced entirely (not appended)."""
        base = {"items": [1, 2, 3]}
        override = {"items": [99]}
        assert deep_merge(base, override) == {"items": [99]}

    def test_new_keys_added(self):
        """Keys present only in override are added."""
        base = {"a": 1}
        override = {"b": 2}
        assert deep_merge(base, override) == {"a": 1, "b": 2}

    def test_base_keys_preserved(self):
        """Keys present only in base are preserved."""
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        assert deep_merge(base, override) == {"a": 1, "b": 99}

    def test_neither_input_mutated(self):
        """Neither base nor override dict is mutated."""
        base = {"x": {"a": 1}}
        override = {"x": {"b": 2}}
        base_copy = {"x": {"a": 1}}
        override_copy = {"x": {"b": 2}}
        deep_merge(base, override)
        assert base == base_copy
        assert override == override_copy

    def test_empty_base(self):
        """Empty base → result is the override."""
        assert deep_merge({}, {"a": 1}) == {"a": 1}

    def test_empty_override(self):
        """Empty override → result is the base."""
        assert deep_merge({"a": 1}, {}) == {"a": 1}

    def test_both_empty(self):
        """Both empty → empty dict."""
        assert deep_merge({}, {}) == {}

    def test_deeply_nested(self):
        """Three levels of nesting are merged correctly."""
        base = {"l1": {"l2": {"l3": "base_val", "only_base": True}}}
        override = {"l1": {"l2": {"l3": "override_val", "only_override": True}}}
        result = deep_merge(base, override)
        assert result == {
            "l1": {"l2": {"l3": "override_val", "only_base": True, "only_override": True}}
        }

    def test_override_dict_with_scalar(self):
        """Scalar in override replaces dict in base."""
        base = {"x": {"nested": True}}
        override = {"x": "now_a_string"}
        assert deep_merge(base, override) == {"x": "now_a_string"}

    def test_override_scalar_with_dict(self):
        """Dict in override replaces scalar in base."""
        base = {"x": "a_string"}
        override = {"x": {"nested": True}}
        assert deep_merge(base, override) == {"x": {"nested": True}}


# ---------------------------------------------------------------------------
# load_yaml_config()
# ---------------------------------------------------------------------------


class TestLoadYamlConfig:
    def test_loads_from_shipped_defaults(self):
        """load_yaml_config loads from shipped defaults directory."""
        data = load_yaml_config("README.yaml")
        assert data.get("_readme") is True

    def test_missing_file_returns_empty_dict(self):
        """Non-existent config name returns empty dict (not an error)."""
        data = load_yaml_config("nonexistent_file_that_does_not_exist.yaml")
        assert data == {}

    def test_user_override_merges_over_shipped(self, monkeypatch, tmp_path):
        """User config merges over shipped defaults."""
        # Point config dir to tmp
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        user_dir = tmp_path / "concierge"
        user_dir.mkdir(parents=True)

        # Create a user override for README.yaml
        user_file = user_dir / "README.yaml"
        user_file.write_text(yaml.dump({"user_key": "user_value"}))

        data = load_yaml_config("README.yaml")
        # Shipped default key preserved
        assert data.get("_readme") is True
        # User key merged in
        assert data.get("user_key") == "user_value"

    def test_user_override_wins_for_same_key(self, monkeypatch, tmp_path):
        """User value overrides shipped value for the same key."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        user_dir = tmp_path / "concierge"
        user_dir.mkdir(parents=True)

        # Override _readme key
        user_file = user_dir / "README.yaml"
        user_file.write_text(yaml.dump({"_readme": False}))

        data = load_yaml_config("README.yaml")
        assert data["_readme"] is False

    def test_env_var_override_highest_priority(self, monkeypatch, tmp_path):
        """Environment variable override has highest priority."""
        # Create env override file
        env_file = tmp_path / "env_override.yaml"
        env_file.write_text(yaml.dump({"_readme": "from_env", "env_only": 42}))

        monkeypatch.setenv("CONCIERGE_README_PATH", str(env_file))
        # Also ensure XDG doesn't interfere
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg_empty"))

        data = load_yaml_config("README.yaml")
        assert data["_readme"] == "from_env"
        assert data["env_only"] == 42

    def test_env_var_name_derivation(self):
        """Verify env var name derivation from config filename."""
        from agentic_concierge.config.external import _env_var_name
        assert _env_var_name("model_profiles.yaml") == "CONCIERGE_MODEL_PROFILES_PATH"
        assert _env_var_name("backends.yaml") == "CONCIERGE_BACKENDS_PATH"
        assert _env_var_name("models.yml") == "CONCIERGE_MODELS_PATH"

    def test_invalid_yaml_returns_empty_dict(self, monkeypatch, tmp_path):
        """Invalid YAML file returns empty dict (not an error)."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        user_dir = tmp_path / "concierge"
        user_dir.mkdir(parents=True)

        bad_file = user_dir / "bad_config.yaml"
        bad_file.write_text(": invalid: yaml: [[[")

        data = load_yaml_config("bad_config.yaml")
        assert data == {}

    def test_non_dict_yaml_returns_empty_dict(self, monkeypatch, tmp_path):
        """YAML file containing a list (not a dict) returns empty dict."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        user_dir = tmp_path / "concierge"
        user_dir.mkdir(parents=True)

        list_file = user_dir / "list_config.yaml"
        list_file.write_text("- item1\n- item2\n")

        data = load_yaml_config("list_config.yaml")
        assert data == {}

    def test_empty_yaml_file_returns_empty_dict(self, monkeypatch, tmp_path):
        """Empty YAML file returns empty dict."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        user_dir = tmp_path / "concierge"
        user_dir.mkdir(parents=True)

        empty_file = user_dir / "empty.yaml"
        empty_file.write_text("")

        data = load_yaml_config("empty.yaml")
        assert data == {}

    def test_full_hierarchy_merge_order(self, monkeypatch, tmp_path):
        """All three sources merge in correct priority order."""
        # Create shipped-like default via a custom test config
        # (We use the real shipped README.yaml as base)

        # Create user override
        user_dir = tmp_path / "xdg" / "concierge"
        user_dir.mkdir(parents=True)
        user_file = user_dir / "README.yaml"
        user_file.write_text(yaml.dump({"user_layer": True, "_readme": "user"}))

        # Create env override
        env_file = tmp_path / "env_override.yaml"
        env_file.write_text(yaml.dump({"env_layer": True, "_readme": "env"}))

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        monkeypatch.setenv("CONCIERGE_README_PATH", str(env_file))

        data = load_yaml_config("README.yaml")

        # env wins for _readme
        assert data["_readme"] == "env"
        # user layer is preserved
        assert data["user_layer"] is True
        # env layer is present
        assert data["env_layer"] is True


# ---------------------------------------------------------------------------
# load_all_yaml_configs()
# ---------------------------------------------------------------------------


class TestLoadAllYamlConfigs:
    def test_loads_from_shipped_directory(self, monkeypatch, tmp_path):
        """load_all_yaml_configs loads all YAML files from a shipped subdirectory."""
        # Create a shipped-like directory structure
        shipped_dir = tmp_path / "shipped" / "templates"
        shipped_dir.mkdir(parents=True)

        (shipped_dir / "alpha.yaml").write_text(yaml.dump({"id": "alpha", "tools": ["a"]}))
        (shipped_dir / "beta.yaml").write_text(yaml.dump({"id": "beta", "tools": ["b"]}))

        # Patch shipped_defaults_dir to point to our test directory
        monkeypatch.setattr(
            "agentic_concierge.config.external.shipped_defaults_dir",
            lambda: tmp_path / "shipped",
        )
        # Ensure no user dir interference
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "no_user"))

        result = load_all_yaml_configs("templates")
        assert "alpha" in result
        assert "beta" in result
        assert result["alpha"]["id"] == "alpha"
        assert result["beta"]["tools"] == ["b"]

    def test_user_overrides_shipped(self, monkeypatch, tmp_path):
        """User templates override shipped templates of the same name."""
        shipped_dir = tmp_path / "shipped" / "templates"
        shipped_dir.mkdir(parents=True)
        (shipped_dir / "alpha.yaml").write_text(
            yaml.dump({"id": "alpha", "tools": ["a"], "shipped_only": True})
        )

        user_dir = tmp_path / "xdg" / "concierge" / "templates"
        user_dir.mkdir(parents=True)
        (user_dir / "alpha.yaml").write_text(
            yaml.dump({"tools": ["a_override"]})
        )

        monkeypatch.setattr(
            "agentic_concierge.config.external.shipped_defaults_dir",
            lambda: tmp_path / "shipped",
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

        result = load_all_yaml_configs("templates")
        assert result["alpha"]["tools"] == ["a_override"]  # user wins (list replaced)
        assert result["alpha"]["shipped_only"] is True  # shipped-only key preserved
        assert result["alpha"]["id"] == "alpha"  # shipped-only key preserved

    def test_user_adds_new_templates(self, monkeypatch, tmp_path):
        """User can add templates not present in shipped defaults."""
        shipped_dir = tmp_path / "shipped" / "templates"
        shipped_dir.mkdir(parents=True)
        (shipped_dir / "alpha.yaml").write_text(yaml.dump({"id": "alpha"}))

        user_dir = tmp_path / "xdg" / "concierge" / "templates"
        user_dir.mkdir(parents=True)
        (user_dir / "custom.yaml").write_text(yaml.dump({"id": "custom", "tools": ["x"]}))

        monkeypatch.setattr(
            "agentic_concierge.config.external.shipped_defaults_dir",
            lambda: tmp_path / "shipped",
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

        result = load_all_yaml_configs("templates")
        assert "alpha" in result
        assert "custom" in result
        assert result["custom"]["id"] == "custom"

    def test_empty_directory_returns_empty_dict(self, monkeypatch, tmp_path):
        """Non-existent directory returns empty dict."""
        monkeypatch.setattr(
            "agentic_concierge.config.external.shipped_defaults_dir",
            lambda: tmp_path / "shipped",
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "no_user"))

        result = load_all_yaml_configs("nonexistent_subdir")
        assert result == {}

    def test_ignores_non_yaml_files(self, monkeypatch, tmp_path):
        """Non-YAML files in the directory are ignored."""
        shipped_dir = tmp_path / "shipped" / "templates"
        shipped_dir.mkdir(parents=True)
        (shipped_dir / "valid.yaml").write_text(yaml.dump({"id": "valid"}))
        (shipped_dir / "readme.txt").write_text("not a yaml config")
        (shipped_dir / "data.json").write_text('{"not": "yaml"}')

        monkeypatch.setattr(
            "agentic_concierge.config.external.shipped_defaults_dir",
            lambda: tmp_path / "shipped",
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "no_user"))

        result = load_all_yaml_configs("templates")
        assert list(result.keys()) == ["valid"]

    def test_yml_extension_also_loaded(self, monkeypatch, tmp_path):
        """Files with .yml extension are also loaded."""
        shipped_dir = tmp_path / "shipped" / "templates"
        shipped_dir.mkdir(parents=True)
        (shipped_dir / "yaml_ext.yaml").write_text(yaml.dump({"ext": "yaml"}))
        (shipped_dir / "yml_ext.yml").write_text(yaml.dump({"ext": "yml"}))

        monkeypatch.setattr(
            "agentic_concierge.config.external.shipped_defaults_dir",
            lambda: tmp_path / "shipped",
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "no_user"))

        result = load_all_yaml_configs("templates")
        assert "yaml_ext" in result
        assert "yml_ext" in result


# ---------------------------------------------------------------------------
# Integration: round-trip YAML → load → verify
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_complex_yaml_roundtrip(self, monkeypatch, tmp_path):
        """Complex YAML with nested dicts, lists, and multiple types loads correctly."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        user_dir = tmp_path / "concierge"
        user_dir.mkdir(parents=True)

        complex_config = {
            "version": 1,
            "profiles": {
                "qwen2.5": {
                    "supports_tool_calling": True,
                    "capabilities": {
                        "code_python": 0.7,
                        "reasoning": 0.7,
                    },
                },
                "sqlcoder": {
                    "supports_tool_calling": False,
                    "capabilities": {
                        "code_sql": 0.9,
                    },
                },
            },
            "tags": ["local", "gpu"],
        }

        config_file = user_dir / "test_complex.yaml"
        config_file.write_text(yaml.dump(complex_config, default_flow_style=False))

        loaded = load_yaml_config("test_complex.yaml")
        assert loaded["version"] == 1
        assert loaded["profiles"]["qwen2.5"]["supports_tool_calling"] is True
        assert loaded["profiles"]["qwen2.5"]["capabilities"]["code_python"] == 0.7
        assert loaded["profiles"]["sqlcoder"]["supports_tool_calling"] is False
        assert loaded["tags"] == ["local", "gpu"]
