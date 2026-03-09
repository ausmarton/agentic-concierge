"""Externalized configuration loader with hierarchy resolution.

Resolution order (highest priority wins):

1. Environment variable override (CONCIERGE_<NAME>_PATH → path to YAML file)
2. User config directory (~/.config/concierge/ or XDG_CONFIG_HOME/concierge/)
3. Shipped defaults (bundled in this package at config/defaults/)

When both a user file and a shipped default exist for the same config name,
the user file is deep-merged *over* the shipped default: user values override
shipped values for scalars; nested dicts are merged recursively; lists are
replaced entirely (not appended).

Usage::

    from agentic_concierge.config.external import load_yaml_config

    # Load model profiles (shipped default + user overrides merged)
    profiles = load_yaml_config("model_profiles.yaml")

    # Load with explicit env var override
    # CONCIERGE_MODEL_PROFILES_PATH=/tmp/my_profiles.yaml
    profiles = load_yaml_config("model_profiles.yaml")
"""

from __future__ import annotations

import importlib.resources
import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


def config_dir() -> Path:
    """User configuration directory.

    Respects ``XDG_CONFIG_HOME`` (Linux/macOS convention).
    Falls back to ``~/.config/concierge/``.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "concierge"
    return Path.home() / ".config" / "concierge"


def shipped_defaults_dir() -> Path:
    """Path to shipped default configs within the installed package.

    Uses ``importlib.resources`` so it works for installed packages,
    editable installs, and zip-imported packages alike.
    """
    ref = importlib.resources.files("agentic_concierge.config.defaults")
    # importlib.resources.files returns a Traversable; for our use
    # (reading YAML files) we need a real Path.  On CPython with
    # filesystem packages this is already a PosixPath.
    if isinstance(ref, Path):
        return ref
    # Fallback for non-filesystem packages (rare): use as_file context
    # manager is impractical here, so derive the path.
    return Path(str(ref))


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge *override* into *base*, returning a new dict.

    - Scalar values in *override* replace those in *base*.
    - Nested dicts are merged recursively.
    - Lists are **replaced** (not appended) — this is intentional so that
      a user can fully redefine a list without accumulating entries.
    - Keys present only in *base* are preserved.
    - Keys present only in *override* are added.

    Neither *base* nor *override* is mutated.
    """
    merged: Dict[str, Any] = {}
    all_keys = set(base) | set(override)
    for key in all_keys:
        if key in override and key in base:
            bv, ov = base[key], override[key]
            if isinstance(bv, dict) and isinstance(ov, dict):
                merged[key] = deep_merge(bv, ov)
            else:
                merged[key] = ov  # override wins (scalars and lists)
        elif key in override:
            merged[key] = override[key]
        else:
            merged[key] = base[key]
    return merged


# ---------------------------------------------------------------------------
# YAML loading with hierarchy
# ---------------------------------------------------------------------------


def _env_var_name(config_name: str) -> str:
    """Derive an environment variable name from a config filename.

    ``"model_profiles.yaml"`` → ``"CONCIERGE_MODEL_PROFILES_PATH"``
    """
    stem = config_name.rsplit(".", 1)[0]  # strip extension
    return f"CONCIERGE_{stem.upper()}_PATH"


def _read_yaml(path: Path) -> Dict[str, Any]:
    """Read a YAML file and return its contents as a dict.

    Returns an empty dict if the file does not exist or is empty/invalid.
    """
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            logger.warning("Config file %s did not contain a YAML mapping; ignoring", path)
            return {}
        return data
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse YAML config %s: %s", path, exc)
        return {}
    except OSError as exc:
        logger.warning("Failed to read config file %s: %s", path, exc)
        return {}


def load_yaml_config(name: str) -> Dict[str, Any]:
    """Load an externalized YAML config file with hierarchy resolution.

    Args:
        name: Config file name relative to the config directory,
              e.g. ``"model_profiles.yaml"`` or ``"templates/engineering.yaml"``.

    Returns:
        Merged configuration dict.  Empty dict if no config files are found.

    Resolution order:

    1. **Environment variable** — ``CONCIERGE_<NAME>_PATH`` points to an
       explicit file path.  When set and the file exists, it is loaded as
       the sole override (merged over shipped defaults).
    2. **User config directory** — ``~/.config/concierge/<name>``.
    3. **Shipped defaults** — bundled in this package.

    When multiple sources exist, they are deep-merged (user over shipped).
    """
    # --- Shipped defaults (lowest priority) ---
    shipped_path = shipped_defaults_dir() / name
    shipped = _read_yaml(shipped_path)

    # --- User config (middle priority) ---
    user_path = config_dir() / name
    user = _read_yaml(user_path)

    # --- Environment variable override (highest priority) ---
    env_name = _env_var_name(name)
    env_path_str = os.environ.get(env_name)
    env: Dict[str, Any] = {}
    if env_path_str:
        env_path = Path(env_path_str)
        env = _read_yaml(env_path)
        if env:
            logger.debug("Loaded config override from %s=%s", env_name, env_path)

    # --- Merge: shipped → user → env ---
    result = shipped
    if user:
        result = deep_merge(result, user)
    if env:
        result = deep_merge(result, env)

    return result


def load_all_yaml_configs(directory_name: str) -> Dict[str, Dict[str, Any]]:
    """Load all YAML files from a subdirectory, merged across hierarchy.

    Used for loading templates from ``templates/`` where each file is a
    separate config entity.

    Args:
        directory_name: Subdirectory name relative to the config directory,
                        e.g. ``"templates"``.

    Returns:
        Dict mapping filename stems to their merged config dicts.
        Example: ``{"engineering": {...}, "research": {...}}``
    """
    result: Dict[str, Dict[str, Any]] = {}

    # Collect from shipped defaults
    shipped_dir = shipped_defaults_dir() / directory_name
    if shipped_dir.is_dir():
        for f in sorted(shipped_dir.iterdir()):
            if f.suffix in (".yaml", ".yml") and f.is_file():
                data = _read_yaml(f)
                if data:
                    result[f.stem] = data

    # Overlay from user config
    user_dir = config_dir() / directory_name
    if user_dir.is_dir():
        for f in sorted(user_dir.iterdir()):
            if f.suffix in (".yaml", ".yml") and f.is_file():
                data = _read_yaml(f)
                if data:
                    if f.stem in result:
                        result[f.stem] = deep_merge(result[f.stem], data)
                    else:
                        result[f.stem] = data

    return result
