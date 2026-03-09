"""Model advisor: recommend profile tier and models based on system resources.

Given a ``SystemProbe`` snapshot, ``advise_profile()`` determines the
``ProfileTier`` and returns a ``SystemProfile`` with the recommended models
for that tier.  All recommendations use models known to support native tool
calling.

Model recommendations and the KV-cache memory estimate are loaded from
``config/defaults/models.yaml`` (shipped defaults) and can be overridden
via ``~/.config/concierge/models.yaml`` or ``CONCIERGE_MODELS_PATH``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

from agentic_concierge.config.features import ProfileTier

if TYPE_CHECKING:
    from agentic_concierge.bootstrap.system_probe import SystemProbe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# YAML-backed model table (lazy-loaded)
# ---------------------------------------------------------------------------

# Hardcoded fallback — used only when YAML cannot be loaded.
_FALLBACK_MODEL_TABLE: dict[ProfileTier, dict[str, str]] = {
    ProfileTier.NANO:   {"routing": "qwen2.5:0.5b", "fast": "qwen2.5:3b",  "quality": "phi3:mini"},
    ProfileTier.SMALL:  {"routing": "qwen2.5:0.5b", "fast": "qwen2.5:7b",  "quality": "qwen2.5:7b"},
    ProfileTier.MEDIUM: {"routing": "qwen2.5:0.5b", "fast": "qwen2.5:7b",  "quality": "qwen2.5:14b"},
    ProfileTier.LARGE:  {"routing": "qwen2.5:0.5b", "fast": "qwen2.5:14b", "quality": "qwen2.5:32b"},
    ProfileTier.SERVER: {"routing": "qwen2.5:0.5b", "fast": "qwen2.5:32b", "quality": "qwen2.5:72b"},
}

_FALLBACK_MODEL_CTX_MB = 2048

# Module-level cache — set by _load_models_config() on first access.
_cached_model_table: Optional[dict[ProfileTier, dict[str, str]]] = None
_cached_model_ctx_mb: Optional[int] = None


def _load_models_config() -> Dict:
    """Load models.yaml via the config hierarchy; return raw dict."""
    from agentic_concierge.config.external import load_yaml_config

    return load_yaml_config("models.yaml")


def _parse_model_table(raw: Dict) -> dict[ProfileTier, dict[str, str]]:
    """Parse ``tier_models`` section from raw YAML into a ProfileTier-keyed dict."""
    tier_models = raw.get("tier_models")
    if not isinstance(tier_models, dict):
        return dict(_FALLBACK_MODEL_TABLE)

    tier_name_to_enum = {t.name.lower(): t for t in ProfileTier}
    table: dict[ProfileTier, dict[str, str]] = {}

    for tier_name, models in tier_models.items():
        tier = tier_name_to_enum.get(tier_name.lower())
        if tier is None:
            logger.warning("Unknown tier %r in models.yaml; skipping", tier_name)
            continue
        if not isinstance(models, dict):
            logger.warning("Tier %r value must be a mapping; skipping", tier_name)
            continue
        table[tier] = {
            "routing": str(models.get("routing", "")),
            "fast": str(models.get("fast", "")),
            "quality": str(models.get("quality", "")),
        }

    # Fill any missing tiers from fallback.
    for tier in ProfileTier:
        if tier not in table:
            table[tier] = _FALLBACK_MODEL_TABLE[tier]

    return table


def _loaded_model_table() -> dict[ProfileTier, dict[str, str]]:
    """Return the model table, loading from YAML on first access."""
    global _cached_model_table
    if _cached_model_table is None:
        try:
            raw = _load_models_config()
            _cached_model_table = _parse_model_table(raw)
        except Exception:
            logger.debug("Failed to load models.yaml; using fallback", exc_info=True)
            _cached_model_table = dict(_FALLBACK_MODEL_TABLE)
    return _cached_model_table


def _loaded_model_ctx_mb() -> int:
    """Return model_ctx_mb, loading from YAML on first access."""
    global _cached_model_ctx_mb
    if _cached_model_ctx_mb is None:
        try:
            raw = _load_models_config()
            _cached_model_ctx_mb = int(raw.get("model_ctx_mb", _FALLBACK_MODEL_CTX_MB))
        except Exception:
            _cached_model_ctx_mb = _FALLBACK_MODEL_CTX_MB
    return _cached_model_ctx_mb


def reload_models_config() -> None:
    """Clear cached model config — forces re-load on next access.

    Intended for tests and live config reloading.
    """
    global _cached_model_table, _cached_model_ctx_mb
    _cached_model_table = None
    _cached_model_ctx_mb = None


@dataclass
class SystemProfile:
    """Recommended configuration derived from the detected hardware."""

    tier: ProfileTier
    routing_model: str
    fast_model: str
    quality_model: str
    max_concurrent_agents: int
    ram_total_mb: int
    ram_available_mb: int
    total_vram_mb: int
    cpu_cores: int
    cpu_arch: str
    gpu_count: int


def advise_profile(probe: "SystemProbe") -> SystemProfile:
    """Derive the best ``SystemProfile`` for the detected hardware.

    Tier thresholds:

    ========  ============================================
    nano      RAM < 8 GB (regardless of GPU)
    small     8–16 GB RAM, total VRAM < 4 GB
    medium    16–32 GB RAM OR 4–12 GB VRAM
    large     32–64 GB RAM OR 12–24 GB VRAM
    server    64+ GB RAM OR 24+ GB VRAM OR 2+ GPUs
    ========  ============================================
    """
    total_vram_mb = probe.total_vram_mb
    vram_gb = total_vram_mb / 1024
    gpu_count = len(probe.gpu_devices)

    if gpu_count >= 2 or probe.ram_total_mb >= 64 * 1024 or vram_gb >= 24:
        tier = ProfileTier.SERVER
    elif probe.ram_total_mb >= 32 * 1024 or vram_gb >= 12:
        tier = ProfileTier.LARGE
    elif probe.ram_total_mb >= 16 * 1024 or vram_gb >= 4:
        tier = ProfileTier.MEDIUM
    elif probe.ram_total_mb >= 8 * 1024:
        tier = ProfileTier.SMALL
    else:
        tier = ProfileTier.NANO

    models = _loaded_model_table()[tier]

    # max_concurrent_agents: leave 15% + 512 MB overhead; each agent needs ~ctx MB
    model_ctx_mb = _loaded_model_ctx_mb()
    overhead_mb = max(2048, probe.ram_total_mb * 0.15) + 512
    usable_mb = probe.ram_available_mb - overhead_mb
    max_concurrent = max(1, min(math.floor(usable_mb / model_ctx_mb), probe.cpu_cores - 1))

    return SystemProfile(
        tier=tier,
        routing_model=models["routing"],
        fast_model=models["fast"],
        quality_model=models["quality"],
        max_concurrent_agents=max_concurrent,
        ram_total_mb=probe.ram_total_mb,
        ram_available_mb=probe.ram_available_mb,
        total_vram_mb=total_vram_mb,
        cpu_cores=probe.cpu_cores,
        cpu_arch=probe.cpu_arch,
        gpu_count=gpu_count,
    )
