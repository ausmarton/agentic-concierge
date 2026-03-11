"""Platform detection and hardware-adaptive backend configuration.

Detects GPU hardware and produces a ``PlatformProfile`` with appropriate
backend environment variables and preferences.  Auto-detection is best-effort
and never raises — unknown hardware returns the ``generic`` profile.

Override auto-detection via:

- ``~/.config/concierge/platform.yaml`` (set ``platform: <profile-name>``)
- ``CONCIERGE_PLATFORM`` environment variable (set to a profile name)
- ``CONCIERGE_PLATFORM_PATH`` environment variable (path to a YAML file)

The user YAML file can also define custom profiles under ``profiles:``.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class GPUInfo:
    """Detected GPU hardware information."""

    vendor: str = "unknown"  # "nvidia", "amd", "apple", "intel", "unknown"
    name: str = ""           # e.g. "Radeon 8060S", "GeForce RTX 4090"
    arch: str = ""           # e.g. "RDNA3.5", "Ada Lovelace", ""
    pci_id: str = ""         # e.g. "1002:150e" (vendor:device)


@dataclass
class PlatformProfile:
    """Hardware-adaptive backend configuration for a platform."""

    name: str                                    # "strix-halo", "apple-silicon", etc.
    backend_env: Dict[str, str] = field(default_factory=dict)
    preferred_backends: list[str] = field(default_factory=lambda: ["ollama"])
    notes: str = ""


# ---------------------------------------------------------------------------
# GPU architecture lookup tables
# ---------------------------------------------------------------------------

# AMD PCI device IDs → architecture.  This is a representative subset;
# full tables are at https://pci-ids.ucw.cz/read/PC/1002
_AMD_ARCH_BY_DEVICE_PREFIX: list[tuple[str, str]] = [
    # RDNA 3.5 (Strix Point / Strix Halo)
    ("150e", "RDNA3.5"),
    ("1502", "RDNA3.5"),
    ("1586", "RDNA3.5"),
    # RDNA 3 (RX 7000 series)
    ("744c", "RDNA3"),
    ("7480", "RDNA3"),
    ("73a5", "RDNA3"),
    ("73bf", "RDNA3"),
    # RDNA 2 (RX 6000 series)
    ("73df", "RDNA2"),
    ("73ff", "RDNA2"),
    ("7340", "RDNA2"),
]


def _amd_arch_from_device_id(device_id: str) -> str:
    """Return AMD GPU architecture from a 4-hex PCI device ID, or ''."""
    device_lower = device_id.lower()
    for prefix, arch in _AMD_ARCH_BY_DEVICE_PREFIX:
        if device_lower == prefix:
            return arch
    return ""


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

# Regex for lspci VGA/3D lines: "xx:xx.x VGA compatible controller: Vendor Device (rev xx)"
_LSPCI_VGA_RE = re.compile(
    r"^[0-9a-f:.]+\s+(?:VGA compatible|3D|Display)\s+controller:\s+(.+)",
    re.IGNORECASE,
)


def detect_gpu() -> GPUInfo:
    """Detect GPU hardware.  Never raises — returns unknown on failure."""
    system = platform.system()

    # --- Apple Silicon ---
    if system == "Darwin" and platform.machine() == "arm64":
        return GPUInfo(vendor="apple", name="Apple Silicon", arch="Apple Silicon")

    # --- Linux: use lspci ---
    if system == "Linux":
        return _detect_gpu_lspci()

    return GPUInfo()


def _detect_gpu_lspci() -> GPUInfo:
    """Parse ``lspci -nn`` output for GPU vendor, device name, and PCI IDs."""
    try:
        result = subprocess.run(
            ["lspci", "-nn"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return GPUInfo()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return GPUInfo()

    # Collect all GPU entries; prefer discrete (NVIDIA/AMD) over integrated (Intel)
    candidates: list[GPUInfo] = []

    for line in result.stdout.splitlines():
        m = _LSPCI_VGA_RE.match(line)
        if not m:
            continue
        desc = m.group(1)

        # Extract PCI IDs from the description: [vendor:device]
        pci_match = re.search(r"\[([0-9a-fA-F]{4}):([0-9a-fA-F]{4})\]", desc)
        vendor_id = pci_match.group(1).lower() if pci_match else ""
        device_id = pci_match.group(2).lower() if pci_match else ""

        # Clean up the device name: remove PCI ID brackets [xxxx:xxxx] but
        # keep friendly-name brackets like [RTX A6000]
        name = re.sub(r"\s*\[[0-9a-fA-F]{4}:[0-9a-fA-F]{4}\]", "", desc).strip()
        # Remove trailing "(rev XX)"
        name = re.sub(r"\s*\(rev\s+[0-9a-fA-F]+\)$", "", name).strip()

        if vendor_id == "10de":
            candidates.append(GPUInfo(vendor="nvidia", name=name, pci_id=f"{vendor_id}:{device_id}"))
        elif vendor_id == "1002":
            arch = _amd_arch_from_device_id(device_id)
            candidates.append(GPUInfo(vendor="amd", name=name, arch=arch, pci_id=f"{vendor_id}:{device_id}"))
        elif vendor_id == "8086":
            candidates.append(GPUInfo(vendor="intel", name=name, pci_id=f"{vendor_id}:{device_id}"))

    if not candidates:
        return GPUInfo()

    # Prefer discrete GPUs: NVIDIA > AMD > Intel > unknown
    priority = {"nvidia": 0, "amd": 1, "intel": 2}
    candidates.sort(key=lambda g: priority.get(g.vendor, 3))
    return candidates[0]


# ---------------------------------------------------------------------------
# Profile loading from YAML
# ---------------------------------------------------------------------------

_cached_profiles: Optional[Dict[str, PlatformProfile]] = None


def _load_profiles() -> Dict[str, PlatformProfile]:
    """Load platform profiles from YAML config hierarchy."""
    global _cached_profiles
    if _cached_profiles is not None:
        return _cached_profiles

    try:
        from agentic_concierge.config.external import load_yaml_config
        raw = load_yaml_config("platform.yaml")
        profiles_raw = raw.get("profiles")
        if isinstance(profiles_raw, dict):
            _cached_profiles = {}
            for name, pdata in profiles_raw.items():
                if not isinstance(pdata, dict):
                    continue
                _cached_profiles[name] = PlatformProfile(
                    name=str(name),
                    backend_env=dict(pdata.get("backend_env", {})),
                    preferred_backends=list(pdata.get("preferred_backends", ["ollama"])),
                    notes=str(pdata.get("notes", "")),
                )
            return _cached_profiles
    except Exception:
        logger.debug("Failed to load platform profiles from YAML; using fallback")

    _cached_profiles = _fallback_profiles()
    return _cached_profiles


def _fallback_profiles() -> Dict[str, PlatformProfile]:
    """Hardcoded fallback profiles when YAML loading fails."""
    return {
        "strix-halo": PlatformProfile(
            name="strix-halo",
            backend_env={"OLLAMA_VULKAN": "1", "HIP_VISIBLE_DEVICES": "-1"},
            preferred_backends=["ollama", "llama_cpp"],
            notes="AMD Strix Halo APU — ROCm/HIP broken on RDNA 3.5; Vulkan required",
        ),
        "apple-silicon": PlatformProfile(
            name="apple-silicon",
            backend_env={},
            preferred_backends=["mlx", "ollama"],
            notes="Apple Silicon — Metal acceleration via MLX or Ollama",
        ),
        "nvidia-cuda": PlatformProfile(
            name="nvidia-cuda",
            backend_env={},
            preferred_backends=["vllm", "ollama", "llama_cpp"],
            notes="NVIDIA GPU — CUDA acceleration via vLLM or Ollama",
        ),
        "generic": PlatformProfile(
            name="generic",
            backend_env={},
            preferred_backends=["ollama"],
            notes="No GPU detected or unknown hardware",
        ),
    }


def reload_profiles() -> None:
    """Clear cached profiles — forces re-load on next access."""
    global _cached_profiles
    _cached_profiles = None


def platform_profiles() -> Dict[str, PlatformProfile]:
    """Return all known platform profiles, loaded from YAML."""
    return dict(_load_profiles())


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------


def _gpu_to_profile_name(gpu: GPUInfo) -> str:
    """Map detected GPU info to a profile name."""
    if gpu.vendor == "apple":
        return "apple-silicon"

    if gpu.vendor == "amd":
        if gpu.arch == "RDNA3.5":
            return "strix-halo"
        # Other AMD GPUs with ROCm support can use generic/ollama
        return "generic"

    if gpu.vendor == "nvidia":
        return "nvidia-cuda"

    return "generic"


def detect_platform(gpu: Optional[GPUInfo] = None) -> PlatformProfile:
    """Detect platform hardware and return the matching profile.

    Resolution order (highest priority wins):

    1. ``CONCIERGE_PLATFORM`` env var (profile name, e.g. ``"strix-halo"``)
    2. User ``platform.yaml`` with a ``platform:`` key
    3. Auto-detection from GPU hardware

    Args:
        gpu: Pre-detected GPU info.  If ``None``, calls ``detect_gpu()``.

    Returns:
        The matching ``PlatformProfile``.  Falls back to ``generic`` if
        the specified or detected profile is not found.
    """
    profiles = _load_profiles()

    # --- Priority 1: CONCIERGE_PLATFORM env var ---
    env_platform = os.environ.get("CONCIERGE_PLATFORM")
    if env_platform:
        profile = profiles.get(env_platform)
        if profile:
            logger.info("Platform override via CONCIERGE_PLATFORM: %s", env_platform)
            return profile
        logger.warning(
            "CONCIERGE_PLATFORM=%s not found in known profiles; falling back to auto-detect",
            env_platform,
        )

    # --- Priority 2: User platform.yaml with 'platform' key ---
    try:
        from agentic_concierge.config.external import load_yaml_config
        raw = load_yaml_config("platform.yaml")
        platform_key = raw.get("platform")
        if isinstance(platform_key, str) and platform_key:
            profile = profiles.get(platform_key)
            if profile:
                logger.info("Platform from config: %s", platform_key)
                return profile
            logger.warning(
                "platform: '%s' in platform.yaml not found in known profiles",
                platform_key,
            )
    except Exception:
        pass

    # --- Priority 3: Auto-detect from GPU ---
    if gpu is None:
        gpu = detect_gpu()

    profile_name = _gpu_to_profile_name(gpu)
    profile = profiles.get(profile_name)
    if profile:
        logger.info("Auto-detected platform: %s (GPU: %s)", profile_name, gpu.name)
        return profile

    # Ultimate fallback
    return profiles.get("generic", PlatformProfile(name="generic"))


def apply_backend_env(profile: PlatformProfile) -> None:
    """Apply platform-specific environment variables.

    Sets each env var from ``profile.backend_env`` into ``os.environ``
    **only if not already set** — explicit user configuration is never
    overwritten.
    """
    for key, value in profile.backend_env.items():
        if key not in os.environ:
            os.environ[key] = value
            logger.debug("Set %s=%s (platform: %s)", key, value, profile.name)
        else:
            logger.debug(
                "Skipped %s=%s — already set to %s",
                key, value, os.environ[key],
            )
