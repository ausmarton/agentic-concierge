"""Tests for platform detection and hardware-adaptive configuration.

Verifies:
- GPU detection via lspci parsing (NVIDIA, AMD, Apple Silicon, Intel, unknown)
- AMD architecture lookup from PCI device IDs
- GPU info → platform profile mapping
- Platform profile loading from shipped YAML defaults
- User YAML override of platform selection
- CONCIERGE_PLATFORM env var override
- Custom user-defined profiles
- apply_backend_env() behaviour
- Cache invalidation via reload_profiles()
- Error resilience (malformed YAML, unknown profiles)
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agentic_concierge.config.platform import (
    GPUInfo,
    PlatformProfile,
    _amd_arch_from_device_id,
    _detect_gpu_lspci,
    _gpu_to_profile_name,
    apply_backend_env,
    detect_gpu,
    detect_platform,
    platform_profiles,
    reload_profiles,
)


@pytest.fixture(autouse=True)
def _clean_caches():
    """Ensure each test starts and ends with clean caches and env vars."""
    reload_profiles()
    yield
    os.environ.pop("CONCIERGE_PLATFORM", None)
    os.environ.pop("CONCIERGE_PLATFORM_PATH", None)
    os.environ.pop("XDG_CONFIG_HOME", None)
    os.environ.pop("OLLAMA_VULKAN", None)
    os.environ.pop("HIP_VISIBLE_DEVICES", None)
    reload_profiles()


# ---------------------------------------------------------------------------
# GPUInfo and PlatformProfile dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:

    def test_gpu_info_defaults(self):
        gpu = GPUInfo()
        assert gpu.vendor == "unknown"
        assert gpu.name == ""
        assert gpu.arch == ""
        assert gpu.pci_id == ""

    def test_gpu_info_populated(self):
        gpu = GPUInfo(vendor="amd", name="Radeon 8060S", arch="RDNA3.5", pci_id="1002:150e")
        assert gpu.vendor == "amd"
        assert gpu.arch == "RDNA3.5"

    def test_platform_profile_defaults(self):
        p = PlatformProfile(name="test")
        assert p.backend_env == {}
        assert p.preferred_backends == ["ollama"]
        assert p.notes == ""

    def test_platform_profile_populated(self):
        p = PlatformProfile(
            name="strix-halo",
            backend_env={"OLLAMA_VULKAN": "1"},
            preferred_backends=["ollama", "llama_cpp"],
            notes="Vulkan required",
        )
        assert p.backend_env["OLLAMA_VULKAN"] == "1"
        assert p.preferred_backends == ["ollama", "llama_cpp"]


# ---------------------------------------------------------------------------
# AMD architecture lookup
# ---------------------------------------------------------------------------


class TestAMDArchLookup:

    def test_rdna35_device_id(self):
        assert _amd_arch_from_device_id("150e") == "RDNA3.5"

    def test_rdna35_uppercase(self):
        assert _amd_arch_from_device_id("150E") == "RDNA3.5"

    def test_rdna3_device_id(self):
        assert _amd_arch_from_device_id("744c") == "RDNA3"

    def test_rdna2_device_id(self):
        assert _amd_arch_from_device_id("73df") == "RDNA2"

    def test_unknown_device_id(self):
        assert _amd_arch_from_device_id("9999") == ""


# ---------------------------------------------------------------------------
# GPU detection — lspci parsing
# ---------------------------------------------------------------------------


LSPCI_NVIDIA = """\
00:02.0 VGA compatible controller: Intel Corporation HD Graphics 630 [8086:5912] (rev 04)
01:00.0 3D controller: NVIDIA Corporation GA102GL [RTX A6000] [10de:2230] (rev a1)
"""

LSPCI_AMD_STRIX = """\
06:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Phoenix3 [1002:150e] (rev c8)
"""

LSPCI_AMD_RDNA3 = """\
03:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XT/7900 XTX] [1002:744c] (rev c8)
"""

LSPCI_INTEL_ONLY = """\
00:02.0 VGA compatible controller: Intel Corporation UHD Graphics 630 [8086:3e92] (rev 00)
"""

LSPCI_NO_GPU = """\
00:00.0 Host bridge: Intel Corporation 8th Gen Core Processor Host Bridge/DRAM Registers [8086:3ec2] (rev 08)
00:1f.0 ISA bridge: Intel Corporation C246 Chipset LPC/eSPI [8086:a2c8] (rev 10)
"""


class TestDetectGpuLspci:

    def _mock_lspci(self, output, returncode=0):
        return MagicMock(returncode=returncode, stdout=output)

    def test_nvidia_detected(self):
        with patch("subprocess.run", return_value=self._mock_lspci(LSPCI_NVIDIA)):
            gpu = _detect_gpu_lspci()
        assert gpu.vendor == "nvidia"
        assert "RTX A6000" in gpu.name
        assert gpu.pci_id == "10de:2230"

    def test_amd_strix_halo_detected(self):
        with patch("subprocess.run", return_value=self._mock_lspci(LSPCI_AMD_STRIX)):
            gpu = _detect_gpu_lspci()
        assert gpu.vendor == "amd"
        assert gpu.arch == "RDNA3.5"
        assert gpu.pci_id == "1002:150e"

    def test_amd_rdna3_detected(self):
        with patch("subprocess.run", return_value=self._mock_lspci(LSPCI_AMD_RDNA3)):
            gpu = _detect_gpu_lspci()
        assert gpu.vendor == "amd"
        assert gpu.arch == "RDNA3"
        assert "Navi 31" in gpu.name

    def test_intel_detected(self):
        with patch("subprocess.run", return_value=self._mock_lspci(LSPCI_INTEL_ONLY)):
            gpu = _detect_gpu_lspci()
        assert gpu.vendor == "intel"
        assert gpu.pci_id.startswith("8086:")

    def test_no_gpu_returns_unknown(self):
        with patch("subprocess.run", return_value=self._mock_lspci(LSPCI_NO_GPU)):
            gpu = _detect_gpu_lspci()
        assert gpu.vendor == "unknown"

    def test_lspci_not_found_returns_unknown(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            gpu = _detect_gpu_lspci()
        assert gpu.vendor == "unknown"

    def test_lspci_timeout_returns_unknown(self):
        import subprocess as sp
        with patch("subprocess.run", side_effect=sp.TimeoutExpired(cmd="lspci", timeout=5)):
            gpu = _detect_gpu_lspci()
        assert gpu.vendor == "unknown"

    def test_lspci_nonzero_return_code(self):
        with patch("subprocess.run", return_value=self._mock_lspci("", returncode=1)):
            gpu = _detect_gpu_lspci()
        assert gpu.vendor == "unknown"


# ---------------------------------------------------------------------------
# detect_gpu() top-level dispatcher
# ---------------------------------------------------------------------------


class TestDetectGpu:

    def test_apple_silicon_detected(self):
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
        ):
            gpu = detect_gpu()
        assert gpu.vendor == "apple"
        assert gpu.arch == "Apple Silicon"

    def test_linux_delegates_to_lspci(self):
        with (
            patch("platform.system", return_value="Linux"),
            patch(
                "agentic_concierge.config.platform._detect_gpu_lspci",
                return_value=GPUInfo(vendor="nvidia", name="RTX 4090"),
            ),
        ):
            gpu = detect_gpu()
        assert gpu.vendor == "nvidia"

    def test_windows_returns_unknown(self):
        with patch("platform.system", return_value="Windows"):
            gpu = detect_gpu()
        assert gpu.vendor == "unknown"


# ---------------------------------------------------------------------------
# GPU → profile name mapping
# ---------------------------------------------------------------------------


class TestGpuToProfileName:

    def test_apple(self):
        assert _gpu_to_profile_name(GPUInfo(vendor="apple")) == "apple-silicon"

    def test_nvidia(self):
        assert _gpu_to_profile_name(GPUInfo(vendor="nvidia")) == "nvidia-cuda"

    def test_amd_rdna35(self):
        assert _gpu_to_profile_name(GPUInfo(vendor="amd", arch="RDNA3.5")) == "strix-halo"

    def test_amd_rdna3(self):
        assert _gpu_to_profile_name(GPUInfo(vendor="amd", arch="RDNA3")) == "generic"

    def test_amd_no_arch(self):
        assert _gpu_to_profile_name(GPUInfo(vendor="amd")) == "generic"

    def test_intel(self):
        assert _gpu_to_profile_name(GPUInfo(vendor="intel")) == "generic"

    def test_unknown(self):
        assert _gpu_to_profile_name(GPUInfo()) == "generic"


# ---------------------------------------------------------------------------
# Shipped YAML profiles
# ---------------------------------------------------------------------------


class TestShippedProfiles:

    def test_four_profiles_loaded(self):
        profiles = platform_profiles()
        assert set(profiles.keys()) == {"strix-halo", "apple-silicon", "nvidia-cuda", "generic"}

    def test_all_profiles_are_platform_profile_instances(self):
        for p in platform_profiles().values():
            assert isinstance(p, PlatformProfile)

    def test_strix_halo_env_vars(self):
        p = platform_profiles()["strix-halo"]
        assert p.backend_env["OLLAMA_VULKAN"] == "1"
        assert p.backend_env["HIP_VISIBLE_DEVICES"] == "-1"

    def test_strix_halo_backends(self):
        p = platform_profiles()["strix-halo"]
        assert "ollama" in p.preferred_backends

    def test_apple_silicon_no_env_vars(self):
        p = platform_profiles()["apple-silicon"]
        assert p.backend_env == {}

    def test_nvidia_backends(self):
        p = platform_profiles()["nvidia-cuda"]
        assert "vllm" in p.preferred_backends

    def test_generic_backends(self):
        p = platform_profiles()["generic"]
        assert p.preferred_backends == ["ollama"]


# ---------------------------------------------------------------------------
# detect_platform() — priority resolution
# ---------------------------------------------------------------------------


class TestDetectPlatform:

    def test_auto_detect_nvidia(self):
        gpu = GPUInfo(vendor="nvidia", name="RTX 4090")
        profile = detect_platform(gpu=gpu)
        assert profile.name == "nvidia-cuda"

    def test_auto_detect_strix_halo(self):
        gpu = GPUInfo(vendor="amd", arch="RDNA3.5")
        profile = detect_platform(gpu=gpu)
        assert profile.name == "strix-halo"
        assert profile.backend_env["OLLAMA_VULKAN"] == "1"

    def test_auto_detect_apple(self):
        gpu = GPUInfo(vendor="apple")
        profile = detect_platform(gpu=gpu)
        assert profile.name == "apple-silicon"

    def test_auto_detect_unknown(self):
        gpu = GPUInfo()
        profile = detect_platform(gpu=gpu)
        assert profile.name == "generic"

    def test_env_var_overrides_auto_detect(self):
        os.environ["CONCIERGE_PLATFORM"] = "strix-halo"
        # Even with an nvidia GPU, the env var wins
        gpu = GPUInfo(vendor="nvidia")
        profile = detect_platform(gpu=gpu)
        assert profile.name == "strix-halo"

    def test_env_var_unknown_falls_back_to_auto_detect(self):
        os.environ["CONCIERGE_PLATFORM"] = "nonexistent-platform"
        gpu = GPUInfo(vendor="nvidia")
        profile = detect_platform(gpu=gpu)
        assert profile.name == "nvidia-cuda"

    def test_user_yaml_platform_key_overrides(self, tmp_path):
        user_cfg = tmp_path / "concierge"
        user_cfg.mkdir()
        (user_cfg / "platform.yaml").write_text(yaml.dump({"platform": "apple-silicon"}))
        os.environ["XDG_CONFIG_HOME"] = str(tmp_path)
        reload_profiles()

        gpu = GPUInfo(vendor="nvidia")
        profile = detect_platform(gpu=gpu)
        assert profile.name == "apple-silicon"

    def test_env_var_beats_user_yaml(self, tmp_path):
        user_cfg = tmp_path / "concierge"
        user_cfg.mkdir()
        (user_cfg / "platform.yaml").write_text(yaml.dump({"platform": "apple-silicon"}))
        os.environ["XDG_CONFIG_HOME"] = str(tmp_path)
        os.environ["CONCIERGE_PLATFORM"] = "nvidia-cuda"
        reload_profiles()

        gpu = GPUInfo(vendor="amd", arch="RDNA3.5")
        profile = detect_platform(gpu=gpu)
        assert profile.name == "nvidia-cuda"

    def test_calls_detect_gpu_when_no_gpu_provided(self):
        with patch(
            "agentic_concierge.config.platform.detect_gpu",
            return_value=GPUInfo(vendor="nvidia"),
        ) as mock_detect:
            profile = detect_platform()
        mock_detect.assert_called_once()
        assert profile.name == "nvidia-cuda"


# ---------------------------------------------------------------------------
# User override — custom profiles
# ---------------------------------------------------------------------------


class TestUserOverride:

    def test_custom_profile_from_user_yaml(self, tmp_path):
        user_cfg = tmp_path / "concierge"
        user_cfg.mkdir()
        custom = {
            "platform": "my-custom-rig",
            "profiles": {
                "my-custom-rig": {
                    "backend_env": {"MY_BACKEND_VAR": "42"},
                    "preferred_backends": ["vllm"],
                    "notes": "Custom test rig",
                },
            },
        }
        (user_cfg / "platform.yaml").write_text(yaml.dump(custom))
        os.environ["XDG_CONFIG_HOME"] = str(tmp_path)
        reload_profiles()

        profiles = platform_profiles()
        assert "my-custom-rig" in profiles
        assert profiles["my-custom-rig"].backend_env["MY_BACKEND_VAR"] == "42"

        # Platform selection uses the custom profile
        profile = detect_platform(gpu=GPUInfo())
        assert profile.name == "my-custom-rig"

    def test_user_overrides_shipped_profile(self, tmp_path):
        user_cfg = tmp_path / "concierge"
        user_cfg.mkdir()
        override = {
            "profiles": {
                "strix-halo": {
                    "backend_env": {"OLLAMA_VULKAN": "1", "HIP_VISIBLE_DEVICES": "-1", "EXTRA_VAR": "yes"},
                    "preferred_backends": ["ollama"],
                    "notes": "Override with extra var",
                },
            },
        }
        (user_cfg / "platform.yaml").write_text(yaml.dump(override))
        os.environ["XDG_CONFIG_HOME"] = str(tmp_path)
        reload_profiles()

        p = platform_profiles()["strix-halo"]
        assert p.backend_env.get("EXTRA_VAR") == "yes"
        # Original env vars still present (deep merge)
        assert p.backend_env["OLLAMA_VULKAN"] == "1"


# ---------------------------------------------------------------------------
# apply_backend_env
# ---------------------------------------------------------------------------


class TestApplyBackendEnv:

    def test_sets_env_vars(self):
        profile = PlatformProfile(
            name="test",
            backend_env={"OLLAMA_VULKAN": "1", "HIP_VISIBLE_DEVICES": "-1"},
        )
        apply_backend_env(profile)
        assert os.environ["OLLAMA_VULKAN"] == "1"
        assert os.environ["HIP_VISIBLE_DEVICES"] == "-1"

    def test_does_not_overwrite_existing(self):
        os.environ["OLLAMA_VULKAN"] = "0"
        profile = PlatformProfile(
            name="test",
            backend_env={"OLLAMA_VULKAN": "1"},
        )
        apply_backend_env(profile)
        assert os.environ["OLLAMA_VULKAN"] == "0"

    def test_empty_env_is_noop(self):
        profile = PlatformProfile(name="test", backend_env={})
        apply_backend_env(profile)  # should not raise


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------


class TestCacheInvalidation:

    def test_reload_clears_cache(self, tmp_path):
        # Load once
        profiles1 = platform_profiles()
        assert "generic" in profiles1

        # Add custom profile via user config
        user_cfg = tmp_path / "concierge"
        user_cfg.mkdir()
        custom = {
            "profiles": {
                "custom-rig": {
                    "backend_env": {},
                    "preferred_backends": ["ollama"],
                },
            },
        }
        (user_cfg / "platform.yaml").write_text(yaml.dump(custom))
        os.environ["XDG_CONFIG_HOME"] = str(tmp_path)
        reload_profiles()

        profiles2 = platform_profiles()
        assert "custom-rig" in profiles2


# ---------------------------------------------------------------------------
# Error resilience
# ---------------------------------------------------------------------------


class TestErrorResilience:

    def test_malformed_yaml_uses_fallback(self, tmp_path):
        user_cfg = tmp_path / "concierge"
        user_cfg.mkdir()
        (user_cfg / "platform.yaml").write_text("not: [valid: yaml")
        os.environ["XDG_CONFIG_HOME"] = str(tmp_path)
        reload_profiles()

        profiles = platform_profiles()
        # Should still have profiles from shipped defaults
        assert len(profiles) >= 4

    def test_invalid_profile_data_skipped(self, tmp_path):
        user_cfg = tmp_path / "concierge"
        user_cfg.mkdir()
        bad = {
            "profiles": {
                "bad-profile": "not a dict",
                "good-profile": {
                    "backend_env": {},
                    "preferred_backends": ["ollama"],
                },
            },
        }
        (user_cfg / "platform.yaml").write_text(yaml.dump(bad))
        os.environ["XDG_CONFIG_HOME"] = str(tmp_path)
        reload_profiles()

        profiles = platform_profiles()
        assert "bad-profile" not in profiles
        assert "good-profile" in profiles

    def test_unknown_platform_in_env_falls_back(self):
        os.environ["CONCIERGE_PLATFORM"] = "does-not-exist"
        gpu = GPUInfo(vendor="apple")
        profile = detect_platform(gpu=gpu)
        assert profile.name == "apple-silicon"
