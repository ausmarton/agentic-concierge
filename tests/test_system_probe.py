"""Tests for bootstrap/system_probe.py — all external calls mocked."""

from __future__ import annotations

import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_concierge.bootstrap.system_probe import (
    GPUDevice,
    SystemProbe,
    _probe_amd_sysfs,
    _probe_gpus,
    probe_system,
)


# ---------------------------------------------------------------------------
# GPUDevice
# ---------------------------------------------------------------------------

def test_gpu_device_fields():
    d = GPUDevice(name="NVIDIA A100", vram_mb=40960, vendor="nvidia")
    assert d.name == "NVIDIA A100"
    assert d.vram_mb == 40960
    assert d.vendor == "nvidia"


# ---------------------------------------------------------------------------
# SystemProbe.total_vram_mb
# ---------------------------------------------------------------------------

def test_total_vram_mb_empty():
    probe = SystemProbe(cpu_cores=4, cpu_arch="x86_64", ram_total_mb=16384, ram_available_mb=8000)
    assert probe.total_vram_mb == 0


def test_total_vram_mb_sum():
    probe = SystemProbe(
        cpu_cores=4, cpu_arch="x86_64", ram_total_mb=16384, ram_available_mb=8000,
        gpu_devices=[
            GPUDevice(name="GPU A", vram_mb=8192, vendor="nvidia"),
            GPUDevice(name="GPU B", vram_mb=8192, vendor="nvidia"),
        ],
    )
    assert probe.total_vram_mb == 16384


# ---------------------------------------------------------------------------
# _probe_gpus — NVIDIA
# ---------------------------------------------------------------------------

def test_probe_gpus_nvidia():
    nvidia_output = "NVIDIA GeForce RTX 3090, 24576\n"
    mock_result = MagicMock(returncode=0, stdout=nvidia_output)
    with patch("subprocess.run", return_value=mock_result):
        devices = _probe_gpus(is_apple=False, ram_total_mb=32768)
    assert len(devices) == 1
    assert devices[0].vendor == "nvidia"
    assert devices[0].name == "NVIDIA GeForce RTX 3090"
    assert devices[0].vram_mb == 24576


def test_probe_gpus_nvidia_multi():
    nvidia_output = "GPU A, 8192\nGPU B, 8192\n"
    mock_result = MagicMock(returncode=0, stdout=nvidia_output)
    with patch("subprocess.run", return_value=mock_result):
        devices = _probe_gpus(is_apple=False, ram_total_mb=32768)
    assert len(devices) == 2
    assert all(d.vendor == "nvidia" for d in devices)


# ---------------------------------------------------------------------------
# _probe_gpus — AMD
# ---------------------------------------------------------------------------

def test_probe_gpus_amd():
    import json
    amd_json = json.dumps({"card0": {"VRAM Total Memory (B)": str(8 * 1024 * 1024 * 1024)}})
    # NVIDIA not found → FileNotFoundError; AMD returns data
    def fake_run(cmd, **kwargs):
        if "nvidia-smi" in cmd[0]:
            raise FileNotFoundError
        return MagicMock(returncode=0, stdout=amd_json)

    with patch("subprocess.run", side_effect=fake_run):
        devices = _probe_gpus(is_apple=False, ram_total_mb=32768)
    assert len(devices) == 1
    assert devices[0].vendor == "amd"
    assert devices[0].vram_mb == 8192


# ---------------------------------------------------------------------------
# _probe_gpus — Apple Silicon
# ---------------------------------------------------------------------------

def test_probe_gpus_apple_silicon():
    devices = _probe_gpus(is_apple=True, ram_total_mb=16384)
    assert len(devices) == 1
    assert devices[0].vendor == "apple"
    assert devices[0].vram_mb == 16384
    assert "Apple Silicon" in devices[0].name


# ---------------------------------------------------------------------------
# _probe_gpus — no GPU
# ---------------------------------------------------------------------------

def test_probe_gpus_no_gpu():
    with patch("subprocess.run", side_effect=FileNotFoundError), \
         patch("agentic_concierge.bootstrap.system_probe._probe_amd_sysfs", return_value=[]):
        devices = _probe_gpus(is_apple=False, ram_total_mb=16384)
    assert devices == []


# ---------------------------------------------------------------------------
# probe_system (async)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_probe_system_basic():
    """probe_system() returns a valid SystemProbe without raising."""
    with (
        patch("agentic_concierge.bootstrap.system_probe._check_internet", return_value=False),
        patch("agentic_concierge.bootstrap.system_probe._check_ollama", return_value=False),
        patch("agentic_concierge.bootstrap.system_probe._check_vllm", return_value=False),
        patch("subprocess.run", side_effect=FileNotFoundError),
    ):
        probe = await probe_system()
    assert isinstance(probe, SystemProbe)
    assert probe.cpu_cores >= 1
    assert probe.ram_total_mb >= 0
    assert probe.internet_reachable is False
    assert probe.ollama_reachable is False


@pytest.mark.asyncio
async def test_probe_system_internet_reachable():
    with (
        patch("agentic_concierge.bootstrap.system_probe._check_internet", return_value=True),
        patch("agentic_concierge.bootstrap.system_probe._check_ollama", return_value=False),
        patch("agentic_concierge.bootstrap.system_probe._check_vllm", return_value=False),
        patch("subprocess.run", side_effect=FileNotFoundError),
    ):
        probe = await probe_system()
    assert probe.internet_reachable is True


@pytest.mark.asyncio
async def test_probe_system_ollama_reachable():
    with (
        patch("agentic_concierge.bootstrap.system_probe._check_internet", return_value=False),
        patch("agentic_concierge.bootstrap.system_probe._check_ollama", return_value=True),
        patch("agentic_concierge.bootstrap.system_probe._check_vllm", return_value=False),
        patch("subprocess.run", side_effect=FileNotFoundError),
        patch("shutil.which", return_value="/usr/bin/ollama"),
    ):
        probe = await probe_system()
    assert probe.ollama_installed is True


@pytest.mark.asyncio
async def test_probe_system_llama_server_not_installed():
    with (
        patch("agentic_concierge.bootstrap.system_probe._check_internet", return_value=False),
        patch("agentic_concierge.bootstrap.system_probe._check_ollama", return_value=False),
        patch("agentic_concierge.bootstrap.system_probe._check_vllm", return_value=False),
        patch("subprocess.run", side_effect=FileNotFoundError),
        patch("shutil.which", return_value=None),
    ):
        probe = await probe_system()
    assert probe.llama_server_installed is False


@pytest.mark.asyncio
async def test_probe_system_vllm_reachable():
    with (
        patch("agentic_concierge.bootstrap.system_probe._check_internet", return_value=False),
        patch("agentic_concierge.bootstrap.system_probe._check_ollama", return_value=False),
        patch("agentic_concierge.bootstrap.system_probe._check_vllm", return_value=True),
        patch("subprocess.run", side_effect=FileNotFoundError),
    ):
        probe = await probe_system("http://localhost:8000")
    assert probe.vllm_reachable is True


# ---------------------------------------------------------------------------
# _probe_amd_sysfs — AMD APU detection via sysfs
# ---------------------------------------------------------------------------


def test_probe_amd_sysfs_returns_list():
    """sysfs fallback always returns a list (may be empty on non-AMD machines)."""
    devices = _probe_amd_sysfs(ram_total_mb=128000)
    assert isinstance(devices, list)
    # On a machine with AMD GPU, we should get results
    # On other machines, returns empty — either way it shouldn't crash
    for d in devices:
        assert d.vendor == "amd"
        assert d.vram_mb > 0


def test_probe_gpus_falls_through_to_sysfs():
    """When nvidia-smi and rocm-smi both fail, _probe_gpus calls _probe_amd_sysfs."""
    fake_device = GPUDevice(name="AMD APU", vram_mb=128000, vendor="amd")
    with patch("subprocess.run", side_effect=FileNotFoundError), \
         patch("agentic_concierge.bootstrap.system_probe._probe_amd_sysfs",
               return_value=[fake_device]) as mock_sysfs:
        devices = _probe_gpus(is_apple=False, ram_total_mb=128000)

    mock_sysfs.assert_called_once_with(128000)
    assert len(devices) == 1
    assert devices[0].name == "AMD APU"
    assert devices[0].vram_mb == 128000
