"""System probe: detect CPU, RAM, GPU, disk, network, and backend availability.

All external calls (subprocess, HTTP) are best-effort — failures return safe
defaults (False / 0) rather than raising, so ``probe_system()`` always returns
a valid ``SystemProbe`` regardless of the host environment.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GPUDevice:
    """A single GPU device detected on the system."""

    name: str
    vram_mb: int
    vendor: str  # "nvidia" | "amd" | "apple"


@dataclass
class SystemProbe:
    """Snapshot of system resources and LLM backend availability."""

    cpu_cores: int
    cpu_arch: str             # "x86_64" | "aarch64" | "apple_silicon"
    ram_total_mb: int
    ram_available_mb: int
    gpu_devices: list[GPUDevice] = field(default_factory=list)
    disk_free_mb: int = 0
    internet_reachable: bool = False
    ollama_installed: bool = False
    ollama_reachable: bool = False
    vllm_reachable: bool = False
    llama_server_installed: bool = False

    @property
    def total_vram_mb(self) -> int:
        """Sum of VRAM across all GPU devices."""
        return sum(d.vram_mb for d in self.gpu_devices)


async def probe_system(vllm_base_url: str = "http://localhost:8000") -> SystemProbe:
    """Probe system resources and LLM backend availability.

    All I/O is async where possible; subprocess calls are best-effort with
    short timeouts.  Never raises — returns a safe default on any failure.
    """
    import asyncio

    cpu_cores = os.cpu_count() or 1
    machine = platform.machine()
    system = platform.system()
    is_apple = system == "Darwin" and machine == "arm64"

    if is_apple:
        cpu_arch = "apple_silicon"
    elif machine == "aarch64":
        cpu_arch = "aarch64"
    else:
        cpu_arch = "x86_64"

    # RAM via psutil (core dep)
    try:
        import psutil
        mem = psutil.virtual_memory()
        ram_total_mb = mem.total // (1024 * 1024)
        ram_available_mb = mem.available // (1024 * 1024)
    except Exception:
        ram_total_mb = 0
        ram_available_mb = 0

    gpu_devices = _probe_gpus(is_apple, ram_total_mb)

    # Disk free at user data path
    try:
        from platformdirs import user_data_path
        cache_path = str(user_data_path("agentic-concierge"))
    except Exception:
        cache_path = os.path.expanduser("~")
    try:
        disk_free_mb = shutil.disk_usage(cache_path).free // (1024 * 1024)
    except Exception:
        disk_free_mb = 0

    # Async network + backend probes
    internet_reachable, ollama_reachable, vllm_reachable = await asyncio.gather(
        _check_internet(),
        _check_ollama(),
        _check_vllm(vllm_base_url),
    )

    ollama_installed = shutil.which("ollama") is not None
    llama_server_installed = shutil.which("llama-server") is not None

    return SystemProbe(
        cpu_cores=cpu_cores,
        cpu_arch=cpu_arch,
        ram_total_mb=ram_total_mb,
        ram_available_mb=ram_available_mb,
        gpu_devices=gpu_devices,
        disk_free_mb=disk_free_mb,
        internet_reachable=internet_reachable,
        ollama_installed=ollama_installed,
        ollama_reachable=ollama_reachable,
        vllm_reachable=vllm_reachable,
        llama_server_installed=llama_server_installed,
    )


def _probe_gpus(is_apple: bool, ram_total_mb: int) -> list[GPUDevice]:
    """Detect GPU devices. Returns empty list if none found or detection fails."""
    devices: list[GPUDevice] = []

    # Apple Silicon — unified memory; report RAM as VRAM
    if is_apple:
        devices.append(GPUDevice(name="Apple Silicon", vram_mb=ram_total_mb, vendor="apple"))
        return devices

    # NVIDIA via nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = line.split(",", 1)
                if len(parts) == 2:
                    name = parts[0].strip()
                    try:
                        vram_mb = int(parts[1].strip())
                    except ValueError:
                        vram_mb = 0
                    devices.append(GPUDevice(name=name, vram_mb=vram_mb, vendor="nvidia"))
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    if devices:
        return devices

    # AMD via rocm-smi
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            # rocm-smi JSON: {"card0": {"VRAM Total Memory (B)": "...", ...}, ...}
            for card_name, card_info in data.items():
                if isinstance(card_name, str) and card_name.startswith("card"):
                    try:
                        vram_bytes = int(card_info.get("VRAM Total Memory (B)", 0))
                        vram_mb = vram_bytes // (1024 * 1024)
                    except (ValueError, TypeError):
                        vram_mb = 0
                    devices.append(
                        GPUDevice(name=f"AMD {card_name}", vram_mb=vram_mb, vendor="amd")
                    )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    if devices:
        return devices

    # Fallback: AMD APU detection via sysfs + lspci
    # rocm-smi doesn't work on APUs (e.g. Strix Halo), but the GPU is
    # detectable via /sys/class/drm and lspci.
    devices = _probe_amd_sysfs(ram_total_mb)

    return devices


def _probe_amd_sysfs(ram_total_mb: int) -> list[GPUDevice]:
    """Detect AMD GPUs via sysfs and lspci when rocm-smi is unavailable.

    This covers AMD APUs (e.g. Strix Halo / Radeon 8060S) where rocm-smi
    fails but the GPU is visible in /sys/class/drm and lspci.
    """
    import re

    devices: list[GPUDevice] = []

    # Check sysfs for AMD GPU (vendor 0x1002)
    try:
        from pathlib import Path
        drm_dir = Path("/sys/class/drm")
        if not drm_dir.exists():
            return devices

        seen_devices: set[str] = set()
        for card_dir in sorted(drm_dir.iterdir()):
            # Only look at card0, card1, etc. — skip card1-DP-1 etc.
            if not re.match(r"^card\d+$", card_dir.name):
                continue
            vendor_file = card_dir / "device" / "vendor"
            if not vendor_file.exists():
                continue
            vendor = vendor_file.read_text().strip()
            if vendor != "0x1002":  # AMD
                continue

            device_path = card_dir / "device" / "device"
            device_id = device_path.read_text().strip() if device_path.exists() else ""

            # Deduplicate by PCI device path
            pci_path = str((card_dir / "device").resolve())
            if pci_path in seen_devices:
                continue
            seen_devices.add(pci_path)

            # Read VRAM from sysfs if available
            vram_file = card_dir / "device" / "mem_info_vram_total"
            vram_mb = 0
            if vram_file.exists():
                try:
                    vram_bytes = int(vram_file.read_text().strip())
                    vram_mb = vram_bytes // (1024 * 1024)
                except (ValueError, OSError):
                    pass

            # For APUs with unified memory, if VRAM is small (< 4 GB),
            # the GPU can use system RAM. Report system RAM as usable.
            if vram_mb < 4096 and ram_total_mb > 0:
                vram_mb = ram_total_mb

            # Try to get a friendly name via lspci
            name = _lspci_gpu_name(device_id)
            if not name:
                name = f"AMD GPU ({device_id})"

            devices.append(GPUDevice(name=name, vram_mb=vram_mb, vendor="amd"))
    except Exception:
        logger.debug("sysfs AMD GPU detection failed", exc_info=True)

    return devices


def _lspci_gpu_name(device_id: str) -> str:
    """Get the friendly GPU name from lspci output matching a device ID."""
    import re
    try:
        result = subprocess.run(
            ["lspci", "-nn"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return ""
        for line in result.stdout.splitlines():
            m = re.match(
                r"^[0-9a-f:.]+\s+(?:VGA compatible|3D|Display)\s+controller:\s+(.+)",
                line,
                re.IGNORECASE,
            )
            if not m:
                continue
            desc = m.group(1)
            # Check if this line contains our device ID
            if device_id.lstrip("0x") in desc.lower():
                # Clean up the name
                name = re.sub(r"\s*\[[0-9a-fA-F]{4}:[0-9a-fA-F]{4}\]", "", desc).strip()
                name = re.sub(r"\s*\(rev\s+[0-9a-fA-F]+\)$", "", name).strip()
                return name
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return ""


async def _check_internet() -> bool:
    """Ping 1.1.1.1 to test internet connectivity."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.head("https://1.1.1.1", timeout=3.0)
        return resp.status_code < 500
    except Exception:
        return False


async def _check_ollama() -> bool:
    """Check if the Ollama server is reachable."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:11434/api/tags", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


async def _check_vllm(base_url: str) -> bool:
    """Check if a vLLM server is reachable at *base_url*."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/health", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False
