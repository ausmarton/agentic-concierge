"""Backend manager: probe and ensure LLM backends are healthy.

Only backends enabled in the ``FeatureSet`` are probed — disabled backends
have zero resource cost (no HTTP calls, no subprocess, no imports).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


def _has_llama_cpp_python() -> bool:
    """Check if llama-cpp-python is importable (without importing it)."""
    import importlib.util
    return importlib.util.find_spec("llama_cpp") is not None


def _has_vllm() -> bool:
    """Check if vLLM is importable (without importing it)."""
    import importlib.util
    return importlib.util.find_spec("vllm") is not None


def _has_vllm_image() -> bool:
    """Check if a vLLM container image is available locally (podman/docker)."""
    container_cmd = shutil.which("podman") or shutil.which("docker")
    if not container_cmd:
        return False
    try:
        result = subprocess.run(
            [container_cmd, "images", "--format", "{{.Repository}}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return False
        return any("vllm" in line.lower() for line in result.stdout.splitlines())
    except Exception:
        return False


if TYPE_CHECKING:
    from agentic_concierge.config.features import FeatureSet
    from agentic_concierge.config.schema import ConciergeConfig


class BackendStatus(str, Enum):
    """Health status of a backend."""

    HEALTHY = "healthy"
    UNREACHABLE = "unreachable"
    DISABLED = "disabled"
    NOT_INSTALLED = "not_installed"
    NOT_AVAILABLE = "not_available"  # optional dependency missing


@dataclass
class BackendHealth:
    """Health report for a single backend."""

    name: str
    status: BackendStatus
    base_url: str = ""
    models: List[str] = field(default_factory=list)
    error: Optional[str] = None
    hint: Optional[str] = None


class BackendManager:
    """Probe and manage LLM backends.

    Only backends enabled in the ``FeatureSet`` are probed.  The result of
    the last ``probe_all()`` call is cached in ``_health`` and accessible via
    ``get_healthy_backends()``.
    """

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        vllm_base_url: str = "http://localhost:8000",
        llama_cpp_base_url: str = "http://localhost:8100",
    ) -> None:
        self.ollama_base_url = ollama_base_url
        self.vllm_base_url = vllm_base_url
        self.llama_cpp_base_url = llama_cpp_base_url
        self._health: Dict[str, BackendHealth] = {}
        self._vllm_pull_in_progress: bool = False
        self._vllm_pull_image: str = ""

    async def probe_all(self, feature_set: "FeatureSet") -> Dict[str, BackendHealth]:
        """Probe all enabled backends concurrently.

        Returns a dict mapping backend name → ``BackendHealth``.  Disabled
        backends are included with ``status=DISABLED`` but not actually probed.
        """
        from agentic_concierge.config.features import Feature

        coros: Dict[str, Any] = {}  # type: ignore[type-arg]
        if feature_set.is_enabled(Feature.OLLAMA):
            coros["ollama"] = self.probe_ollama()
        if feature_set.is_enabled(Feature.LLAMA_CPP):
            coros["llama_cpp"] = asyncio.to_thread(self.probe_llama_cpp)
        if feature_set.is_enabled(Feature.VLLM):
            coros["vllm"] = self.probe_vllm(self.vllm_base_url)

        results: Dict[str, BackendHealth] = {}
        if coros:
            raw = await asyncio.gather(*coros.values(), return_exceptions=True)
            for name, result in zip(coros.keys(), raw):
                if isinstance(result, Exception):
                    results[name] = BackendHealth(
                        name=name, status=BackendStatus.UNREACHABLE, error=str(result)
                    )
                else:
                    results[name] = result

        # Mark disabled backends (no probe, zero cost)
        for feature, name in [
            (Feature.OLLAMA, "ollama"),
            (Feature.LLAMA_CPP, "llama_cpp"),
            (Feature.VLLM, "vllm"),
        ]:
            if not feature_set.is_enabled(feature):
                results[name] = BackendHealth(name=name, status=BackendStatus.DISABLED)

        self._health = results
        return results

    async def probe_ollama(self) -> BackendHealth:
        """Check Ollama health and list available models."""
        installed = shutil.which("ollama") is not None
        if not installed:
            return BackendHealth(
                name="ollama",
                status=BackendStatus.NOT_INSTALLED,
                hint="Install Ollama from https://ollama.com",
            )

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.ollama_base_url}/api/tags", timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                models = [m["name"] for m in data.get("models", [])]
                return BackendHealth(
                    name="ollama",
                    status=BackendStatus.HEALTHY,
                    base_url=self.ollama_base_url,
                    models=models,
                )
            return BackendHealth(
                name="ollama",
                status=BackendStatus.UNREACHABLE,
                error=f"HTTP {resp.status_code}",
                hint="Start Ollama with: ollama serve",
            )
        except Exception as e:
            return BackendHealth(
                name="ollama",
                status=BackendStatus.UNREACHABLE,
                error=str(e),
                hint="Start Ollama with: ollama serve",
            )

    def probe_llama_cpp(self) -> BackendHealth:
        """Check if llama-server is available (binary or Python package)."""
        binary = shutil.which("llama-server")
        has_python_pkg = _has_llama_cpp_python()

        if binary is None and not has_python_pkg:
            return BackendHealth(
                name="llama_cpp",
                status=BackendStatus.NOT_INSTALLED,
                hint="Run: concierge bootstrap (auto-installs llama-cpp-python[server])",
            )

        mode = "binary" if binary else "python -m llama_cpp.server"

        # Check if any managed instances are running
        import socket
        running_ports: list[int] = []
        for port in range(8100, 8110):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.1)
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    running_ports.append(port)
        if running_ports:
            return BackendHealth(
                name="llama_cpp",
                status=BackendStatus.HEALTHY,
                hint=f"llama-server running on port(s): {running_ports} (via {mode})",
            )
        return BackendHealth(
            name="llama_cpp",
            status=BackendStatus.HEALTHY,
            hint=f"llama-server available ({mode}); models started on demand.",
        )

    async def probe_vllm(self, base_url: str) -> BackendHealth:
        """Check vLLM health and list available models."""
        try:
            async with httpx.AsyncClient() as client:
                health_resp = await client.get(f"{base_url}/health", timeout=3.0)
                if health_resp.status_code != 200:
                    return BackendHealth(
                        name="vllm",
                        status=BackendStatus.UNREACHABLE,
                        base_url=base_url,
                        error=f"Health check HTTP {health_resp.status_code}",
                        hint=(
                            "Start vLLM: python -m vllm.entrypoints.openai.api_server "
                            "--model <model>"
                        ),
                    )
                models_resp = await client.get(f"{base_url}/v1/models", timeout=3.0)
                models: List[str] = []
                if models_resp.status_code == 200:
                    data = models_resp.json()
                    models = [m["id"] for m in data.get("data", [])]
            return BackendHealth(
                name="vllm",
                status=BackendStatus.HEALTHY,
                base_url=base_url,
                models=models,
            )
        except Exception as e:
            # Check if vLLM is available but just not running
            # (consistent with llama_cpp: "installed → healthy, started on demand")
            if _has_vllm():
                return BackendHealth(
                    name="vllm",
                    status=BackendStatus.HEALTHY,
                    hint=(
                        "vLLM installed (pip); not currently running. "
                        "Models started on demand."
                    ),
                )
            if _has_vllm_image():
                return BackendHealth(
                    name="vllm",
                    status=BackendStatus.HEALTHY,
                    hint="vLLM available (container image); models started on demand.",
                )
            return BackendHealth(
                name="vllm",
                status=BackendStatus.UNREACHABLE,
                base_url=base_url,
                error=str(e),
                hint="Run: concierge bootstrap (auto-installs vLLM)",
            )

    async def ensure_ollama(self, config: "ConciergeConfig") -> BackendHealth:
        """Ensure Ollama is running; attempt to start it if configured."""
        health = await self.probe_ollama()
        if health.status == BackendStatus.HEALTHY:
            return health
        if health.status == BackendStatus.NOT_INSTALLED:
            return health

        if config.local_llm_ensure_available and config.local_llm_start_cmd:
            logger.info("Ollama unreachable; starting with: %s", config.local_llm_start_cmd)
            try:
                subprocess.Popen(
                    config.local_llm_start_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                deadline = asyncio.get_event_loop().time() + config.local_llm_start_timeout_s
                while asyncio.get_event_loop().time() < deadline:
                    await asyncio.sleep(2)
                    health = await self.probe_ollama()
                    if health.status == BackendStatus.HEALTHY:
                        logger.info("Ollama is now healthy.")
                        return health
            except Exception as e:
                logger.warning("Failed to start Ollama: %s", e)

        return health

    async def ensure_vllm(
        self,
        config: "ConciergeConfig",
        feature_set: "FeatureSet",
        install_if_missing: bool = False,
    ) -> BackendHealth:
        """Ensure vLLM is available; optionally install and start it.

        If ``install_if_missing`` is ``True`` and vLLM is not available (neither
        pip package nor container image), installs via pip or pulls a container
        image as fallback (e.g. when Python >= 3.14).  If
        ``config.vllm_ensure_available`` is ``True``, also attempts to start
        the vLLM server.
        """
        from agentic_concierge.config.features import Feature

        if not feature_set.is_enabled(Feature.VLLM):
            return BackendHealth(name="vllm", status=BackendStatus.DISABLED)

        # Step 1: Install vLLM if requested and not available
        install_error: Optional[str] = None
        if install_if_missing and not _has_vllm() and not _has_vllm_image():
            install_error = await self._install_vllm()

        health = await self.probe_vllm(self.vllm_base_url)
        if health.status == BackendStatus.HEALTHY:
            return health

        # If installation was attempted and failed, surface the error
        if install_error is not None:
            return BackendHealth(
                name="vllm",
                status=BackendStatus.NOT_INSTALLED,
                error=install_error,
                hint=install_error,
            )

        # Check availability: pip package OR container image
        has_native = _has_vllm()
        has_container = _has_vllm_image()

        if not has_native and not has_container:
            if self._vllm_pull_in_progress:
                return BackendHealth(
                    name="vllm",
                    status=BackendStatus.HEALTHY,
                    hint=f"Pulling {self._vllm_pull_image} in background; re-run 'concierge doctor' to check.",
                )
            return BackendHealth(
                name="vllm",
                status=BackendStatus.NOT_INSTALLED,
                hint="Run: concierge bootstrap (auto-installs vllm)",
            )

        # Container-only path: report available even when not running
        container_only = has_container and not has_native

        if not config.vllm_ensure_available:
            if container_only:
                return BackendHealth(
                    name="vllm",
                    status=BackendStatus.HEALTHY,
                    hint="vLLM available via container image; models started on demand.",
                )
            return health

        # Determine model to serve
        model = config.vllm_model
        if not model:
            # Auto-select from Ollama's available models (largest first)
            try:
                ollama_health = await self.probe_ollama()
                if ollama_health.models:
                    model = ollama_health.models[0]
            except Exception:
                pass
        if not model:
            if container_only:
                return BackendHealth(
                    name="vllm",
                    status=BackendStatus.HEALTHY,
                    hint="vLLM container available; set vllm_model in config to auto-start.",
                )
            logger.warning("vLLM auto-start: no model configured or discovered")
            return BackendHealth(
                name="vllm",
                status=BackendStatus.UNREACHABLE,
                error="No model for vLLM (set vllm_model in config)",
                hint="Set vllm_model in config or ensure Ollama has models pulled.",
            )

        # Start vLLM — container or native
        if container_only:
            return await self._start_vllm_container_server(config, model)

        # Map Ollama-style model name to HF model ID for native vLLM
        from agentic_concierge.infrastructure.backends.model_downloader import (
            get_hf_model_id,
        )
        hf_id = get_hf_model_id(model)
        vllm_model = hf_id if hf_id else model

        cmd = list(config.vllm_start_cmd) + [
            "--model", vllm_model,
            "--gpu-memory-utilization", str(config.vllm_gpu_memory_utilization),
            "--host", "0.0.0.0",
            "--port", str(self.vllm_base_url.split(":")[-1].rstrip("/")),
        ]

        logger.info("vLLM unreachable; starting with: %s", cmd)
        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            deadline = asyncio.get_event_loop().time() + config.vllm_start_timeout_s
            while asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(2)
                health = await self.probe_vllm(self.vllm_base_url)
                if health.status == BackendStatus.HEALTHY:
                    logger.info("vLLM is now healthy (model=%s).", model)
                    return health
        except Exception as e:
            logger.warning("Failed to start vLLM: %s", e)

        return health

    async def ensure_llama_server(
        self,
        venv_pip: Optional[str] = None,
    ) -> BackendHealth:
        """Ensure llama-server is available; pip-install llama-cpp-python if missing.

        Checks for the ``llama-server`` binary first, then for the
        ``llama-cpp-python`` Python package (which provides
        ``python -m llama_cpp.server``).  If neither is found, installs
        ``llama-cpp-python[server]`` into the concierge venv.

        Args:
            venv_pip: Path to the venv's ``pip`` binary.  If ``None``,
                attempts to discover it from ``platformdirs``.
        """
        # Already available — either binary or Python package
        if shutil.which("llama-server") is not None or _has_llama_cpp_python():
            return self.probe_llama_cpp()

        # Find the venv pip/python if not provided
        if venv_pip is None:
            try:
                from platformdirs import user_data_path
                venv_dir = user_data_path("agentic-concierge") / "venv"
                venv_python = str(venv_dir / "bin" / "python")
                if os.path.isfile(venv_python):
                    venv_pip = venv_python
                else:
                    venv_pip = str(venv_dir / "bin" / "pip")
            except Exception:
                venv_pip = "pip"

        logger.info("llama-server not found; installing llama-cpp-python[server]…")
        try:
            if venv_pip.endswith("python"):
                cmd = [venv_pip, "-m", "pip", "install", "llama-cpp-python[server]"]
            else:
                cmd = [venv_pip, "install", "llama-cpp-python[server]"]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # compilation can be slow
            )
            if result.returncode == 0:
                logger.info("llama-cpp-python[server] installed successfully.")
                return self.probe_llama_cpp()
            else:
                logger.warning(
                    "Failed to install llama-cpp-python[server]: %s",
                    result.stderr[-500:] if result.stderr else "unknown error",
                )
                return BackendHealth(
                    name="llama_cpp",
                    status=BackendStatus.NOT_INSTALLED,
                    error="pip install failed",
                    hint=f"Manual install: {' '.join(cmd)}",
                )
        except Exception as e:
            logger.warning("Failed to install llama-cpp-python: %s", e)
            return BackendHealth(
                name="llama_cpp",
                status=BackendStatus.NOT_INSTALLED,
                error=str(e),
                hint="Install manually: pip install llama-cpp-python[server]",
            )

    async def _install_vllm(self) -> Optional[str]:
        """Install vLLM; returns an error message on failure, None on success.

        vLLM requires Python <3.14 (numba constraint).  If the venv Python
        is too new, falls back to starting vLLM via a container (podman/docker).
        """
        import sys
        py_version = sys.version_info

        # Check if container runtime is available (for fallback)
        container_cmd = shutil.which("podman") or shutil.which("docker")

        # Check Python version compatibility (vLLM requires <3.14)
        if py_version >= (3, 14):
            if container_cmd:
                return await self._setup_vllm_container(container_cmd)
            return (
                f"vLLM requires Python <3.14 (current: {py_version.major}.{py_version.minor}). "
                f"Install podman or docker to run vLLM in a container."
            )

        # Try pip install
        try:
            from platformdirs import user_data_path
            venv_dir = user_data_path("agentic-concierge") / "venv"
            venv_python = str(venv_dir / "bin" / "python")
            if not os.path.isfile(venv_python):
                venv_python = "python"
        except Exception:
            venv_python = "python"

        logger.info("vLLM not found; installing vllm…")
        cmd = [venv_python, "-m", "pip", "install", "vllm"]

        try:
            result = await asyncio.to_thread(
                subprocess.run, cmd,
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0:
                logger.info("vLLM installed successfully.")
                return None
            else:
                err = result.stderr[-500:] if result.stderr else "unknown error"
                logger.warning("Failed to install vllm: %s", err)
                # Fall back to container if pip fails
                if container_cmd:
                    return await self._setup_vllm_container(container_cmd)
                return f"pip install vllm failed: {err[:200]}"
        except Exception as e:
            logger.warning("Failed to install vllm: %s", e)
            if container_cmd:
                return await self._setup_vllm_container(container_cmd)
            return str(e)

    async def _setup_vllm_container(self, container_cmd: str) -> Optional[str]:
        """Pull the vLLM container image. Returns error message or None.

        Selects ROCm image for AMD GPUs, CUDA image for NVIDIA.  Waits up
        to 90 seconds for cached/small images; if still pulling, lets the
        process continue in the background and returns success so bootstrap
        is not blocked by multi-GB downloads.
        """
        from agentic_concierge.config.platform import detect_gpu

        gpu = detect_gpu()
        image = "docker.io/rocm/vllm:latest" if gpu.vendor == "amd" else "docker.io/vllm/vllm-openai:latest"
        logger.info("Pulling vLLM container image %s via %s…", image, container_cmd)

        try:
            proc = subprocess.Popen(
                [container_cmd, "pull", image],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            def _wait_for_pull() -> tuple:
                try:
                    _, stderr = proc.communicate(timeout=90)
                    return proc.returncode, (stderr or b"").decode(errors="replace")
                except subprocess.TimeoutExpired:
                    return None, None  # still pulling

            returncode, stderr = await asyncio.to_thread(_wait_for_pull)

            if returncode is None:
                # Pull still in progress — let it continue in background
                self._vllm_pull_in_progress = True
                self._vllm_pull_image = image
                logger.info(
                    "vLLM image pull continues in background (PID %d). "
                    "Re-run 'concierge doctor' to check status.",
                    proc.pid,
                )
                return None  # Not an error — just slow

            if returncode == 0:
                logger.info("vLLM container image pulled successfully.")
                return None

            err = (stderr or "")[-300:]
            logger.warning("Failed to pull vLLM image: %s", err)
            return f"Failed to pull {image}: {err[:200]}"
        except Exception as e:
            logger.warning("Failed to start vLLM image pull: %s", e)
            return str(e)

    async def _start_vllm_container_server(
        self, config: "ConciergeConfig", model: str,
    ) -> BackendHealth:
        """Start vLLM as a container (podman/docker)."""
        container_cmd = shutil.which("podman") or shutil.which("docker")
        if not container_cmd:
            return BackendHealth(
                name="vllm",
                status=BackendStatus.HEALTHY,
                hint="vLLM container image available; container runtime not found.",
            )

        from agentic_concierge.config.platform import detect_gpu

        gpu = detect_gpu()
        image = "docker.io/rocm/vllm:latest" if gpu.vendor == "amd" else "docker.io/vllm/vllm-openai:latest"

        # GPU device flags
        if gpu.vendor == "amd":
            gpu_flags = [
                "--device", "/dev/kfd", "--device", "/dev/dri",
                "--group-add", "video",
            ]
        elif gpu.vendor == "nvidia":
            gpu_flags = ["--gpus", "all"]
        else:
            gpu_flags = []

        # Map Ollama-style model name to HF model ID for vLLM
        from agentic_concierge.infrastructure.backends.model_downloader import (
            get_hf_model_id,
        )
        hf_id = get_hf_model_id(model)
        vllm_model = hf_id if hf_id else model

        # Volume mount for HF model cache persistence
        hf_cache = os.path.expanduser("~/.cache/huggingface")
        os.makedirs(hf_cache, exist_ok=True)

        port = self.vllm_base_url.split(":")[-1].rstrip("/")
        cmd = [
            container_cmd, "run", "-d", "--rm",
            "--name", "concierge-vllm",
            *gpu_flags,
            "-v", f"{hf_cache}:/root/.cache/huggingface",
            "-p", f"{port}:8000",
            image,
            "--model", vllm_model,
            "--gpu-memory-utilization", str(config.vllm_gpu_memory_utilization),
        ]

        logger.info("Starting vLLM container: %s", " ".join(cmd))
        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            deadline = asyncio.get_event_loop().time() + config.vllm_start_timeout_s
            while asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(2)
                health = await self.probe_vllm(self.vllm_base_url)
                if health.status == BackendStatus.HEALTHY:
                    logger.info("vLLM container healthy (model=%s).", model)
                    return health
        except Exception as e:
            logger.warning("Failed to start vLLM container: %s", e)

        return BackendHealth(
            name="vllm",
            status=BackendStatus.HEALTHY,
            hint=f"vLLM container available; startup pending. Manual: {container_cmd} run …",
        )

    def get_healthy_backends(self) -> List[str]:
        """Return names of backends with ``status=HEALTHY`` from the last probe."""
        return [
            name for name, h in self._health.items()
            if h.status == BackendStatus.HEALTHY
        ]

    # -------------------------------------------------------------------
    # Backend model provisioning
    # -------------------------------------------------------------------

    async def ensure_backend_models(
        self,
        model_names: List[str],
        healthy_backends: List[str],
        interactive: bool = True,
    ) -> Dict[str, List[str]]:
        """Ensure profile models are available on all healthy backends.

        For each healthy non-Ollama backend, downloads/provisions the
        required models.  Ollama is skipped because first_run.py handles
        its model pulling separately via the /api/pull API.

        Args:
            model_names: Ollama-style model names to provision.
            healthy_backends: Names of backends that are healthy.
            interactive: Show progress when True.

        Returns:
            Dict mapping backend name -> list of models provisioned.
        """
        provisioned: Dict[str, List[str]] = {}

        for backend_name in healthy_backends:
            if backend_name == "ollama":
                continue  # Ollama has its own pull mechanism

            if backend_name == "llama_cpp":
                provisioned["llama_cpp"] = await self._provision_llama_cpp_models(
                    model_names, interactive,
                )
            elif backend_name == "vllm":
                # vLLM auto-downloads HF models on server startup;
                # we just need to ensure the name mapping is available
                # and the HF cache dir exists for container mode.
                provisioned["vllm"] = self._provision_vllm_models(model_names)

        return provisioned

    async def _provision_llama_cpp_models(
        self,
        model_names: List[str],
        interactive: bool,
    ) -> List[str]:
        """Download GGUF files for llama_cpp backend."""
        from agentic_concierge.infrastructure.backends.model_downloader import (
            default_model_dir,
            download_gguf,
            get_model_source,
        )

        model_dir = default_model_dir()
        os.makedirs(model_dir, exist_ok=True)

        provisioned = []
        for name in model_names:
            source = get_model_source(name)
            if source is None:
                logger.info(
                    "No GGUF source for %s; skipping llama_cpp provision", name,
                )
                continue

            if interactive:
                size_mb = source.get("size_mb", 0)
                _print_status(f"[dim]Downloading {name} GGUF ({size_mb} MB)...[/dim]")

            def _progress(downloaded_mb, total_mb):
                if interactive and total_mb:
                    pct = int(downloaded_mb / total_mb * 100)
                    _print_status(
                        f"[dim]  {name}: {pct}% "
                        f"({downloaded_mb:.0f}/{total_mb:.0f} MB)[/dim]"
                    )

            result = await download_gguf(
                name, model_dir,
                progress_callback=_progress if interactive else None,
            )
            if result:
                provisioned.append(name)
                if interactive:
                    _print_status(f"[green]{name} GGUF ready[/green]")
            else:
                if interactive:
                    _print_status(
                        f"[yellow]Failed to download {name} GGUF[/yellow]"
                    )

        return provisioned

    def _provision_vllm_models(self, model_names: List[str]) -> List[str]:
        """Ensure vLLM model name mapping is available.

        vLLM auto-downloads HuggingFace models on startup, so we just
        need to ensure the HF cache directory exists (for container mode)
        and log the mapping.
        """
        from agentic_concierge.infrastructure.backends.model_downloader import (
            get_hf_model_id,
        )

        # Ensure HF cache directory exists (needed for container volume mount)
        hf_cache = os.path.expanduser("~/.cache/huggingface")
        os.makedirs(hf_cache, exist_ok=True)

        mapped = []
        for name in model_names:
            hf_id = get_hf_model_id(name)
            if hf_id:
                logger.info("vLLM model mapping: %s -> %s", name, hf_id)
                mapped.append(name)

        return mapped


def _print_status(msg: str) -> None:
    """Print a status line using Rich if available, otherwise plain print."""
    try:
        from rich import print as rprint
        rprint(msg)
    except ImportError:
        print(msg)
