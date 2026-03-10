"""Backend manager: probe and ensure LLM backends are healthy.

Only backends enabled in the ``FeatureSet`` are probed — disabled backends
have zero resource cost (no HTTP calls, no subprocess, no imports).
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

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
        """Check if llama-server is installed and if any managed processes are running."""
        binary = shutil.which("llama-server")
        if binary is None:
            return BackendHealth(
                name="llama_cpp",
                status=BackendStatus.NOT_INSTALLED,
                hint="Install llama.cpp: https://github.com/ggerganov/llama.cpp#build",
            )
        # llama-server is installed; check if any managed instances are running
        # by probing the port range used by LlamaCppBackend.
        import socket
        running_ports: list[int] = []
        for port in range(8100, 8110):  # check first 10 ports in the range
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.1)
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    running_ports.append(port)
        if running_ports:
            return BackendHealth(
                name="llama_cpp",
                status=BackendStatus.HEALTHY,
                hint=f"llama-server running on port(s): {running_ports}",
            )
        return BackendHealth(
            name="llama_cpp",
            status=BackendStatus.HEALTHY,
            hint="llama-server installed; models started on demand.",
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
            return BackendHealth(
                name="vllm",
                status=BackendStatus.UNREACHABLE,
                base_url=base_url,
                error=str(e),
                hint=(
                    "Start vLLM: python -m vllm.entrypoints.openai.api_server "
                    "--model <model>"
                ),
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

    def get_healthy_backends(self) -> List[str]:
        """Return names of backends with ``status=HEALTHY`` from the last probe."""
        return [
            name for name, h in self._health.items()
            if h.status == BackendStatus.HEALTHY
        ]
