"""Tests for bootstrap/backend_manager.py — all HTTP calls mocked."""

from __future__ import annotations

import importlib.util
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agentic_concierge.bootstrap.backend_manager import (
    BackendHealth,
    BackendManager,
    BackendStatus,
)
from agentic_concierge.config.features import Feature, FeatureSet


def _fs(*features: Feature) -> FeatureSet:
    """Build a FeatureSet with the given features enabled."""
    return FeatureSet(enabled=frozenset(features))


def _all_fs() -> FeatureSet:
    return FeatureSet.all_enabled()


# ---------------------------------------------------------------------------
# probe_ollama
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_probe_ollama_healthy():
    tags_resp = MagicMock()
    tags_resp.status_code = 200
    tags_resp.json.return_value = {"models": [{"name": "qwen2.5:7b"}]}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=tags_resp)

    mgr = BackendManager()
    with (
        patch("shutil.which", return_value="/usr/bin/ollama"),
        patch("httpx.AsyncClient", return_value=mock_client),
    ):
        health = await mgr.probe_ollama()

    assert health.status == BackendStatus.HEALTHY
    assert "qwen2.5:7b" in health.models


@pytest.mark.asyncio
async def test_probe_ollama_not_installed():
    mgr = BackendManager()
    with patch("shutil.which", return_value=None):
        health = await mgr.probe_ollama()
    assert health.status == BackendStatus.NOT_INSTALLED
    assert "ollama.com" in health.hint.lower()


@pytest.mark.asyncio
async def test_probe_ollama_unreachable():
    mgr = BackendManager()
    with (
        patch("shutil.which", return_value="/usr/bin/ollama"),
        patch("httpx.AsyncClient.__aenter__", side_effect=httpx.ConnectError("refused")),
    ):
        health = await mgr.probe_ollama()
    assert health.status == BackendStatus.UNREACHABLE
    assert health.hint is not None


# ---------------------------------------------------------------------------
# probe_vllm
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_probe_vllm_healthy():
    health_resp = MagicMock()
    health_resp.status_code = 200
    models_resp = MagicMock()
    models_resp.status_code = 200
    models_resp.json.return_value = {"data": [{"id": "mistral-7b"}]}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=[health_resp, models_resp])

    mgr = BackendManager()
    with patch("httpx.AsyncClient", return_value=mock_client):
        health = await mgr.probe_vllm("http://localhost:8000")

    assert health.status == BackendStatus.HEALTHY
    assert "mistral-7b" in health.models


@pytest.mark.asyncio
async def test_probe_vllm_unreachable():
    mgr = BackendManager()
    with patch(
        "httpx.AsyncClient.__aenter__",
        side_effect=httpx.ConnectError("refused"),
    ):
        health = await mgr.probe_vllm("http://localhost:8000")
    assert health.status == BackendStatus.UNREACHABLE


# ---------------------------------------------------------------------------
# probe_llama_cpp
# ---------------------------------------------------------------------------

def test_probe_llama_cpp_installed():
    mgr = BackendManager()
    with patch("shutil.which", return_value="/usr/bin/llama-server"):
        health = mgr.probe_llama_cpp()
    assert health.status == BackendStatus.HEALTHY
    assert "llama-server" in health.hint.lower()


def test_probe_llama_cpp_not_installed():
    mgr = BackendManager()
    with (
        patch("shutil.which", return_value=None),
        patch("agentic_concierge.bootstrap.backend_manager._has_llama_cpp_python", return_value=False),
    ):
        health = mgr.probe_llama_cpp()
    assert health.status == BackendStatus.NOT_INSTALLED
    assert "bootstrap" in health.hint.lower()


def test_probe_llama_cpp_python_package_only():
    """When llama-server binary is missing but Python package is installed, report healthy."""
    mgr = BackendManager()
    with (
        patch("shutil.which", return_value=None),
        patch("agentic_concierge.bootstrap.backend_manager._has_llama_cpp_python", return_value=True),
    ):
        health = mgr.probe_llama_cpp()
    assert health.status == BackendStatus.HEALTHY
    assert "python" in health.hint.lower()


# ---------------------------------------------------------------------------
# probe_all
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_probe_all_disabled_backend_not_probed():
    """Disabled backends are marked DISABLED without any I/O."""
    # Only OLLAMA enabled; llama_cpp, vLLM disabled
    fs = _fs(Feature.OLLAMA)
    mgr = BackendManager()
    tags_resp = MagicMock()
    tags_resp.status_code = 200
    tags_resp.json.return_value = {"models": [{"name": "qwen2.5:7b"}]}
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=tags_resp)
    with (
        patch("shutil.which", return_value="/usr/bin/ollama"),
        patch("httpx.AsyncClient", return_value=mock_client),
    ):
        result = await mgr.probe_all(fs)

    assert result["ollama"].status == BackendStatus.HEALTHY
    assert result["llama_cpp"].status == BackendStatus.DISABLED
    assert result["vllm"].status == BackendStatus.DISABLED


@pytest.mark.asyncio
async def test_probe_all_get_healthy_backends():
    fs = _fs(Feature.OLLAMA)
    mgr = BackendManager()
    tags_resp = MagicMock()
    tags_resp.status_code = 200
    tags_resp.json.return_value = {"models": [{"name": "qwen2.5:7b"}]}
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=tags_resp)
    with (
        patch("shutil.which", return_value="/usr/bin/ollama"),
        patch("httpx.AsyncClient", return_value=mock_client),
    ):
        await mgr.probe_all(fs)
    healthy = mgr.get_healthy_backends()
    assert "ollama" in healthy
    assert "llama_cpp" not in healthy


@pytest.mark.asyncio
async def test_probe_all_all_disabled():
    fs = FeatureSet(enabled=frozenset())
    mgr = BackendManager()
    result = await mgr.probe_all(fs)
    for h in result.values():
        assert h.status == BackendStatus.DISABLED


# ---------------------------------------------------------------------------
# get_healthy_backends before probe
# ---------------------------------------------------------------------------

def test_get_healthy_backends_empty_before_probe():
    mgr = BackendManager()
    assert mgr.get_healthy_backends() == []
