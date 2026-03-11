"""Tests for infrastructure/backends/model_downloader.py."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_concierge.infrastructure.backends.model_downloader import (
    _load_aliases,
    _save_aliases,
    default_model_dir,
    download_gguf,
    get_gguf_url,
    get_hf_model_id,
    get_model_source,
    invalidate_sources_cache,
    list_aliased_models,
    resolve_alias,
)


@pytest.fixture(autouse=True)
def _clear_sources_cache():
    """Ensure the module-level cache is cleared between tests."""
    invalidate_sources_cache()
    yield
    invalidate_sources_cache()


# ---------------------------------------------------------------------------
# get_model_source
# ---------------------------------------------------------------------------


def test_get_model_source_known():
    source = get_model_source("qwen2.5:7b")
    assert source is not None
    assert source["gguf_repo"] == "Qwen/Qwen2.5-7B-Instruct-GGUF"
    assert source["gguf_file"] == "qwen2.5-7b-instruct-q4_k_m.gguf"
    assert source["hf_model_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert source["size_mb"] == 4680


def test_get_model_source_unknown():
    assert get_model_source("nonexistent:model") is None


# ---------------------------------------------------------------------------
# get_hf_model_id
# ---------------------------------------------------------------------------


def test_get_hf_model_id():
    assert get_hf_model_id("qwen2.5:7b") == "Qwen/Qwen2.5-7B-Instruct"
    assert get_hf_model_id("phi3:mini") == "microsoft/Phi-3-mini-4k-instruct"
    assert get_hf_model_id("unknown:model") is None


# ---------------------------------------------------------------------------
# get_gguf_url
# ---------------------------------------------------------------------------


def test_get_gguf_url():
    result = get_gguf_url("qwen2.5:7b")
    assert result is not None
    url, filename = result
    assert filename == "qwen2.5-7b-instruct-q4_k_m.gguf"
    assert "huggingface.co" in url
    assert "Qwen/Qwen2.5-7B-Instruct-GGUF" in url
    assert filename in url


def test_get_gguf_url_unknown():
    assert get_gguf_url("unknown:model") is None


# ---------------------------------------------------------------------------
# default_model_dir
# ---------------------------------------------------------------------------


def test_default_model_dir():
    d = default_model_dir()
    assert isinstance(d, str)
    assert "gguf" in d.lower() or "models" in d.lower()


# ---------------------------------------------------------------------------
# Alias roundtrip
# ---------------------------------------------------------------------------


def test_alias_roundtrip(tmp_path):
    aliases = {"qwen2.5:7b": "qwen2.5-7b-instruct-q4_k_m.gguf"}
    _save_aliases(str(tmp_path), aliases)
    loaded = _load_aliases(str(tmp_path))
    assert loaded == aliases


def test_load_aliases_missing_dir(tmp_path):
    """Returns empty dict when model_aliases.json does not exist."""
    missing = str(tmp_path / "no_such_dir")
    assert _load_aliases(missing) == {}


# ---------------------------------------------------------------------------
# resolve_alias
# ---------------------------------------------------------------------------


def test_resolve_alias(tmp_path):
    aliases = {"qwen2.5:7b": "qwen2.5-7b-instruct-q4_k_m.gguf"}
    _save_aliases(str(tmp_path), aliases)
    assert resolve_alias(str(tmp_path), "qwen2.5:7b") == "qwen2.5-7b-instruct-q4_k_m.gguf"
    assert resolve_alias(str(tmp_path), "unknown:model") is None


# ---------------------------------------------------------------------------
# list_aliased_models
# ---------------------------------------------------------------------------


def test_list_aliased_models(tmp_path):
    # Create a fake GGUF file
    (tmp_path / "qwen2.5-7b-instruct-q4_k_m.gguf").write_bytes(b"fake")
    # Register alias
    aliases = {
        "qwen2.5:7b": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "missing:model": "does-not-exist.gguf",
    }
    _save_aliases(str(tmp_path), aliases)

    result = list_aliased_models(str(tmp_path))
    assert "qwen2.5:7b" in result
    assert "missing:model" not in result  # file doesn't exist


# ---------------------------------------------------------------------------
# download_gguf
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_gguf_already_exists(tmp_path):
    """Skip download when file already exists with correct size."""
    gguf_file = "qwen2.5-7b-instruct-q4_k_m.gguf"
    dest = tmp_path / gguf_file
    # Write a file roughly matching expected size (4680 MB expected, we fake it)
    dest.write_bytes(b"x" * 1024)

    # Patch get_model_source to say expected_mb=0 (so any size is accepted)
    with patch(
        "agentic_concierge.infrastructure.backends.model_downloader.get_model_source",
        return_value={"gguf_repo": "R", "gguf_file": gguf_file, "size_mb": 0},
    ):
        result = await download_gguf("qwen2.5:7b", str(tmp_path))

    assert result == gguf_file
    # Check alias was registered
    aliases = _load_aliases(str(tmp_path))
    assert aliases.get("qwen2.5:7b") == gguf_file


@pytest.mark.asyncio
async def test_download_gguf_no_source():
    """Returns None when model has no GGUF source info."""
    result = await download_gguf("unknown:model", "/tmp/models")
    assert result is None


@pytest.mark.asyncio
async def test_download_gguf_http_failure(tmp_path):
    """Returns None on HTTP error."""
    import httpx

    mock_response = MagicMock()
    mock_response.status_code = 404

    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=mock_stream_ctx)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await download_gguf("qwen2.5:0.5b", str(tmp_path))

    assert result is None
