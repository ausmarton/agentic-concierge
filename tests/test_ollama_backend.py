"""Tests for OllamaBackend — InferenceBackend implementation for Ollama.

All HTTP calls are mocked.  Tests verify:
- Protocol compliance (isinstance check)
- Health check (healthy, unreachable, error)
- Model listing (available, raw, loaded)
- Model lifecycle (load, unload)
- Chat client creation
- Memory estimation
- Model pulling (success, failure, stream parsing)
- Embedding-only model filtering
- Base URL normalization
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agentic_concierge.infrastructure.backends.ollama import (
    OllamaBackend,
    _is_chat_capable,
    _model_name,
    _parse_size_bytes,
)
from agentic_concierge.infrastructure.backends.protocol import (
    InferenceBackend,
    ModelSlot,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backend():
    return OllamaBackend(base_url="http://localhost:11434")


def _mock_response(status_code=200, json_data=None):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    return resp


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocol:

    def test_ollama_backend_is_inference_backend(self, backend):
        assert isinstance(backend, InferenceBackend)

    def test_name_property(self, backend):
        assert backend.name == "ollama"

    def test_base_url_property(self, backend):
        assert backend.base_url == "http://localhost:11434"

    def test_chat_url_property(self, backend):
        assert backend.chat_url == "http://localhost:11434/v1"


# ---------------------------------------------------------------------------
# Base URL normalization
# ---------------------------------------------------------------------------


class TestBaseUrlNormalization:

    def test_strips_trailing_slash(self):
        b = OllamaBackend(base_url="http://localhost:11434/")
        assert b.base_url == "http://localhost:11434"

    def test_strips_v1_suffix(self):
        b = OllamaBackend(base_url="http://localhost:11434/v1")
        assert b.base_url == "http://localhost:11434"

    def test_strips_v1_with_trailing_slash(self):
        b = OllamaBackend(base_url="http://localhost:11434/v1/")
        assert b.base_url == "http://localhost:11434"

    def test_preserves_custom_port(self):
        b = OllamaBackend(base_url="http://myserver:9999")
        assert b.base_url == "http://myserver:9999"


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealthCheck:

    @pytest.mark.asyncio
    async def test_healthy(self, backend):
        mock_resp = _mock_response(200, {"models": []})
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            assert await backend.health_check() is True

    @pytest.mark.asyncio
    async def test_unhealthy_status(self, backend):
        mock_resp = _mock_response(503)
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            assert await backend.health_check() is False

    @pytest.mark.asyncio
    async def test_connection_error(self, backend):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=httpx.ConnectError("refused")):
            assert await backend.health_check() is False


# ---------------------------------------------------------------------------
# list_available
# ---------------------------------------------------------------------------


MOCK_TAGS_RESPONSE = {
    "models": [
        {"name": "qwen2.5:7b", "details": {"family": "qwen2.5", "parameter_size": "7.6B"}},
        {"name": "llama3.1:8b", "details": {"family": "llama", "parameter_size": "8.0B"}},
        {"name": "nomic-embed-text:latest", "details": {"family": "nomic-embed"}},
    ]
}


class TestListAvailable:

    @pytest.mark.asyncio
    async def test_returns_chat_models_only(self, backend):
        mock_resp = _mock_response(200, MOCK_TAGS_RESPONSE)
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            models = await backend.list_available()
        assert "qwen2.5:7b" in models
        assert "llama3.1:8b" in models
        assert "nomic-embed-text:latest" not in models

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self, backend):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=Exception("fail")):
            models = await backend.list_available()
        assert models == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_non_200(self, backend):
        mock_resp = _mock_response(503)
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            models = await backend.list_available()
        assert models == []


class TestListAvailableRaw:

    @pytest.mark.asyncio
    async def test_returns_all_model_dicts(self, backend):
        mock_resp = _mock_response(200, MOCK_TAGS_RESPONSE)
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            models = await backend.list_available_raw()
        assert len(models) == 3
        assert models[0]["name"] == "qwen2.5:7b"

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self, backend):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=Exception("fail")):
            models = await backend.list_available_raw()
        assert models == []


# ---------------------------------------------------------------------------
# list_loaded
# ---------------------------------------------------------------------------


MOCK_PS_RESPONSE = {
    "models": [
        {
            "name": "qwen2.5:7b",
            "model": "qwen2.5:7b",
            "size": 4_500_000_000,
            "size_vram": 4_500_000_000,
            "details": {"quantization_level": "Q4_K_M"},
        },
    ]
}


class TestListLoaded:

    @pytest.mark.asyncio
    async def test_returns_model_slots(self, backend):
        mock_resp = _mock_response(200, MOCK_PS_RESPONSE)
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            slots = await backend.list_loaded()
        assert len(slots) == 1
        assert isinstance(slots[0], ModelSlot)
        assert slots[0].model_id == "qwen2.5:7b"
        assert slots[0].backend == "ollama"
        assert slots[0].memory_mb == 4_500_000_000 // (1024 * 1024)
        assert slots[0].quantization == "Q4_K_M"

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self, backend):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=Exception("fail")):
            slots = await backend.list_loaded()
        assert slots == []

    @pytest.mark.asyncio
    async def test_empty_when_no_models_loaded(self, backend):
        mock_resp = _mock_response(200, {"models": []})
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            slots = await backend.list_loaded()
        assert slots == []


# ---------------------------------------------------------------------------
# load_model / unload_model
# ---------------------------------------------------------------------------


class TestLoadModel:

    @pytest.mark.asyncio
    async def test_load_model_success(self, backend):
        generate_resp = _mock_response(200)
        ps_resp = _mock_response(200, MOCK_PS_RESPONSE)

        async def mock_post(url, **kwargs):
            return generate_resp

        async def mock_get(url, **kwargs):
            return ps_resp

        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=mock_post),
            patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=mock_get),
        ):
            slot = await backend.load_model("qwen2.5:7b")
        assert slot.model_id == "qwen2.5:7b"
        assert slot.backend == "ollama"

    @pytest.mark.asyncio
    async def test_load_model_failure_raises(self, backend):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=Exception("connection refused")):
            with pytest.raises(RuntimeError, match="Failed to load model"):
                await backend.load_model("qwen2.5:7b")


class TestUnloadModel:

    @pytest.mark.asyncio
    async def test_unload_model_success(self, backend):
        mock_resp = _mock_response(200)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            await backend.unload_model("qwen2.5:7b")  # should not raise

    @pytest.mark.asyncio
    async def test_unload_model_failure_logs_warning(self, backend):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=Exception("fail")):
            # Should not raise, just log
            await backend.unload_model("qwen2.5:7b")


# ---------------------------------------------------------------------------
# build_client
# ---------------------------------------------------------------------------


class TestBuildClient:

    def test_returns_ollama_chat_client(self, backend):
        client = backend.build_client("qwen2.5:7b")
        from agentic_concierge.infrastructure.ollama.client import OllamaChatClient
        assert isinstance(client, OllamaChatClient)

    def test_client_base_url_has_v1(self, backend):
        client = backend.build_client("qwen2.5:7b")
        assert client._base_url == "http://localhost:11434/v1"


# ---------------------------------------------------------------------------
# estimate_memory
# ---------------------------------------------------------------------------


class TestEstimateMemory:

    @pytest.mark.asyncio
    async def test_from_size_field(self, backend):
        mock_resp = _mock_response(200, {"size": 4_500_000_000})
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            mb = await backend.estimate_memory("qwen2.5:7b")
        assert mb == 4_500_000_000 // (1024 * 1024)

    @pytest.mark.asyncio
    async def test_from_parameter_size(self, backend):
        mock_resp = _mock_response(200, {
            "size": 0,
            "details": {"parameter_size": "7.6B"},
        })
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            mb = await backend.estimate_memory("qwen2.5:7b")
        # 7.6B * 500MB/B ≈ 3800MB
        assert mb == int(7.6 * 500)

    @pytest.mark.asyncio
    async def test_returns_zero_on_error(self, backend):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=Exception("fail")):
            mb = await backend.estimate_memory("qwen2.5:7b")
        assert mb == 0

    @pytest.mark.asyncio
    async def test_returns_zero_on_non_200(self, backend):
        mock_resp = _mock_response(404)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            mb = await backend.estimate_memory("nonexistent:latest")
        assert mb == 0


# ---------------------------------------------------------------------------
# pull_model
# ---------------------------------------------------------------------------


class TestPullModel:

    @pytest.mark.asyncio
    async def test_pull_success(self, backend):
        lines = [
            '{"status": "pulling manifest"}',
            '{"status": "downloading"}',
            '{"status": "success"}',
        ]

        class FakeStream:
            status_code = 200
            async def aiter_lines(self):
                for line in lines:
                    yield line
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass

        with patch("httpx.AsyncClient.stream", return_value=FakeStream()):
            result = await backend.pull_model("qwen2.5:7b")
        assert result is True

    @pytest.mark.asyncio
    async def test_pull_error_in_stream(self, backend):
        lines = [
            '{"status": "pulling manifest"}',
            '{"error": "model not found"}',
        ]

        class FakeStream:
            status_code = 200
            async def aiter_lines(self):
                for line in lines:
                    yield line
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass

        with patch("httpx.AsyncClient.stream", return_value=FakeStream()):
            result = await backend.pull_model("nonexistent:latest")
        assert result is False

    @pytest.mark.asyncio
    async def test_pull_connection_error(self, backend):
        with patch("httpx.AsyncClient.stream", side_effect=httpx.ConnectError("refused")):
            result = await backend.pull_model("qwen2.5:7b")
        assert result is False

    @pytest.mark.asyncio
    async def test_pull_non_200_status(self, backend):
        class FakeStream:
            status_code = 404
            async def aiter_lines(self):
                return
                yield  # make it an async generator
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass

        with patch("httpx.AsyncClient.stream", return_value=FakeStream()):
            result = await backend.pull_model("nonexistent:latest")
        assert result is False


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestIsChatCapable:

    def test_chat_model(self):
        assert _is_chat_capable({"name": "qwen2.5:7b", "details": {"family": "qwen2.5"}}) is True

    def test_embedding_model(self):
        assert _is_chat_capable({"name": "nomic-embed-text:latest", "details": {"family": "nomic-embed"}}) is False

    def test_bge_model(self):
        assert _is_chat_capable({"name": "bge-m3:latest", "details": {}}) is False

    def test_embed_in_name(self):
        assert _is_chat_capable({"name": "custom-embed-v1:latest", "details": {}}) is False

    def test_embed_with_instruct(self):
        assert _is_chat_capable({"name": "some-embed-instruct:latest", "details": {}}) is True


class TestModelName:

    def test_name_field(self):
        assert _model_name({"name": "qwen2.5:7b"}) == "qwen2.5:7b"

    def test_model_field(self):
        assert _model_name({"model": "llama3.1:8b"}) == "llama3.1:8b"

    def test_empty(self):
        assert _model_name({}) == ""


class TestParseSizeBytes:

    def test_plain_int(self):
        assert _parse_size_bytes("4500000000") == 4_500_000_000

    def test_gb(self):
        assert _parse_size_bytes("4.5 GB") == int(4.5 * 1024**3)

    def test_mb(self):
        assert _parse_size_bytes("512 MB") == 512 * 1024**2

    def test_empty(self):
        assert _parse_size_bytes("") == 0

    def test_unparseable(self):
        assert _parse_size_bytes("not a number") == 0


# ---------------------------------------------------------------------------
# ModelSlot dataclass
# ---------------------------------------------------------------------------


class TestModelSlot:

    def test_defaults(self):
        slot = ModelSlot(model_id="test", backend="ollama")
        assert slot.memory_mb == 0
        assert slot.vram_mb == 0
        assert slot.quantization == ""
        assert slot.metadata == {}

    def test_populated(self):
        slot = ModelSlot(
            model_id="qwen2.5:7b",
            backend="ollama",
            memory_mb=4500,
            vram_mb=4500,
            quantization="Q4_K_M",
        )
        assert slot.model_id == "qwen2.5:7b"
        assert slot.quantization == "Q4_K_M"
