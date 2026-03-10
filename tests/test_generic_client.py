"""Tests for GenericChatClient and build_chat_client() factory (Phase 4)."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import pytest

from agentic_concierge.config.schema import ModelConfig
from agentic_concierge.infrastructure.chat import build_chat_client
from agentic_concierge.infrastructure.chat.generic import GenericChatClient
from agentic_concierge.infrastructure.ollama.client import OllamaChatClient


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _response_data(content: str = "Hello", tool_calls: list | None = None) -> dict:
    """Build a minimal OpenAI chat-completions response dict."""
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}]}


def _tool_call_data(name: str, args: dict, call_id: str = "c1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


# ---------------------------------------------------------------------------
# build_chat_client factory
# ---------------------------------------------------------------------------

def test_build_chat_client_default_backend_returns_ollama():
    cfg = ModelConfig(base_url="http://localhost:11434/v1", model="test")
    assert cfg.backend == "ollama"
    client = build_chat_client(cfg)
    assert isinstance(client, OllamaChatClient)


def test_build_chat_client_ollama_backend_explicit():
    cfg = ModelConfig(base_url="http://localhost:11434/v1", model="test", backend="ollama")
    client = build_chat_client(cfg)
    assert isinstance(client, OllamaChatClient)


def test_build_chat_client_generic_backend():
    cfg = ModelConfig(base_url="https://api.openai.com/v1", model="gpt-4o", backend="generic")
    client = build_chat_client(cfg)
    assert isinstance(client, GenericChatClient)


def test_build_chat_client_unknown_backend_raises():
    cfg = ModelConfig(base_url="http://example.com/v1", model="test", backend="unknown_backend")
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        build_chat_client(cfg)


def test_build_chat_client_passes_api_key():
    cfg = ModelConfig(
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
        api_key="sk-test123",
        backend="generic",
    )
    client = build_chat_client(cfg)
    assert isinstance(client, GenericChatClient)
    assert client._api_key == "sk-test123"


# ---------------------------------------------------------------------------
# GenericChatClient: correct HTTP request
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generic_client_sends_correct_payload():
    """GenericChatClient sends the full payload with model, messages, tools."""
    data = _response_data("Hi")
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = data

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
        client = GenericChatClient(base_url="http://example.com/v1")
        await client.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
            tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
        )

    _, kwargs = mock_post.call_args
    sent = kwargs["json"]
    assert sent["model"] == "test-model"
    assert len(sent["messages"]) == 1
    assert "tools" in sent
    assert sent["stream"] is False


@pytest.mark.asyncio
async def test_generic_client_sends_auth_header_when_api_key_set():
    data = _response_data()
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = data

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
        client = GenericChatClient(base_url="http://example.com/v1", api_key="sk-abc")
        await client.chat(messages=[], model="m")

    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer sk-abc"


@pytest.mark.asyncio
async def test_generic_client_no_auth_header_when_no_api_key():
    data = _response_data()
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = data

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
        client = GenericChatClient(base_url="http://example.com/v1")
        await client.chat(messages=[], model="m")

    _, kwargs = mock_post.call_args
    assert "Authorization" not in kwargs["headers"]


# ---------------------------------------------------------------------------
# GenericChatClient: response parsing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generic_client_parses_plain_text_response():
    data = _response_data("The answer is 42.")
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = data

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        client = GenericChatClient(base_url="http://example.com/v1")
        result = await client.chat(messages=[], model="m")

    assert result.content == "The answer is 42."
    assert result.tool_calls == []
    assert not result.has_tool_calls


@pytest.mark.asyncio
async def test_generic_client_parses_tool_call_response():
    data = _response_data(
        content=None,
        tool_calls=[_tool_call_data("list_files", {"path": "."})],
    )
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = data

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        client = GenericChatClient(base_url="http://example.com/v1")
        result = await client.chat(messages=[], model="m")

    assert result.has_tool_calls
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "list_files"
    assert result.tool_calls[0].arguments == {"path": "."}


@pytest.mark.asyncio
async def test_generic_client_handles_malformed_json_args():
    """Malformed JSON arguments fall back to {'_raw': ...} instead of raising."""
    data = _response_data(
        content=None,
        tool_calls=[{
            "id": "c1",
            "type": "function",
            "function": {"name": "tool", "arguments": "{not valid json}"},
        }],
    )
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = data

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        client = GenericChatClient(base_url="http://example.com/v1")
        result = await client.chat(messages=[], model="m")

    assert result.tool_calls[0].arguments == {"_raw": "{not valid json}"}


# ---------------------------------------------------------------------------
# GenericChatClient: error handling (no Ollama 400 retry)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generic_client_raises_on_4xx_immediately():
    """GenericChatClient raises httpx.HTTPStatusError on 4xx; no retry."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 400
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "400 Bad Request", request=MagicMock(), response=mock_resp
    )
    mock_resp.json.return_value = {"error": "bad request"}

    call_count = 0

    async def fake_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return mock_resp

    with patch("httpx.AsyncClient.post", new=fake_post):
        client = GenericChatClient(base_url="http://example.com/v1")
        with pytest.raises(httpx.HTTPStatusError):
            await client.chat(messages=[], model="m")

    # Must NOT retry — only one POST call issued.
    assert call_count == 1


@pytest.mark.asyncio
async def test_generic_client_raises_on_500():
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 500
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=mock_resp
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        client = GenericChatClient(base_url="http://example.com/v1")
        with pytest.raises(httpx.HTTPStatusError):
            await client.chat(messages=[], model="m")


# ---------------------------------------------------------------------------
# ModelConfig.backend field
# ---------------------------------------------------------------------------

def test_model_config_default_backend_is_ollama():
    cfg = ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5:7b")
    assert cfg.backend == "ollama"


def test_model_config_backend_persists():
    cfg = ModelConfig(base_url="https://api.openai.com/v1", model="gpt-4o", backend="generic")
    assert cfg.backend == "generic"


# ---------------------------------------------------------------------------
# P10-9: vllm backend in build_chat_client factory
# ---------------------------------------------------------------------------

def test_build_chat_client_vllm_backend():
    from agentic_concierge.infrastructure.chat.vllm import VLLMChatClient
    cfg = ModelConfig(base_url="http://localhost:8000/v1", model="qwen2.5-7b", backend="vllm")
    client = build_chat_client(cfg)
    assert isinstance(client, VLLMChatClient)
    assert client._base_url == "http://localhost:8000/v1"


def test_build_chat_client_vllm_passes_api_key():
    from agentic_concierge.infrastructure.chat.vllm import VLLMChatClient
    cfg = ModelConfig(
        base_url="http://localhost:8000/v1",
        model="qwen2.5-7b",
        api_key="tok-abc",
        backend="vllm",
    )
    client = build_chat_client(cfg)
    assert client._api_key == "tok-abc"


def test_build_chat_client_unknown_backend_error_mentions_backends():
    cfg = ModelConfig(base_url="http://x/v1", model="m", backend="totally_unknown")
    with pytest.raises(ValueError) as exc_info:
        build_chat_client(cfg)
    err = str(exc_info.value)
    assert "vllm" in err
    assert "ollama" in err
