"""Tests for capability probing — micro-prompt probes and cache management.

Verifies:
- tool_calling probe detects tool call capability
- structured_output probe detects JSON generation
- instruction_following probe detects format compliance
- probe_model_capabilities runs all probes
- Cache loading and saving
- probe_or_cache uses cache when available
- probe_or_cache runs probes and saves when not cached
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from agentic_concierge.domain import LLMResponse, ToolCallRequest
from agentic_concierge.infrastructure.backends.capability_probe import (
    _probe_instruction_following,
    _probe_structured_output,
    _probe_tool_calling,
    load_probe_cache,
    probe_model_capabilities,
    probe_or_cache,
    save_probe_cache,
)


# ---------------------------------------------------------------------------
# Fake chat client
# ---------------------------------------------------------------------------


class FakeChatClient:
    """Configurable fake chat client for probe testing.

    Can return different responses per call via ``responses`` list,
    or the same response for all calls via ``content``/``tool_calls``.
    """

    def __init__(
        self,
        content: str = "",
        tool_calls: Optional[List[ToolCallRequest]] = None,
        raise_error: bool = False,
        responses: Optional[List[LLMResponse]] = None,
    ) -> None:
        self._content = content
        self._tool_calls = tool_calls
        self._raise_error = raise_error
        self._responses = list(responses) if responses else None
        self._call_count = 0

    async def chat(self, messages, model, **kwargs) -> LLMResponse:
        if self._raise_error:
            raise RuntimeError("chat failed")
        if self._responses and self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        self._call_count += 1
        return LLMResponse(
            content=self._content,
            tool_calls=self._tool_calls or [],
        )


def _tc(name: str = "get_weather", args: Optional[Dict] = None) -> ToolCallRequest:
    """Helper to create a ToolCallRequest."""
    return ToolCallRequest(
        call_id="1", tool_name=name, arguments=args or {},
    )


def _resp(content: str = "", tool_calls: Optional[List[ToolCallRequest]] = None) -> LLMResponse:
    """Helper to create an LLMResponse."""
    return LLMResponse(content=content, tool_calls=tool_calls or [])


# ---------------------------------------------------------------------------
# tool_calling probe
# ---------------------------------------------------------------------------


class TestProbeToolCalling:

    @pytest.mark.asyncio
    async def test_detects_tool_call(self):
        client = FakeChatClient(tool_calls=[_tc()])
        score = await _probe_tool_calling(client, "test-model")
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_no_tool_call(self):
        client = FakeChatClient(content="The weather is sunny.")
        score = await _probe_tool_calling(client, "test-model")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_error_returns_zero(self):
        client = FakeChatClient(raise_error=True)
        score = await _probe_tool_calling(client, "test-model")
        assert score == 0.0


# ---------------------------------------------------------------------------
# structured_output probe
# ---------------------------------------------------------------------------


class TestProbeStructuredOutput:

    @pytest.mark.asyncio
    async def test_valid_json(self):
        client = FakeChatClient(content='{"name": "Alice", "age": 30}')
        score = await _probe_structured_output(client, "test-model")
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_valid_json_missing_key(self):
        client = FakeChatClient(content='{"foo": "bar"}')
        score = await _probe_structured_output(client, "test-model")
        assert score == 0.5

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        client = FakeChatClient(content="Here is the JSON: {name: Alice}")
        score = await _probe_structured_output(client, "test-model")
        assert score == 0.2

    @pytest.mark.asyncio
    async def test_empty_response(self):
        client = FakeChatClient(content="")
        score = await _probe_structured_output(client, "test-model")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_error_returns_zero(self):
        client = FakeChatClient(raise_error=True)
        score = await _probe_structured_output(client, "test-model")
        assert score == 0.0


# ---------------------------------------------------------------------------
# instruction_following probe
# ---------------------------------------------------------------------------


class TestProbeInstructionFollowing:

    @pytest.mark.asyncio
    async def test_single_word(self):
        client = FakeChatClient(content="Paris")
        score = await _probe_instruction_following(client, "test-model")
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_single_word_with_period(self):
        client = FakeChatClient(content="Paris.")
        score = await _probe_instruction_following(client, "test-model")
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_short_response(self):
        client = FakeChatClient(content="It is Paris")
        score = await _probe_instruction_following(client, "test-model")
        assert score == 0.7

    @pytest.mark.asyncio
    async def test_long_response(self):
        client = FakeChatClient(
            content="The capital of France is Paris, which is located on the Seine.",
        )
        score = await _probe_instruction_following(client, "test-model")
        assert score == 0.3

    @pytest.mark.asyncio
    async def test_error_returns_zero(self):
        client = FakeChatClient(raise_error=True)
        score = await _probe_instruction_following(client, "test-model")
        assert score == 0.0


# ---------------------------------------------------------------------------
# probe_model_capabilities
# ---------------------------------------------------------------------------


class TestProbeModelCapabilities:

    @pytest.mark.asyncio
    async def test_runs_all_probes(self):
        client = FakeChatClient(responses=[
            _resp(tool_calls=[_tc()]),              # tool_calling probe
            _resp(content='{"name": "Alice"}'),     # structured_output probe
            _resp(content="Paris"),                  # instruction_following probe
        ])

        results = await probe_model_capabilities(client, "test-model")

        assert "tool_calling" in results
        assert "structured_output" in results
        assert "instruction_following" in results
        assert "supports_tool_calling" in results

    @pytest.mark.asyncio
    async def test_tool_capable_model(self):
        client = FakeChatClient(responses=[
            _resp(tool_calls=[_tc()]),              # tool_calling → 1.0
            _resp(content='{"name": "A"}'),         # structured_output
            _resp(content="Paris"),                  # instruction_following
        ])

        results = await probe_model_capabilities(client, "test-model")
        assert results["tool_calling"] == 1.0
        assert results["supports_tool_calling"] == 1.0

    @pytest.mark.asyncio
    async def test_non_tool_capable_model(self):
        client = FakeChatClient(content="Paris")

        results = await probe_model_capabilities(client, "test-model")
        assert results["tool_calling"] == 0.0
        assert results["supports_tool_calling"] == 0.0


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


class TestCache:

    def test_save_and_load(self, tmp_path: Path):
        cache = {
            "model-a": {"tool_calling": 1.0, "structured_output": 0.8},
            "model-b": {"tool_calling": 0.0},
        }
        save_probe_cache(cache, str(tmp_path))
        loaded = load_probe_cache(str(tmp_path))
        assert loaded == cache

    def test_load_empty_when_no_file(self, tmp_path: Path):
        assert load_probe_cache(str(tmp_path)) == {}

    def test_load_empty_on_corrupt_file(self, tmp_path: Path):
        cache_file = tmp_path / "model_probes.json"
        cache_file.write_text("not valid json{{{")
        assert load_probe_cache(str(tmp_path)) == {}

    def test_save_creates_directory(self, tmp_path: Path):
        nested = str(tmp_path / "a" / "b" / "c")
        save_probe_cache({"test": {"tc": 1.0}}, nested)
        assert load_probe_cache(nested) == {"test": {"tc": 1.0}}


# ---------------------------------------------------------------------------
# probe_or_cache
# ---------------------------------------------------------------------------


class TestProbeOrCache:

    @pytest.mark.asyncio
    async def test_uses_cache_when_available(self, tmp_path: Path):
        cache = {"test-model": {"tool_calling": 1.0, "structured_output": 0.5}}
        save_probe_cache(cache, str(tmp_path))

        # Client should NOT be called
        client = FakeChatClient(raise_error=True)
        results = await probe_or_cache(client, "test-model", str(tmp_path))
        assert results == cache["test-model"]

    @pytest.mark.asyncio
    async def test_probes_and_caches_when_not_cached(self, tmp_path: Path):
        client = FakeChatClient(responses=[
            _resp(tool_calls=[_tc()]),
            _resp(content='{"name": "A"}'),
            _resp(content="Paris"),
        ])

        results = await probe_or_cache(client, "new-model", str(tmp_path))

        assert "tool_calling" in results
        assert results["tool_calling"] == 1.0

        # Results should be saved to cache
        cached = load_probe_cache(str(tmp_path))
        assert "new-model" in cached
        assert cached["new-model"]["tool_calling"] == 1.0

    @pytest.mark.asyncio
    async def test_different_models_cached_separately(self, tmp_path: Path):
        # Pre-cache model-a
        cache = {"model-a": {"tool_calling": 0.0}}
        save_probe_cache(cache, str(tmp_path))

        # Probe model-b
        client = FakeChatClient(responses=[
            _resp(tool_calls=[_tc()]),
            _resp(content='{"name": "A"}'),
            _resp(content="Paris"),
        ])
        await probe_or_cache(client, "model-b", str(tmp_path))

        # Both should be in cache
        cached = load_probe_cache(str(tmp_path))
        assert "model-a" in cached
        assert "model-b" in cached
        assert cached["model-a"]["tool_calling"] == 0.0
        assert cached["model-b"]["tool_calling"] == 1.0
