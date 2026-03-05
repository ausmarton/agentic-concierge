"""Tests for FallbackPolicy and FallbackChatClient (P6-4).

All tests are mocked — no real cloud call required.  Fast CI safe.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_concierge.domain import LLMResponse, ToolCallRequest
from agentic_concierge.infrastructure.chat.fallback import FallbackChatClient, FallbackPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _response(*, content: str = "", tool_calls: list | None = None) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=tool_calls or [])


def _tool_call(name: str = "shell", args: dict | None = None) -> ToolCallRequest:
    return ToolCallRequest(call_id="c1", tool_name=name, arguments=args or {"cmd": ["ls"]})


def _make_client(response: LLMResponse) -> AsyncMock:
    mock = AsyncMock()
    mock.chat = AsyncMock(return_value=response)
    return mock


def _make_fallback(
    local_response: LLMResponse,
    cloud_response: LLMResponse,
    policy_mode: str = "no_tool_calls",
    cloud_model: str = "gpt-4o",
) -> FallbackChatClient:
    local = _make_client(local_response)
    cloud = _make_client(cloud_response)
    policy = FallbackPolicy(policy_mode)
    return FallbackChatClient(local, cloud, cloud_model, policy)


# ---------------------------------------------------------------------------
# FallbackPolicy — no_tool_calls
# ---------------------------------------------------------------------------


def test_no_tool_calls_policy_triggers_when_no_tool_calls():
    policy = FallbackPolicy("no_tool_calls")
    resp = _response(content="I'm done.", tool_calls=[])
    assert policy.evaluate(resp) == "no_tool_calls"


def test_no_tool_calls_policy_does_not_trigger_with_tool_calls():
    policy = FallbackPolicy("no_tool_calls")
    resp = _response(tool_calls=[_tool_call()])
    assert policy.evaluate(resp) is None


# ---------------------------------------------------------------------------
# FallbackPolicy — malformed_args
# ---------------------------------------------------------------------------


def test_malformed_args_policy_triggers_on_raw_key():
    policy = FallbackPolicy("malformed_args")
    tc = ToolCallRequest(call_id="c1", tool_name="shell", arguments={"_raw": "ls"})
    resp = _response(tool_calls=[tc])
    assert policy.evaluate(resp) == "malformed_args"


def test_malformed_args_policy_no_trigger_for_clean_args():
    policy = FallbackPolicy("malformed_args")
    resp = _response(tool_calls=[_tool_call()])
    assert policy.evaluate(resp) is None


def test_malformed_args_policy_no_trigger_when_no_tool_calls():
    """malformed_args policy: plain text response is not a trigger (wrong mode)."""
    policy = FallbackPolicy("malformed_args")
    resp = _response(content="done")
    assert policy.evaluate(resp) is None


# ---------------------------------------------------------------------------
# FallbackPolicy — always
# ---------------------------------------------------------------------------


def test_always_policy_triggers_on_any_response():
    policy = FallbackPolicy("always")
    assert policy.evaluate(_response(content="ok")) == "always"
    assert policy.evaluate(_response(tool_calls=[_tool_call()])) == "always"


# ---------------------------------------------------------------------------
# FallbackPolicy — unknown / safe default
# ---------------------------------------------------------------------------


def test_unknown_policy_mode_never_triggers():
    """An unrecognised policy mode never triggers fallback — safe default."""
    policy = FallbackPolicy("unicorn")
    assert policy.evaluate(_response()) is None
    assert policy.evaluate(_response(tool_calls=[_tool_call()])) is None


def test_policy_mode_property():
    policy = FallbackPolicy("no_tool_calls")
    assert policy.mode == "no_tool_calls"


# ---------------------------------------------------------------------------
# FallbackChatClient — local response returned when policy not triggered
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_response_returned_when_policy_not_triggered():
    """When the local model calls tools (policy not triggered), local response returned."""
    local_resp = _response(tool_calls=[_tool_call()])
    cloud_resp = _response(tool_calls=[_tool_call(name="other")])
    client = _make_fallback(local_resp, cloud_resp, policy_mode="no_tool_calls")

    result = await client.chat([{"role": "user", "content": "go"}], model="llama3:8b")
    assert result is local_resp
    # Cloud was NOT called.
    client._cloud.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_cloud_response_returned_when_policy_triggered():
    """When local response has no tool calls (policy triggers), cloud response returned."""
    local_resp = _response(content="I think the answer is 42.")
    cloud_resp = _response(tool_calls=[_tool_call()])
    client = _make_fallback(local_resp, cloud_resp, policy_mode="no_tool_calls")

    result = await client.chat([{"role": "user", "content": "go"}], model="llama3:8b")
    assert result is cloud_resp
    # Both local and cloud were called.
    client._local.chat.assert_awaited_once()
    client._cloud.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_cloud_called_with_cloud_model_name_not_local():
    """Cloud client is called with cloud_model, not with the local model name."""
    local_resp = _response(content="plain")
    cloud_resp = _response(tool_calls=[_tool_call()])
    client = _make_fallback(
        local_resp, cloud_resp, policy_mode="no_tool_calls", cloud_model="gpt-4o-mini"
    )

    await client.chat([{"role": "user", "content": "go"}], model="llama3:8b")

    cloud_call_args = client._cloud.chat.call_args
    # model is the second positional arg
    assert cloud_call_args[0][1] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_cloud_called_with_same_messages_and_kwargs():
    """Cloud is called with the same messages, tools, temperature, top_p, max_tokens."""
    messages = [{"role": "user", "content": "test"}]
    tools = [{"type": "function", "function": {"name": "shell"}}]
    local_resp = _response(content="plain")
    cloud_resp = _response(tool_calls=[_tool_call()])
    client = _make_fallback(local_resp, cloud_resp, policy_mode="no_tool_calls")

    await client.chat(
        messages, model="local", tools=tools, temperature=0.2, top_p=0.8, max_tokens=512
    )

    cloud_call = client._cloud.chat.call_args
    assert cloud_call[0][0] is messages
    assert cloud_call[1]["tools"] is tools
    assert cloud_call[1]["temperature"] == 0.2
    assert cloud_call[1]["top_p"] == 0.8
    assert cloud_call[1]["max_tokens"] == 512


# ---------------------------------------------------------------------------
# FallbackChatClient — event queue
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_event_queued_on_trigger():
    """A cloud_fallback event is queued when the policy triggers."""
    local_resp = _response(content="plain")
    cloud_resp = _response(tool_calls=[_tool_call()])
    client = _make_fallback(local_resp, cloud_resp, policy_mode="no_tool_calls")

    await client.chat([{"role": "user", "content": "go"}], model="llama3:8b")
    events = client.pop_events()

    assert len(events) == 1
    assert events[0]["reason"] == "no_tool_calls"
    assert events[0]["local_model"] == "llama3:8b"
    assert events[0]["cloud_model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_no_event_queued_when_not_triggered():
    """pop_events() returns [] when the policy was not triggered."""
    local_resp = _response(tool_calls=[_tool_call()])
    cloud_resp = _response(tool_calls=[_tool_call()])
    client = _make_fallback(local_resp, cloud_resp, policy_mode="no_tool_calls")

    await client.chat([{"role": "user", "content": "go"}], model="llama3:8b")
    events = client.pop_events()

    assert events == []


@pytest.mark.asyncio
async def test_pop_events_drains_queue():
    """pop_events() clears the queue — second call returns []."""
    local_resp = _response(content="plain")
    cloud_resp = _response(tool_calls=[_tool_call()])
    client = _make_fallback(local_resp, cloud_resp, policy_mode="no_tool_calls")

    await client.chat([{"role": "user", "content": "go"}], model="m")
    first = client.pop_events()
    second = client.pop_events()

    assert len(first) == 1
    assert second == []


@pytest.mark.asyncio
async def test_multiple_calls_accumulate_events_until_popped():
    """Events from multiple calls accumulate until pop_events() is called."""
    local_resp = _response(content="plain")
    cloud_resp = _response(tool_calls=[_tool_call()])
    client = _make_fallback(local_resp, cloud_resp, policy_mode="always")

    await client.chat([{"role": "user", "content": "a"}], model="m")
    await client.chat([{"role": "user", "content": "b"}], model="m")
    events = client.pop_events()

    assert len(events) == 2


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


def test_cloud_fallback_defaults_to_none():
    """ConciergeConfig.cloud_fallback is None by default."""
    from agentic_concierge.config import load_config
    config = load_config()
    assert config.cloud_fallback is None


def test_cloud_fallback_config_valid():
    """A valid CloudFallbackConfig is accepted."""
    from agentic_concierge.config.schema import (
        CloudFallbackConfig, ConciergeConfig, ModelConfig, SpecialistConfig,
    )
    cfg = ConciergeConfig(
        models={
            "quality": ModelConfig(base_url="http://localhost:11434/v1", model="llama3:8b"),
            "cloud": ModelConfig(
                base_url="https://api.openai.com/v1", model="gpt-4o", backend="generic"
            ),
        },
        specialists={
            "engineering": SpecialistConfig(description="eng", keywords=[], workflow="engineering"),
        },
        cloud_fallback=CloudFallbackConfig(model_key="cloud", policy="no_tool_calls"),
    )
    assert cfg.cloud_fallback is not None
    assert cfg.cloud_fallback.model_key == "cloud"
    assert cfg.cloud_fallback.policy == "no_tool_calls"


def test_cloud_fallback_default_policy_is_no_tool_calls():
    from agentic_concierge.config.schema import CloudFallbackConfig
    cfg = CloudFallbackConfig(model_key="cloud")
    assert cfg.policy == "no_tool_calls"


# ---------------------------------------------------------------------------
# execute_task integration: cloud_fallback event in runlog
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_task_logs_cloud_fallback_event_in_runlog(tmp_path):
    """When FallbackChatClient is used, cloud_fallback events appear in the runlog."""
    from pathlib import Path

    from agentic_concierge.application.execute_task import execute_task
    from agentic_concierge.config.schema import ConciergeConfig, ModelConfig, SpecialistConfig
    from agentic_concierge.domain import Task, LLMResponse, ToolCallRequest, RunResult
    from agentic_concierge.infrastructure.chat.fallback import FallbackChatClient, FallbackPolicy

    config = ConciergeConfig(
        models={"quality": ModelConfig(base_url="http://localhost:11434/v1", model="local-m")},
        specialists={
            "engineering": SpecialistConfig(
                description="eng", keywords=[], workflow="engineering"
            )
        },
    )

    # Stub pack
    class _Pack:
        specialist_id = "engineering"
        system_prompt = "sys"
        finish_tool_name = "finish_task"
        finish_required_fields = ["summary"]
        tool_definitions = [
            {"type": "function", "function": {"name": "shell"}},
            {"type": "function", "function": {
                "name": "finish_task",
                "parameters": {"properties": {"summary": {}}, "required": ["summary"]},
            }},
        ]
        async def execute_tool(self, name, args): return {"stdout": "ok"}
        async def aopen(self): pass
        async def aclose(self): pass

    # Local mock: returns plain text (no tool calls) — policy triggers
    # Cloud mock: returns shell call, then finish_task
    local_call_count = 0
    cloud_call_count = 0

    async def _local_chat(*args, **kwargs):
        nonlocal local_call_count
        local_call_count += 1
        return LLMResponse(content="I think the answer is done", tool_calls=[])

    async def _cloud_chat(*args, **kwargs):
        nonlocal cloud_call_count
        cloud_call_count += 1
        if cloud_call_count == 1:
            return LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(call_id="c1", tool_name="shell", arguments={"cmd": ["ls"]})],
            )
        return LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(call_id="c2", tool_name="finish_task", arguments={
                "summary": "done", "artifacts": [], "next_steps": [], "notes": "",
            })],
        )

    local_client = MagicMock()
    local_client.chat = _local_chat
    cloud_client = MagicMock()
    cloud_client.chat = _cloud_chat

    policy = FallbackPolicy("no_tool_calls")
    chat_client = FallbackChatClient(local_client, cloud_client, "gpt-4o", policy)

    run_dir_path = str(tmp_path / "runs" / "test-run")
    Path(run_dir_path).mkdir(parents=True)

    events_logged: list = []

    class _Repo:
        def create_run(self):
            from agentic_concierge.domain import RunId
            return RunId("test-run"), run_dir_path, str(tmp_path / "workspace")

        def append_event(self, run_id, kind, payload, step=None):
            events_logged.append({"kind": kind, "payload": payload, "step": step})

    class _Registry:
        def get_pack(self, sid, ws, net, **kw): return _Pack()

    task = Task(prompt="do something", specialist_id="engineering")
    await execute_task(
        task,
        chat_client=chat_client,
        run_repository=_Repo(),
        specialist_registry=_Registry(),
        config=config,
        max_review_iterations=0,
    )

    fallback_events = [e for e in events_logged if e["kind"] == "cloud_fallback"]
    assert fallback_events, "Expected at least one cloud_fallback event in runlog"
    ev = fallback_events[0]["payload"]
    assert ev["reason"] == "no_tool_calls"
    assert ev["local_model"] == "local-m"
    assert ev["cloud_model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_execute_task_auto_wraps_when_cloud_fallback_configured(tmp_path):
    """When config.cloud_fallback is set, execute_task wraps chat_client automatically."""
    from pathlib import Path
    from unittest.mock import patch as _patch

    from agentic_concierge.application.execute_task import execute_task
    from agentic_concierge.config.schema import (
        CloudFallbackConfig, ConciergeConfig, ModelConfig, SpecialistConfig,
    )
    from agentic_concierge.domain import Task, LLMResponse, ToolCallRequest
    from agentic_concierge.infrastructure.chat.fallback import FallbackChatClient

    config = ConciergeConfig(
        models={
            "quality": ModelConfig(base_url="http://localhost:11434/v1", model="local-m"),
            "cloud": ModelConfig(
                base_url="https://api.openai.com/v1", model="gpt-4o", backend="generic"
            ),
        },
        specialists={
            "engineering": SpecialistConfig(
                description="eng", keywords=[], workflow="engineering"
            )
        },
        cloud_fallback=CloudFallbackConfig(model_key="cloud", policy="always"),
    )

    class _Pack:
        specialist_id = "engineering"
        system_prompt = "sys"
        finish_tool_name = "finish_task"
        finish_required_fields = ["summary"]
        tool_definitions = [
            {"type": "function", "function": {"name": "shell"}},
            {"type": "function", "function": {
                "name": "finish_task",
                "parameters": {"properties": {"summary": {}}, "required": ["summary"]},
            }},
        ]
        async def execute_tool(self, name, args): return {"stdout": "ok"}
        async def aopen(self): pass
        async def aclose(self): pass

    call_count = 0

    async def _cloud_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMResponse(content=None, tool_calls=[
                ToolCallRequest(call_id="c1", tool_name="shell", arguments={"cmd": ["ls"]}),
            ])
        return LLMResponse(content=None, tool_calls=[
            ToolCallRequest(call_id="c2", tool_name="finish_task", arguments={
                "summary": "done", "artifacts": [], "next_steps": [], "notes": "",
            }),
        ])

    mock_local = MagicMock()
    # Local returns plain text — "always" policy will override regardless
    mock_local.chat = AsyncMock(return_value=LLMResponse(content="unused", tool_calls=[]))

    mock_cloud = MagicMock()
    mock_cloud.chat = _cloud_chat

    def _fake_build_chat_client(model_cfg):
        # Returns mock_cloud for the cloud config, mock_local for others
        if model_cfg.model == "gpt-4o":
            return mock_cloud
        return mock_local

    run_dir_path = str(tmp_path / "runs" / "test-run")
    Path(run_dir_path).mkdir(parents=True)

    events_logged: list = []

    class _Repo:
        def create_run(self):
            from agentic_concierge.domain import RunId
            return RunId("test-run"), run_dir_path, str(tmp_path / "workspace")
        def append_event(self, run_id, kind, payload, step=None):
            events_logged.append({"kind": kind, "payload": payload})

    class _Registry:
        def get_pack(self, sid, ws, net, **kw): return _Pack()

    task = Task(prompt="do something", specialist_id="engineering")

    with _patch(
        "agentic_concierge.infrastructure.chat.build_chat_client",
        side_effect=_fake_build_chat_client,
    ):
        await execute_task(
            task,
            chat_client=mock_local,   # injected local client
            run_repository=_Repo(),
            specialist_registry=_Registry(),
            config=config,
            max_review_iterations=0,
        )

    # With policy="always", every LLM call goes to cloud.
    # Verify cloud_fallback events appear in runlog.
    fallback_events = [e for e in events_logged if e["kind"] == "cloud_fallback"]
    assert fallback_events, "Expected cloud_fallback events when config.cloud_fallback is set"
    assert fallback_events[0]["payload"]["reason"] == "always"
    assert fallback_events[0]["payload"]["cloud_model"] == "gpt-4o"
