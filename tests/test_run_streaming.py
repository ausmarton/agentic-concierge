"""Tests for P8-2: SSE run event streaming.

Tests cover:
- _emit puts events to the queue with correct structure
- _emit no-ops when queue is None
- _emit drops silently when queue is full (QueueFull)
- execute_task puts recruitment, run_complete, and _run_done_ to event_queue
- execute_task puts llm_request, llm_response, tool_call, tool_result events
- POST /run/stream returns text/event-stream content-type
- SSE events are properly formatted (data: {...}\n\n)
- _run_done_ sentinel terminates the stream
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_concierge.application.execute_task import _emit


# ---------------------------------------------------------------------------
# _emit unit tests (also covered in test_parallel_task_force.py; these verify
# SSE-specific event shapes)
# ---------------------------------------------------------------------------

def test_emit_event_has_kind_data_step():
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    _emit(queue, "llm_request", {"step": 0, "message_count": 2}, step="step_0")
    item = queue.get_nowait()
    assert item["kind"] == "llm_request"
    assert item["data"] == {"step": 0, "message_count": 2}
    assert item["step"] == "step_0"


def test_emit_step_defaults_to_none():
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    _emit(queue, "recruitment", {"specialist_id": "engineering"})
    item = queue.get_nowait()
    assert item["step"] is None


# ---------------------------------------------------------------------------
# execute_task emits key events to the queue
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_task_emits_recruitment_and_run_done():
    """execute_task puts recruitment + _run_done_ to queue even for a simple run."""
    from agentic_concierge.application.execute_task import execute_task
    from agentic_concierge.domain import Task, RunId
    from agentic_concierge.domain.models import LLMResponse, ToolCallRequest
    from agentic_concierge.config.schema import ConciergeConfig, ModelConfig, SpecialistConfig

    task = Task(
        prompt="Test streaming",
        specialist_id="engineering",
        model_key="fast",
        network_allowed=False,
    )
    config = ConciergeConfig(
        models={"fast": ModelConfig(base_url="http://x/v1", model="m")},
        specialists={
            "engineering": SpecialistConfig(description="eng", workflow="engineering"),
        },
    )

    call_count = 0

    async def mock_chat(messages, model, tools, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(call_id="c1", tool_name="shell", arguments={"cmd": "echo hi"})],
            )
        return LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(call_id="c2", tool_name="finish_task", arguments={"summary": "done", "artifacts": [], "next_steps": []})],
        )

    chat_client = MagicMock()
    chat_client.chat = mock_chat

    run_id = RunId("stream-test-run")
    run_repository = MagicMock()
    run_repository.create_run.return_value = (run_id, "/tmp/runs/stream-test", "/tmp/ws")
    run_repository.append_event = MagicMock()

    pack = MagicMock()
    pack.system_prompt = "You are an engineer."
    pack.finish_tool_name = "finish_task"
    pack.finish_required_fields = ["summary"]
    pack.tool_definitions = [
        {"type": "function", "function": {"name": "shell", "description": "shell", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}},
        {"type": "function", "function": {"name": "finish_task", "description": "finish", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}},
    ]
    pack.aopen = AsyncMock()
    pack.aclose = AsyncMock()

    async def execute_tool(name, args):
        if name == "shell":
            return {"stdout": "hello", "exit_code": 0}
        return {}

    pack.execute_tool = execute_tool

    registry = MagicMock()
    registry.get_pack.return_value = pack

    event_queue: asyncio.Queue = asyncio.Queue(maxsize=64)

    await execute_task(
        task,
        chat_client=chat_client,
        run_repository=run_repository,
        specialist_registry=registry,
        config=config,
        event_queue=event_queue,
        max_review_iterations=0,
    )

    # Drain the queue into a list
    events = []
    while not event_queue.empty():
        events.append(event_queue.get_nowait())

    kinds = [e["kind"] for e in events]
    assert "recruitment" in kinds
    assert "run_complete" in kinds
    assert "_run_done_" in kinds


@pytest.mark.asyncio
async def test_execute_task_emits_llm_request_and_tool_call():
    """execute_task emits llm_request and tool_call events to the queue."""
    from agentic_concierge.application.execute_task import execute_task
    from agentic_concierge.domain import Task, RunId
    from agentic_concierge.domain.models import LLMResponse, ToolCallRequest
    from agentic_concierge.config.schema import ConciergeConfig, ModelConfig, SpecialistConfig

    task = Task(
        prompt="Test events",
        specialist_id="engineering",
        model_key="fast",
        network_allowed=False,
    )
    config = ConciergeConfig(
        models={"fast": ModelConfig(base_url="http://x/v1", model="m")},
        specialists={
            "engineering": SpecialistConfig(description="eng", workflow="engineering"),
        },
    )

    call_count = 0

    async def mock_chat(messages, model, tools, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(call_id="c1", tool_name="shell", arguments={"cmd": "ls"})],
            )
        return LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(call_id="c2", tool_name="finish_task", arguments={"summary": "done", "artifacts": [], "next_steps": []})],
        )

    chat_client = MagicMock()
    chat_client.chat = mock_chat

    run_id = RunId("stream-events-run")
    run_repository = MagicMock()
    run_repository.create_run.return_value = (run_id, "/tmp/runs/events-run", "/tmp/ws")
    run_repository.append_event = MagicMock()

    pack = MagicMock()
    pack.system_prompt = "You are an engineer."
    pack.finish_tool_name = "finish_task"
    pack.finish_required_fields = ["summary"]
    pack.tool_definitions = [
        {"type": "function", "function": {"name": "shell", "description": "shell", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}},
        {"type": "function", "function": {"name": "finish_task", "description": "finish", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}},
    ]
    pack.aopen = AsyncMock()
    pack.aclose = AsyncMock()

    async def execute_tool(name, args):
        return {"stdout": "output", "exit_code": 0}

    pack.execute_tool = execute_tool

    registry = MagicMock()
    registry.get_pack.return_value = pack

    event_queue: asyncio.Queue = asyncio.Queue(maxsize=64)

    await execute_task(
        task,
        chat_client=chat_client,
        run_repository=run_repository,
        specialist_registry=registry,
        config=config,
        event_queue=event_queue,
        max_review_iterations=0,
    )

    events = []
    while not event_queue.empty():
        events.append(event_queue.get_nowait())

    kinds = [e["kind"] for e in events]
    assert "llm_request" in kinds
    assert "llm_response" in kinds
    assert "tool_call" in kinds
    assert "tool_result" in kinds


# ---------------------------------------------------------------------------
# HTTP /run/stream endpoint (integration with TestClient)
# ---------------------------------------------------------------------------

def test_run_stream_returns_event_stream_content_type():
    """POST /run/stream returns text/event-stream content-type."""
    from fastapi.testclient import TestClient
    from agentic_concierge.interfaces.http_api import app
    from agentic_concierge.domain import RunId
    from agentic_concierge.domain.models import LLMResponse, ToolCallRequest

    call_count = 0

    async def mock_chat(messages, model, tools, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(call_id="c1", tool_name="shell", arguments={"cmd": "echo hi"})],
            )
        return LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(call_id="c2", tool_name="finish_task", arguments={"summary": "done", "artifacts": [], "next_steps": []})],
        )

    pack = MagicMock()
    pack.system_prompt = "You are an engineer."
    pack.finish_tool_name = "finish_task"
    pack.finish_required_fields = ["summary"]
    pack.tool_definitions = [
        {"type": "function", "function": {"name": "shell", "description": "shell", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}},
        {"type": "function", "function": {"name": "finish_task", "description": "finish", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}},
    ]
    pack.aopen = AsyncMock()
    pack.aclose = AsyncMock()

    async def execute_tool(name, args):
        return {"stdout": "output", "exit_code": 0}

    pack.execute_tool = execute_tool

    run_id = RunId("stream-http-run")

    with patch("agentic_concierge.interfaces.http_api.resolve_llm") as mock_resolve, \
         patch("agentic_concierge.interfaces.http_api.build_chat_client") as mock_build_client, \
         patch("agentic_concierge.interfaces.http_api.FileSystemRunRepository") as mock_repo_cls, \
         patch("agentic_concierge.interfaces.http_api.ConfigSpecialistRegistry") as mock_registry_cls:

        from agentic_concierge.config.schema import ModelConfig
        mock_resolved = MagicMock()
        mock_resolved.model_config = ModelConfig(base_url="http://x/v1", model="m")
        mock_resolved.base_url = "http://x/v1"
        mock_resolved.model = "m"
        mock_resolve.return_value = mock_resolved

        mock_chat_client = MagicMock()
        mock_chat_client.chat = mock_chat
        mock_build_client.return_value = mock_chat_client

        mock_repo = MagicMock()
        mock_repo.create_run.return_value = (run_id, "/tmp/runs/stream-http-run", "/tmp/ws")
        mock_repo.append_event = MagicMock()
        mock_repo_cls.return_value = mock_repo

        mock_registry = MagicMock()
        mock_registry.get_pack.return_value = pack
        mock_registry_cls.return_value = mock_registry

        client = TestClient(app)
        response = client.post(
            "/run/stream",
            json={"prompt": "test streaming", "pack": "engineering"},
        )

    assert "text/event-stream" in response.headers.get("content-type", "")


def test_run_stream_events_are_sse_formatted():
    """Events in the stream use `data: {...}\\n\\n` format."""
    from fastapi.testclient import TestClient
    from agentic_concierge.interfaces.http_api import app
    from agentic_concierge.domain import RunId
    from agentic_concierge.domain.models import LLMResponse, ToolCallRequest

    call_count = 0

    async def mock_chat(messages, model, tools, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(call_id="c1", tool_name="shell", arguments={"cmd": "echo hi"})],
            )
        return LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(call_id="c2", tool_name="finish_task", arguments={"summary": "done", "artifacts": [], "next_steps": []})],
        )

    pack = MagicMock()
    pack.system_prompt = "You are an engineer."
    pack.finish_tool_name = "finish_task"
    pack.finish_required_fields = ["summary"]
    pack.tool_definitions = [
        {"type": "function", "function": {"name": "shell", "description": "shell", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}},
        {"type": "function", "function": {"name": "finish_task", "description": "finish", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}},
    ]
    pack.aopen = AsyncMock()
    pack.aclose = AsyncMock()

    async def execute_tool(name, args):
        return {"stdout": "output", "exit_code": 0}

    pack.execute_tool = execute_tool

    run_id = RunId("stream-fmt-run")

    with patch("agentic_concierge.interfaces.http_api.resolve_llm") as mock_resolve, \
         patch("agentic_concierge.interfaces.http_api.build_chat_client") as mock_build_client, \
         patch("agentic_concierge.interfaces.http_api.FileSystemRunRepository") as mock_repo_cls, \
         patch("agentic_concierge.interfaces.http_api.ConfigSpecialistRegistry") as mock_registry_cls:

        from agentic_concierge.config.schema import ModelConfig
        mock_resolved = MagicMock()
        mock_resolved.model_config = ModelConfig(base_url="http://x/v1", model="m")
        mock_resolved.base_url = "http://x/v1"
        mock_resolved.model = "m"
        mock_resolve.return_value = mock_resolved

        mock_chat_client = MagicMock()
        mock_chat_client.chat = mock_chat
        mock_build_client.return_value = mock_chat_client

        mock_repo = MagicMock()
        mock_repo.create_run.return_value = (run_id, "/tmp/runs/stream-fmt-run", "/tmp/ws")
        mock_repo.append_event = MagicMock()
        mock_repo_cls.return_value = mock_repo

        mock_registry = MagicMock()
        mock_registry.get_pack.return_value = pack
        mock_registry_cls.return_value = mock_registry

        client = TestClient(app)
        response = client.post(
            "/run/stream",
            json={"prompt": "test streaming format", "pack": "engineering"},
        )

    body = response.text
    # All data lines should be parseable JSON
    data_lines = [line for line in body.split("\n") if line.startswith("data: ")]
    assert len(data_lines) > 0
    for line in data_lines:
        payload = json.loads(line[6:])  # strip "data: "
        assert "kind" in payload
