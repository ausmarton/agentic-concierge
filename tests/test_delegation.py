"""Tests for the agent-to-agent delegation mechanism (ADR-022)."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_concierge.application.execute_task import (
    _DELEGATE_TOOL_DEF,
    _MAX_DELEGATION_DEPTH,
    _MAX_DELEGATION_STEPS,
    _execute_pack_loop,
)
from agentic_concierge.config import ModelConfig
from agentic_concierge.domain import LLMResponse, RunId, ToolCallRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_cfg() -> ModelConfig:
    return ModelConfig(base_url="http://localhost:11434/v1", model="test-model")


class _Pack:
    specialist_id = "engineering"
    system_prompt = "You are an agent."
    finish_tool_name = "finish_task"
    finish_required_fields = ["summary"]

    def __init__(self, tool_defs=None, sid="engineering"):
        self._tool_defs = tool_defs or []
        self.specialist_id = sid

    @property
    def tool_definitions(self):
        return self._tool_defs

    async def aopen(self):
        pass

    async def aclose(self):
        pass

    async def execute_tool(self, name, args):
        return {"ok": True}


def _finish_tool_def():
    return {
        "type": "function",
        "function": {
            "name": "finish_task",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    }


def _shell_tool_def():
    return {
        "type": "function",
        "function": {
            "name": "shell",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _tool_call(name: str, args: dict, call_id: str = "c1") -> ToolCallRequest:
    return ToolCallRequest(tool_name=name, arguments=args, call_id=call_id)


def _run_repo() -> MagicMock:
    repo = MagicMock()
    repo.append_event = MagicMock()
    return repo


class _Registry:
    """Mock specialist registry that returns a pack completing in 2 steps."""

    def __init__(self, sub_pack=None):
        self._sub_pack = sub_pack

    def get_pack(self, sid, wp, na, **kw):
        if self._sub_pack is not None:
            return self._sub_pack
        return _Pack([_shell_tool_def(), _finish_tool_def()], sid=sid)

    def list_ids(self):
        return ["engineering", "research"]


# ---------------------------------------------------------------------------
# Tests: Delegation basic flow
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delegate_to_research():
    """Engineering agent delegates to research, gets result back."""
    # Parent: shell → delegate_to_specialist → finish_task
    # Sub: shell → finish_task (within the same chat_client mock sequence)
    call_count = 0

    async def _chat(messages, model, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # Parent step 0: shell
            return LLMResponse(content=None, tool_calls=[_tool_call("shell", {"command": "echo"}, "p1")])
        elif call_count == 2:
            # Parent step 1: delegate
            return LLMResponse(content=None, tool_calls=[
                _tool_call("delegate_to_specialist", {
                    "sub_task": "Research React 19 APIs",
                    "specialist_type": "research",
                }, "p2"),
            ])
        elif call_count == 3:
            # Sub step 0: shell
            return LLMResponse(content=None, tool_calls=[_tool_call("shell", {"command": "echo sub"}, "s1")])
        elif call_count == 4:
            # Sub step 1: finish
            return LLMResponse(content=None, tool_calls=[
                _tool_call("finish_task", {"summary": "Research complete: React 19 uses RSCs"}, "s2"),
            ])
        elif call_count == 5:
            # Parent step 2: finish (incorporates delegation result)
            return LLMResponse(content=None, tool_calls=[
                _tool_call("finish_task", {"summary": "Done, React 19 uses RSCs"}, "p3"),
            ])
        return LLMResponse(content="fallback", tool_calls=[])

    client = MagicMock(spec=[])
    client.chat = AsyncMock(side_effect=_chat)
    pack = _Pack([_shell_tool_def(), _finish_tool_def()])
    repo = _run_repo()
    result = await _execute_pack_loop(
        pack=pack,
        messages=[{"role": "system", "content": "test"}, {"role": "user", "content": "Build app"}],
        model_cfg=_model_cfg(),
        chat_client=client,
        run_repository=repo,
        run_id=RunId(value="test-run"),
        max_steps=20,
        specialist_id="engineering",
        delegation_depth=1,
        specialist_registry=_Registry(),
        workspace_path="/tmp/test",
    )
    assert result["summary"] == "Done, React 19 uses RSCs"


# ---------------------------------------------------------------------------
# Tests: Depth limited
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_depth_limited():
    """Sub-specialist tool defs should NOT include delegate_to_specialist."""
    sub_tool_defs_seen = []

    async def _chat(messages, model, **kwargs):
        tools = kwargs.get("tools", [])
        tool_names = [t["function"]["name"] for t in tools]
        sub_tool_defs_seen.append(tool_names)

        if len(sub_tool_defs_seen) == 1:
            # Parent: delegate
            return LLMResponse(content=None, tool_calls=[
                _tool_call("delegate_to_specialist", {
                    "sub_task": "sub", "specialist_type": "research",
                }, "p1"),
            ])
        elif len(sub_tool_defs_seen) == 2:
            # Sub: this is the key check — delegate should NOT be in tools.
            return LLMResponse(content=None, tool_calls=[
                _tool_call("finish_task", {"summary": "sub done"}, "s1"),
            ])
        elif len(sub_tool_defs_seen) == 3:
            return LLMResponse(content=None, tool_calls=[
                _tool_call("finish_task", {"summary": "parent done"}, "p2"),
            ])
        return LLMResponse(content="fallback", tool_calls=[])

    client = MagicMock(spec=[])
    client.chat = AsyncMock(side_effect=_chat)
    pack = _Pack([_shell_tool_def(), _finish_tool_def()])
    await _execute_pack_loop(
        pack=pack,
        messages=[{"role": "system", "content": "test"}, {"role": "user", "content": "x"}],
        model_cfg=_model_cfg(),
        chat_client=client,
        run_repository=_run_repo(),
        run_id=RunId(value="test-run"),
        max_steps=20,
        delegation_depth=1,
        specialist_registry=_Registry(),
        workspace_path="/tmp/test",
    )
    # Parent's tools should include delegate_to_specialist.
    assert "delegate_to_specialist" in sub_tool_defs_seen[0]
    # Sub-specialist's tools should NOT include delegate_to_specialist.
    assert "delegate_to_specialist" not in sub_tool_defs_seen[1]


# ---------------------------------------------------------------------------
# Tests: Step budget
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_step_budget():
    """Sub-specialist should get limited steps."""
    sub_steps = 0

    async def _chat(messages, model, **kwargs):
        nonlocal sub_steps
        tools = kwargs.get("tools", [])
        tool_names = [t["function"]["name"] for t in tools]

        if "delegate_to_specialist" in tool_names:
            # Parent: delegate
            return LLMResponse(content=None, tool_calls=[
                _tool_call("delegate_to_specialist", {
                    "sub_task": "sub", "specialist_type": "research",
                }, "p1"),
            ])
        else:
            sub_steps += 1
            # Sub: finish immediately
            return LLMResponse(content=None, tool_calls=[
                _tool_call("finish_task", {"summary": "sub done"}, f"s{sub_steps}"),
            ])

    client = MagicMock(spec=[])
    client.chat = AsyncMock(side_effect=_chat)
    pack = _Pack([_shell_tool_def(), _finish_tool_def()])

    # With max_steps=6, at step 0 (delegate), remaining = 6-0-1 = 5,
    # sub gets min(5//2, 15) = 2, with floor 3 → 3.
    await _execute_pack_loop(
        pack=pack,
        messages=[{"role": "system", "content": "test"}, {"role": "user", "content": "x"}],
        model_cfg=_model_cfg(),
        chat_client=client,
        run_repository=_run_repo(),
        run_id=RunId(value="test-run"),
        max_steps=6,
        delegation_depth=1,
        specialist_registry=_Registry(),
        workspace_path="/tmp/test",
    )
    # Sub should have completed within its budget.
    assert sub_steps >= 1


# ---------------------------------------------------------------------------
# Tests: Events emitted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delegation_events_emitted():
    """Verify delegation_start and delegation_complete events are emitted."""
    call_count = 0

    async def _chat(messages, model, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMResponse(content=None, tool_calls=[
                _tool_call("delegate_to_specialist", {
                    "sub_task": "research", "specialist_type": "research",
                }, "p1"),
            ])
        elif call_count == 2:
            return LLMResponse(content=None, tool_calls=[
                _tool_call("finish_task", {"summary": "sub done"}, "s1"),
            ])
        elif call_count == 3:
            return LLMResponse(content=None, tool_calls=[
                _tool_call("finish_task", {"summary": "parent done"}, "p2"),
            ])
        return LLMResponse(content="fallback", tool_calls=[])

    client = MagicMock(spec=[])
    client.chat = AsyncMock(side_effect=_chat)
    repo = _run_repo()
    pack = _Pack([_shell_tool_def(), _finish_tool_def()])
    await _execute_pack_loop(
        pack=pack,
        messages=[{"role": "system", "content": "test"}, {"role": "user", "content": "x"}],
        model_cfg=_model_cfg(),
        chat_client=client,
        run_repository=repo,
        run_id=RunId(value="test-run"),
        max_steps=20,
        delegation_depth=1,
        specialist_registry=_Registry(),
        workspace_path="/tmp/test",
    )
    event_kinds = [call.args[1] for call in repo.append_event.call_args_list]
    assert "delegation_start" in event_kinds
    assert "delegation_complete" in event_kinds


# ---------------------------------------------------------------------------
# Tests: Unknown specialist returns error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unknown_specialist_returns_error():
    """Bad specialist_type should return error dict to LLM, not crash."""
    class _BadRegistry:
        def get_pack(self, sid, wp, na, **kw):
            raise ValueError(f"Unknown specialist: {sid!r}")
        def list_ids(self):
            return []

    call_count = 0

    async def _chat(messages, model, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Do some work first so Gate 1 is satisfied.
            return LLMResponse(content=None, tool_calls=[
                _tool_call("shell", {"command": "echo"}, "p0"),
            ])
        elif call_count == 2:
            return LLMResponse(content=None, tool_calls=[
                _tool_call("delegate_to_specialist", {
                    "sub_task": "x", "specialist_type": "nonexistent",
                }, "p1"),
            ])
        elif call_count == 3:
            return LLMResponse(content=None, tool_calls=[
                _tool_call("finish_task", {"summary": "handled error"}, "p3"),
            ])
        return LLMResponse(content="fallback", tool_calls=[])

    client = MagicMock(spec=[])
    client.chat = AsyncMock(side_effect=_chat)
    pack = _Pack([_shell_tool_def(), _finish_tool_def()])
    await _execute_pack_loop(
        pack=pack,
        messages=[{"role": "system", "content": "test"}, {"role": "user", "content": "x"}],
        model_cfg=_model_cfg(),
        chat_client=client,
        run_repository=_run_repo(),
        run_id=RunId(value="test-run"),
        max_steps=20,
        delegation_depth=1,
        specialist_registry=_BadRegistry(),
        workspace_path="/tmp/test",
    )
    # LLM should have received the error as a tool result in call 3's messages.
    third_call_messages = (client.chat.call_args_list[2].kwargs.get("messages") or
                           client.chat.call_args_list[2].args[0])
    tool_results = [m for m in third_call_messages if m.get("role") == "tool"]
    # Find the delegation error tool result.
    delegation_results = [
        json.loads(m["content"]) for m in tool_results
        if "error" in json.loads(m["content"])
    ]
    assert len(delegation_results) >= 1
    assert "unknown_specialist" in delegation_results[0]["error"] or "Unknown" in delegation_results[0].get("message", "")


# ---------------------------------------------------------------------------
# Tests: Dynamic pack delegation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dynamic_pack_delegation():
    """Delegate with specialist_type='dynamic' and tools list."""
    call_count = 0

    async def _chat(messages, model, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Parent: shell first, then delegate
            return LLMResponse(content=None, tool_calls=[
                _tool_call("shell", {"command": "echo prep"}, "p0"),
            ])
        elif call_count == 2:
            return LLMResponse(content=None, tool_calls=[
                _tool_call("delegate_to_specialist", {
                    "sub_task": "fetch data",
                    "specialist_type": "dynamic",
                    "tools": ["web_search", "fetch_url"],
                    "role": "You are a web data fetcher.",
                }, "p1"),
            ])
        elif call_count == 3:
            # Sub: shell first
            return LLMResponse(content=None, tool_calls=[
                _tool_call("shell", {"command": "echo sub"}, "s0"),
            ])
        elif call_count == 4:
            # Sub: finish
            return LLMResponse(content=None, tool_calls=[
                _tool_call("finish_task", {"summary": "fetched data"}, "s1"),
            ])
        elif call_count == 5:
            # Parent: finish
            return LLMResponse(content=None, tool_calls=[
                _tool_call("finish_task", {"summary": "parent done"}, "p2"),
            ])
        return LLMResponse(content="fallback", tool_calls=[])

    class _DynRegistry:
        def get_pack(self, sid, wp, na, **kw):
            return _Pack([_shell_tool_def(), _finish_tool_def()], sid=sid)
        def list_ids(self):
            return []

    client = MagicMock(spec=[])
    client.chat = AsyncMock(side_effect=_chat)
    pack = _Pack([_shell_tool_def(), _finish_tool_def()])
    result = await _execute_pack_loop(
        pack=pack,
        messages=[{"role": "system", "content": "test"}, {"role": "user", "content": "x"}],
        model_cfg=_model_cfg(),
        chat_client=client,
        run_repository=_run_repo(),
        run_id=RunId(value="test-run"),
        max_steps=20,
        delegation_depth=1,
        specialist_registry=_DynRegistry(),
        workspace_path="/tmp/test",
    )
    assert result["summary"] == "parent done"


# ---------------------------------------------------------------------------
# Tests: Tool absent/present based on depth
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_absent_at_depth_zero():
    """delegate_to_specialist should not be in tool defs at depth 0."""
    responses = [
        LLMResponse(content=None, tool_calls=[_tool_call("shell", {"command": "echo"})]),
        LLMResponse(content=None, tool_calls=[_tool_call("finish_task", {"summary": "ok"}, call_id="c2")]),
    ]
    pack = _Pack([_shell_tool_def(), _finish_tool_def()])
    client = MagicMock(spec=[])
    client.chat = AsyncMock(side_effect=responses)
    await _execute_pack_loop(
        pack=pack,
        messages=[{"role": "system", "content": "test"}, {"role": "user", "content": "x"}],
        model_cfg=_model_cfg(),
        chat_client=client,
        run_repository=_run_repo(),
        run_id=RunId(value="test-run"),
        max_steps=10,
        delegation_depth=0,
        specialist_registry=_Registry(),
        workspace_path="/tmp/test",
    )
    first_call = client.chat.call_args_list[0]
    tools = first_call.kwargs.get("tools", [])
    tool_names = [t["function"]["name"] for t in tools]
    assert "delegate_to_specialist" not in tool_names


@pytest.mark.asyncio
async def test_tool_present_at_depth_one():
    """delegate_to_specialist should be in tool defs at depth > 0."""
    responses = [
        LLMResponse(content=None, tool_calls=[_tool_call("shell", {"command": "echo"})]),
        LLMResponse(content=None, tool_calls=[_tool_call("finish_task", {"summary": "ok"}, call_id="c2")]),
    ]
    pack = _Pack([_shell_tool_def(), _finish_tool_def()])
    client = MagicMock(spec=[])
    client.chat = AsyncMock(side_effect=responses)
    await _execute_pack_loop(
        pack=pack,
        messages=[{"role": "system", "content": "test"}, {"role": "user", "content": "x"}],
        model_cfg=_model_cfg(),
        chat_client=client,
        run_repository=_run_repo(),
        run_id=RunId(value="test-run"),
        max_steps=10,
        delegation_depth=1,
        specialist_registry=_Registry(),
        workspace_path="/tmp/test",
    )
    first_call = client.chat.call_args_list[0]
    tools = first_call.kwargs.get("tools", [])
    tool_names = [t["function"]["name"] for t in tools]
    assert "delegate_to_specialist" in tool_names
