"""Tests for the human approval mechanism (ADR-021)."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_concierge.application.approval import (
    ApprovalChannel,
    ApprovalDecision,
    ApprovalRequest,
)
from agentic_concierge.application.execute_task import (
    _REQUEST_APPROVAL_TOOL_DEF,
    _execute_pack_loop,
)
from agentic_concierge.config import ModelConfig
from agentic_concierge.domain import LLMResponse, RunId, ToolCallRequest
from agentic_concierge.infrastructure.approval import (
    AutoApprovalChannel,
    CliApprovalChannel,
    HttpApprovalChannel,
)


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

    def __init__(self, tool_defs=None):
        self._tool_defs = tool_defs or []

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


def _make_chat_client(responses: List[LLMResponse]) -> MagicMock:
    client = MagicMock()
    client.chat = AsyncMock(side_effect=responses)
    # Ensure pop_events is not present (AsyncMock auto-creates attributes).
    if hasattr(client, "pop_events"):
        del client.pop_events
    return client


def _tool_call(name: str, args: dict, call_id: str = "c1") -> ToolCallRequest:
    return ToolCallRequest(tool_name=name, arguments=args, call_id=call_id)


def _run_repo() -> MagicMock:
    repo = MagicMock()
    repo.append_event = MagicMock()
    return repo


# ---------------------------------------------------------------------------
# Tests: AutoApprovalChannel
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_auto_channel_approves():
    """AutoApprovalChannel always returns approved=True."""
    ch = AutoApprovalChannel()
    decision = await ch.request_approval(ApprovalRequest(action="deploy"))
    assert decision.approved is True
    assert "auto" in decision.comment.lower()


# ---------------------------------------------------------------------------
# Tests: No channel → auto-approve
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_channel_auto_approves():
    """When no approval_channel is provided and LLM somehow calls request_approval,
    it should auto-approve."""
    # LLM calls request_approval, then shell, then finish_task.
    responses = [
        LLMResponse(
            content=None,
            tool_calls=[_tool_call("request_approval", {"action": "deploy", "reason": "test", "command": "git push"})],
        ),
        LLMResponse(
            content=None,
            tool_calls=[_tool_call("shell", {"command": "echo hello"}, call_id="c2")],
        ),
        LLMResponse(
            content=None,
            tool_calls=[_tool_call("finish_task", {"summary": "Done"}, call_id="c3")],
        ),
    ]
    pack = _Pack([_shell_tool_def(), _finish_tool_def()])
    result = await _execute_pack_loop(
        pack=pack,
        messages=[{"role": "system", "content": "test"}, {"role": "user", "content": "deploy"}],
        model_cfg=_model_cfg(),
        chat_client=_make_chat_client(responses),
        run_repository=_run_repo(),
        run_id=RunId(value="test-run"),
        max_steps=10,
        approval_channel=None,  # no channel
    )
    assert result["summary"] == "Done"


# ---------------------------------------------------------------------------
# Tests: Channel denies
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_channel_denies():
    """When channel denies, the LLM gets approved=False in the tool result."""
    class DenyChannel:
        async def request_approval(self, req):
            return ApprovalDecision(approved=False, comment="Too risky")

    responses = [
        LLMResponse(
            content=None,
            tool_calls=[_tool_call("request_approval", {"action": "deploy", "reason": "test", "command": "cmd"})],
        ),
        LLMResponse(
            content=None,
            tool_calls=[_tool_call("shell", {"command": "echo hi"}, call_id="c2")],
        ),
        LLMResponse(
            content=None,
            tool_calls=[_tool_call("finish_task", {"summary": "Aborted"}, call_id="c3")],
        ),
    ]
    pack = _Pack([_shell_tool_def(), _finish_tool_def()])
    client = _make_chat_client(responses)
    repo = _run_repo()
    await _execute_pack_loop(
        pack=pack,
        messages=[{"role": "system", "content": "test"}, {"role": "user", "content": "deploy"}],
        model_cfg=_model_cfg(),
        chat_client=client,
        run_repository=repo,
        run_id=RunId(value="test-run"),
        max_steps=10,
        approval_channel=DenyChannel(),
    )
    # Verify the denial was communicated to the LLM via tool result.
    messages_sent = client.chat.call_args_list
    # The second call should have the tool result with approved=False.
    second_call_messages = messages_sent[1].kwargs.get("messages") or messages_sent[1].args[0]
    tool_results = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_results) >= 1
    denial_content = json.loads(tool_results[0]["content"])
    assert denial_content["approved"] is False
    assert "risky" in denial_content["comment"].lower()


# ---------------------------------------------------------------------------
# Tests: Timeout denies (fail-closed)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_timeout_denies():
    """Channel raising TimeoutError results in denial (fail-closed)."""
    class TimeoutChannel:
        async def request_approval(self, req):
            raise TimeoutError("timed out")

    responses = [
        LLMResponse(
            content=None,
            tool_calls=[_tool_call("request_approval", {"action": "x", "reason": "y", "command": "z"})],
        ),
        LLMResponse(
            content=None,
            tool_calls=[_tool_call("shell", {"command": "echo"}, call_id="c2")],
        ),
        LLMResponse(
            content=None,
            tool_calls=[_tool_call("finish_task", {"summary": "done"}, call_id="c3")],
        ),
    ]
    pack = _Pack([_shell_tool_def(), _finish_tool_def()])
    client = _make_chat_client(responses)
    repo = _run_repo()
    await _execute_pack_loop(
        pack=pack,
        messages=[{"role": "system", "content": "test"}, {"role": "user", "content": "x"}],
        model_cfg=_model_cfg(),
        chat_client=client,
        run_repository=repo,
        run_id=RunId(value="test-run"),
        max_steps=10,
        approval_channel=TimeoutChannel(),
    )
    # Should still complete, with denial communicated to LLM.
    second_call_messages = (client.chat.call_args_list[1].kwargs.get("messages") or
                            client.chat.call_args_list[1].args[0])
    tool_results = [m for m in second_call_messages if m.get("role") == "tool"]
    denial_content = json.loads(tool_results[0]["content"])
    assert denial_content["approved"] is False


# ---------------------------------------------------------------------------
# Tests: Events emitted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_events_emitted():
    """Verify approval_requested and approval_granted events are emitted."""
    responses = [
        LLMResponse(
            content=None,
            tool_calls=[_tool_call("request_approval", {"action": "push", "reason": "ready", "command": "git push"})],
        ),
        LLMResponse(
            content=None,
            tool_calls=[_tool_call("shell", {"command": "echo"}, call_id="c2")],
        ),
        LLMResponse(
            content=None,
            tool_calls=[_tool_call("finish_task", {"summary": "ok"}, call_id="c3")],
        ),
    ]
    pack = _Pack([_shell_tool_def(), _finish_tool_def()])
    repo = _run_repo()
    await _execute_pack_loop(
        pack=pack,
        messages=[{"role": "system", "content": "test"}, {"role": "user", "content": "push"}],
        model_cfg=_model_cfg(),
        chat_client=_make_chat_client(responses),
        run_repository=repo,
        run_id=RunId(value="test-run"),
        max_steps=10,
        approval_channel=AutoApprovalChannel(),
    )
    event_kinds = [call.args[1] for call in repo.append_event.call_args_list]
    assert "approval_requested" in event_kinds
    assert "approval_granted" in event_kinds


# ---------------------------------------------------------------------------
# Tests: Tool absent/present based on channel
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_absent_when_no_channel():
    """request_approval tool should not appear in tool defs when no channel."""
    # LLM calls finish_task after doing some work.
    responses = [
        LLMResponse(content=None, tool_calls=[_tool_call("shell", {"command": "echo"})]),
        LLMResponse(content=None, tool_calls=[_tool_call("finish_task", {"summary": "ok"}, call_id="c2")]),
    ]
    pack = _Pack([_shell_tool_def(), _finish_tool_def()])
    client = _make_chat_client(responses)
    await _execute_pack_loop(
        pack=pack,
        messages=[{"role": "system", "content": "test"}, {"role": "user", "content": "x"}],
        model_cfg=_model_cfg(),
        chat_client=client,
        run_repository=_run_repo(),
        run_id=RunId(value="test-run"),
        max_steps=10,
        approval_channel=None,
    )
    # Check that the tool defs passed to chat() do NOT include request_approval.
    first_call = client.chat.call_args_list[0]
    tools = first_call.kwargs.get("tools", [])
    tool_names = [t["function"]["name"] for t in tools]
    assert "request_approval" not in tool_names


@pytest.mark.asyncio
async def test_tool_present_when_channel_active():
    """request_approval tool should appear in tool defs when channel is provided."""
    responses = [
        LLMResponse(content=None, tool_calls=[_tool_call("shell", {"command": "echo"})]),
        LLMResponse(content=None, tool_calls=[_tool_call("finish_task", {"summary": "ok"}, call_id="c2")]),
    ]
    pack = _Pack([_shell_tool_def(), _finish_tool_def()])
    client = _make_chat_client(responses)
    await _execute_pack_loop(
        pack=pack,
        messages=[{"role": "system", "content": "test"}, {"role": "user", "content": "x"}],
        model_cfg=_model_cfg(),
        chat_client=client,
        run_repository=_run_repo(),
        run_id=RunId(value="test-run"),
        max_steps=10,
        approval_channel=AutoApprovalChannel(),
    )
    first_call = client.chat.call_args_list[0]
    tools = first_call.kwargs.get("tools", [])
    tool_names = [t["function"]["name"] for t in tools]
    assert "request_approval" in tool_names


# ---------------------------------------------------------------------------
# Tests: Prompt includes approval actions
# ---------------------------------------------------------------------------

def test_prompt_includes_approval_actions():
    """System prompt should contain approval actions when configured."""
    from agentic_concierge.infrastructure.specialists.prompts import generate_system_prompt
    prompt = generate_system_prompt(
        "You are an agent.", ["shell"],
        approval_actions=["deploy", "push"],
    )
    assert "deploy" in prompt
    assert "push" in prompt
    assert "request_approval" in prompt.lower() or "approval" in prompt.lower()


# ---------------------------------------------------------------------------
# Tests: CliApprovalChannel with mocked input
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cli_channel_with_mocked_input(monkeypatch):
    """CliApprovalChannel should work with mocked stdin input."""
    inputs = iter(["y", "looks good"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    ch = CliApprovalChannel(timeout_s=5.0)
    decision = await ch.request_approval(ApprovalRequest(action="deploy"))
    assert decision.approved is True
    assert decision.comment == "looks good"


# ---------------------------------------------------------------------------
# Tests: HttpApprovalChannel resolve
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_http_channel_resolve():
    """HttpApprovalChannel should resolve when resolve() is called."""
    ch = HttpApprovalChannel(timeout_s=5.0)
    req = ApprovalRequest(action="deploy")

    async def _resolve_after_delay():
        await asyncio.sleep(0.05)
        ch.resolve(req.request_id, ApprovalDecision(approved=True, comment="ok"))

    asyncio.create_task(_resolve_after_delay())
    decision = await ch.request_approval(req)
    assert decision.approved is True
    assert decision.comment == "ok"


# ---------------------------------------------------------------------------
# Tests: Approval in parallel mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_approval_in_parallel_mode():
    """Approval channel is threaded through V2 LeafExecutionContext to pack loop."""
    from agentic_concierge.application.leaf_adapter import LeafExecutionContext, build_leaf_executor

    channel = AutoApprovalChannel()
    mock_loop = AsyncMock(return_value={"summary": "ok", "action": "final"})
    ctx = LeafExecutionContext(
        specialist_registry=MagicMock(get_pack=MagicMock(return_value=_Pack([_finish_tool_def()]))),
        run_repository=_run_repo(),
        run_id=RunId(value="test-run"),
        workspace_path="/tmp/test",
        config=MagicMock(models={"quality": _model_cfg()}),
        network_allowed=True,
        max_steps=10,
        approval_channel=channel,
        pack_loop_fn=mock_loop,
    )
    executor = build_leaf_executor(ctx)

    from agentic_concierge.application.task_graph import TaskGraph, TaskNode
    from agentic_concierge.application.agent_roles import ModelAssignment

    graph = TaskGraph.from_root("Deploy", node_id="root")
    child = graph.add_child("root", "Deploy service", node_id="c1")

    handle = MagicMock()
    handle.model_id = "test-model"
    handle.chat_client = AsyncMock()
    handle.slot = MagicMock()
    handle.slot.model_id = "test-model"
    handle.slot.backend = "ollama"
    assignment = ModelAssignment(
        role="engineer", model_id="test-model", handle=handle,
        requirements={"instruction_following": 0.7},
    )

    result = await executor(child, graph, assignment)
    assert result["summary"] == "ok"
    # Verify approval_channel was passed to pack loop
    call_kwargs = mock_loop.call_args[1]
    assert call_kwargs["approval_channel"] is channel


# ---------------------------------------------------------------------------
# Tests: Approval + review both active
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_approval_with_review():
    """Both approval and review mechanisms should work together."""
    # LLM: request_approval → shell → finish_task, review approves.
    responses = [
        LLMResponse(
            content=None,
            tool_calls=[_tool_call("request_approval", {"action": "push", "reason": "ok", "command": "git push"})],
        ),
        LLMResponse(content=None, tool_calls=[_tool_call("shell", {"command": "echo"}, call_id="c2")]),
        LLMResponse(content=None, tool_calls=[_tool_call("finish_task", {"summary": "done"}, call_id="c3")]),
        # Reviewer: approve
        LLMResponse(content=None, tool_calls=[_tool_call("approve_work", {"comment": "lgtm"}, call_id="c4")]),
    ]
    pack = _Pack([_shell_tool_def(), _finish_tool_def()])
    result = await _execute_pack_loop(
        pack=pack,
        messages=[{"role": "system", "content": "test"}, {"role": "user", "content": "push"}],
        model_cfg=_model_cfg(),
        chat_client=_make_chat_client(responses),
        run_repository=_run_repo(),
        run_id=RunId(value="test-run"),
        max_steps=10,
        approval_channel=AutoApprovalChannel(),
        max_review_iterations=1,
    )
    assert result["summary"] == "done"
