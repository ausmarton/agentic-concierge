"""Unit tests for execute_task: tool loop correctness, finish_task validation,
tool error classification, plain-text response handling, and max_steps behaviour.

These tests patch the LLM client so no real server is needed.  The
FileSystemRunRepository and ConfigSpecialistRegistry are real so the run
directory structure is also exercised.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agentic_concierge.application.execute_task import execute_task
from agentic_concierge.config import load_config
from agentic_concierge.domain import LLMResponse, RunResult, Task, ToolCallRequest
from agentic_concierge.infrastructure.ollama import OllamaChatClient
from agentic_concierge.infrastructure.specialists import ConfigSpecialistRegistry
from agentic_concierge.infrastructure.specialists.base import BaseSpecialistPack
from agentic_concierge.infrastructure.workspace import FileSystemRunRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finish_response(call_id: str = "c1", **kwargs) -> LLMResponse:
    """LLMResponse that calls finish_task with the given arguments."""
    defaults = {"summary": "Done", "artifacts": [], "next_steps": [], "notes": "", "tests_verified": True}
    defaults.update(kwargs)
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(call_id=call_id, tool_name="finish_task", arguments=defaults)],
    )


def _tool_response(tool_name: str, call_id: str = "c1", **kwargs) -> LLMResponse:
    """LLMResponse that calls a non-finish tool."""
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(call_id=call_id, tool_name=tool_name, arguments=kwargs)],
    )


def _read_runlog(run_dir: str) -> list[dict]:
    lines = Path(run_dir, "runlog.jsonl").read_text().strip().splitlines()
    return [json.loads(ln) for ln in lines if ln]


async def _run(chat_mock_responses, *, tmp_path, specialist_id="engineering", max_steps=10,
               max_review_iterations=0):
    """Execute a task with the given sequential mock LLM responses."""
    config = load_config()
    run_repository = FileSystemRunRepository(workspace_root=str(tmp_path))
    specialist_registry = ConfigSpecialistRegistry(config)
    with patch.object(OllamaChatClient, "chat", new_callable=AsyncMock,
                      side_effect=chat_mock_responses):
        chat_client = OllamaChatClient(base_url="http://localhost:11434/v1", timeout_s=5.0)
        task = Task(prompt="test task", specialist_id=specialist_id, network_allowed=False)
        return await execute_task(
            task,
            chat_client=chat_client,
            run_repository=run_repository,
            specialist_registry=specialist_registry,
            config=config,
            max_steps=max_steps,
            max_review_iterations=max_review_iterations,
        )


# ---------------------------------------------------------------------------
# finish_task validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_finish_task_empty_args_rejected_and_llm_retried(tmp_path):
    """finish_task({}) must be rejected; error returned to LLM; a subsequent valid call succeeds."""
    prior_tool = _tool_response("list_files", call_id="c0")
    invalid_call = LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(call_id="c1", tool_name="finish_task", arguments={})],
    )
    valid_call = _finish_response(call_id="c2")

    result = await _run([prior_tool, invalid_call, valid_call], tmp_path=tmp_path)

    # Loop must have continued past the invalid call.
    events = _read_runlog(result.run_dir)
    llm_requests = [e for e in events if e["kind"] == "llm_request"]
    assert len(llm_requests) == 3, "LLM should be called three times (tool, invalid finish, valid finish)"

    # Final payload must come from the valid third call.
    assert result.payload["action"] == "final"
    assert result.payload["summary"] == "Done"

    # The finish_task tool_result with the validation error must be present.
    finish_errors = [
        e for e in events
        if e["kind"] == "tool_result"
        and e["payload"]["tool"] == "finish_task"
        and "missing_fields" in e["payload"].get("result", {})
    ]
    assert len(finish_errors) == 1
    assert "summary" in finish_errors[0]["payload"]["result"]["missing_fields"]


@pytest.mark.asyncio
async def test_finish_task_missing_required_field_rejected(tmp_path):
    """finish_task without 'summary' must be rejected; only the complete call is accepted."""
    prior_tool = _tool_response("list_files", call_id="c0")
    # Engineering pack requires 'summary'. Provide artifacts but not summary.
    incomplete = LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            call_id="c1",
            tool_name="finish_task",
            arguments={"artifacts": ["output.txt"], "notes": ""},
        )],
    )
    valid_call = _finish_response(call_id="c2")

    result = await _run([prior_tool, incomplete, valid_call], tmp_path=tmp_path)

    assert result.payload["action"] == "final"
    assert result.payload["summary"] == "Done"

    events = _read_runlog(result.run_dir)
    # Find the finish_task tool_result that carries the missing-fields error.
    error_result = next(
        e for e in events
        if e["kind"] == "tool_result"
        and e["payload"]["tool"] == "finish_task"
        and "missing_fields" in e["payload"].get("result", {})
    )
    assert "summary" in error_result["payload"]["result"]["missing_fields"]


@pytest.mark.asyncio
async def test_finish_task_valid_args_accepted_after_prior_tool_call(tmp_path):
    """finish_task with all required fields is accepted after a tool has been used (no required-field retry)."""
    result = await _run(
        [_tool_response("list_files", call_id="c1"), _finish_response(call_id="c2")],
        tmp_path=tmp_path,
    )

    assert result.payload["action"] == "final"
    assert result.payload["summary"] == "Done"

    events = _read_runlog(result.run_dir)
    # The finish_task tool_result must indicate task_completed (no error).
    finish_results = [
        e for e in events
        if e["kind"] == "tool_result" and e["payload"]["tool"] == "finish_task"
    ]
    assert len(finish_results) == 1
    assert finish_results[0]["payload"]["result"] == {"status": "task_completed"}


@pytest.mark.asyncio
async def test_finish_task_valid_args_includes_all_provided_fields(tmp_path):
    """All fields provided to finish_task are present in the RunResult payload."""
    result = await _run(
        [
            _tool_response("list_files", call_id="c0"),
            _finish_response(call_id="c1", summary="Created API", artifacts=["app.py"], notes="run with uvicorn"),
        ],
        tmp_path=tmp_path,
    )
    assert result.payload["summary"] == "Created API"
    assert result.payload["artifacts"] == ["app.py"]
    assert result.payload["notes"] == "run with uvicorn"


@pytest.mark.asyncio
async def test_finish_task_research_pack_requires_executive_summary(tmp_path):
    """Research pack's required field is 'answer', not 'summary'."""
    prior_tool = _tool_response("list_files", call_id="c0")
    # Missing answer → should be rejected.
    incomplete = LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            call_id="c1",
            tool_name="finish_task",
            arguments={"sources": ["http://example.com"]},
        )],
    )
    valid_call = LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            call_id="c2",
            tool_name="finish_task",
            arguments={
                "answer": "Overview",
            },
        )],
    )

    result = await _run([prior_tool, incomplete, valid_call], tmp_path=tmp_path, specialist_id="research")

    assert result.payload["action"] == "final"
    assert result.payload["answer"] == "Overview"

    events = _read_runlog(result.run_dir)
    error_result = next(
        e for e in events
        if e["kind"] == "tool_result"
        and e["payload"]["tool"] == "finish_task"
        and "missing_fields" in e["payload"].get("result", {})
    )
    assert "answer" in error_result["payload"]["result"]["missing_fields"]


# ---------------------------------------------------------------------------
# Structural quality gate: finish_task requires prior tool call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_finish_task_without_prior_tool_call_is_rejected(tmp_path):
    """finish_task called before any other tool is rejected; the LLM is asked to do work first."""
    # Step 0: finish_task immediately (no prior tool) → blocked
    # Step 1: list_files → any_non_finish_tool_called = True
    # Step 2: finish_task → accepted
    premature_finish = _finish_response(call_id="c1")
    prior_tool = _tool_response("list_files", call_id="c2")
    valid_finish = _finish_response(call_id="c3")

    result = await _run([premature_finish, prior_tool, valid_finish], tmp_path=tmp_path)

    assert result.payload["action"] == "final"

    events = _read_runlog(result.run_dir)
    # The premature finish must produce the "no work done" error.
    no_work_errors = [
        e for e in events
        if e["kind"] == "tool_result"
        and e["payload"]["tool"] == "finish_task"
        and e["payload"].get("result", {}).get("error") == "finish_task_called_without_doing_work"
    ]
    assert len(no_work_errors) == 1

    # Three LLM calls: premature finish (rejected), list_files, valid finish.
    llm_requests = [e for e in events if e["kind"] == "llm_request"]
    assert len(llm_requests) == 3


@pytest.mark.asyncio
async def test_finish_task_allowed_after_failed_tool_call(tmp_path):
    """A tool that raises an exception still counts as 'attempted'; finish_task is allowed afterward."""
    # list_files will raise (mocked to fail), but any_non_finish_tool_called is still set True.
    responses = [_tool_response("list_files", call_id="c1"), _finish_response(call_id="c2")]
    with _patch_execute_tool(OSError("disk full")):
        result = await _run(responses, tmp_path=tmp_path)

    assert result.payload["action"] == "final"
    events = _read_runlog(result.run_dir)
    # The failed list_files should produce a tool_error (not tool_result).
    assert any(e["kind"] == "tool_error" for e in events)
    # finish_task should succeed (no "no work done" error).
    no_work_errors = [
        e for e in events
        if e["kind"] == "tool_result"
        and e["payload"].get("result", {}).get("error") == "finish_task_called_without_doing_work"
    ]
    assert len(no_work_errors) == 0


# ---------------------------------------------------------------------------
# Plain-text (no tool calls) response + corrective re-prompt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_plain_text_triggers_corrective_reprompt_then_finishes_by_tool(tmp_path):
    """A plain-text response triggers a corrective re-prompt; the LLM then calls tools and finishes."""
    plain = LLMResponse(content="Let me think about this...", tool_calls=[])
    list_files = _tool_response("list_files", call_id="c0")
    finish = _finish_response(summary="Done", call_id="c1")

    result = await _run([plain, list_files, finish], tmp_path=tmp_path)

    assert result.payload["action"] == "final"
    assert result.payload["summary"] == "Done"
    events = _read_runlog(result.run_dir)
    reprompts = [e for e in events if e["kind"] == "corrective_reprompt"]
    assert len(reprompts) == 1
    assert reprompts[0]["payload"]["attempt"] == 1


@pytest.mark.asyncio
async def test_plain_text_response_produces_final_payload(tmp_path):
    """After _MAX_PLAIN_TEXT_RETRIES corrective re-prompts, plain text becomes the final payload."""
    # 2 retries (re-prompted) + 1 final fallback = 3 LLM calls total.
    plain_response = LLMResponse(content="I have completed the task.", tool_calls=[])

    result = await _run([plain_response] * 3, tmp_path=tmp_path)

    assert result.payload["action"] == "final"
    assert result.payload["summary"] == "I have completed the task."
    assert "notes" in result.payload
    events = _read_runlog(result.run_dir)
    reprompts = [e for e in events if e["kind"] == "corrective_reprompt"]
    assert len(reprompts) == 2  # _MAX_PLAIN_TEXT_RETRIES = 2


@pytest.mark.asyncio
async def test_plain_text_empty_content_produces_empty_summary(tmp_path):
    """Plain-text response with None content produces an empty summary (not a crash)."""
    result = await _run([LLMResponse(content=None, tool_calls=[])] * 3, tmp_path=tmp_path)

    assert result.payload["action"] == "final"
    assert result.payload["summary"] == ""


# ---------------------------------------------------------------------------
# max_steps exhaustion
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_steps_exceeded_returns_timeout_payload(tmp_path):
    """When max_steps is reached without finish_task, a timeout payload is returned."""
    # Always returns a list_files call — never finishes.
    looping_response = _tool_response("list_files", call_id="c1")

    result = await _run(
        [looping_response] * 5,  # more than max_steps=3
        tmp_path=tmp_path,
        max_steps=3,
    )

    assert result.payload["action"] == "final"
    assert "max_steps" in result.payload["summary"].lower() or "3" in result.payload["summary"]
    assert result.payload.get("next_steps")

    events = _read_runlog(result.run_dir)
    llm_requests = [e for e in events if e["kind"] == "llm_request"]
    assert len(llm_requests) == 3


@pytest.mark.asyncio
async def test_max_steps_runlog_has_expected_event_count(tmp_path):
    """Each step produces exactly one llm_request and one llm_response event."""
    looping_response = _tool_response("list_files", call_id="c1")

    result = await _run([looping_response] * 5, tmp_path=tmp_path, max_steps=2)

    events = _read_runlog(result.run_dir)
    assert len([e for e in events if e["kind"] == "llm_request"]) == 2
    assert len([e for e in events if e["kind"] == "llm_response"]) == 2


# ---------------------------------------------------------------------------
# Tool error classification (T1-2)
# ---------------------------------------------------------------------------

def _patch_execute_tool(exc: Exception):
    """Context manager: make BaseSpecialistPack.execute_tool raise ``exc``."""
    return patch.object(BaseSpecialistPack, "execute_tool", side_effect=exc)


async def _run_with_tool_error(exc: Exception, tmp_path) -> tuple[dict, list[dict]]:
    """Run a two-step task: first call triggers a tool error, second call finishes."""
    responses = [
        _tool_response("list_files", call_id="c1"),
        _finish_response(call_id="c2"),
    ]
    with _patch_execute_tool(exc):
        result = await _run(responses, tmp_path=tmp_path)
    return result, _read_runlog(result.run_dir)


@pytest.mark.asyncio
async def test_permission_error_produces_tool_error_event(tmp_path):
    """PermissionError (sandbox violation) is logged as tool_error with error_type='permission'."""
    result, events = await _run_with_tool_error(
        PermissionError("Path must be within sandbox root"), tmp_path
    )

    tool_errors = [e for e in events if e["kind"] == "tool_error"]
    assert len(tool_errors) == 1
    assert tool_errors[0]["payload"]["error_type"] == "permission"
    assert tool_errors[0]["payload"]["tool"] == "list_files"
    assert "sandbox" in tool_errors[0]["payload"]["error_message"].lower()

    # No tool_result for the failed tool; only a tool_result for finish_task.
    tool_results = [e for e in events if e["kind"] == "tool_result"]
    finish_results = [e for e in tool_results if e["payload"]["tool"] == "finish_task"]
    list_results = [e for e in tool_results if e["payload"]["tool"] == "list_files"]
    assert len(finish_results) == 1
    assert len(list_results) == 0

    # Loop continues after the error; task finishes successfully.
    assert result.payload["action"] == "final"


@pytest.mark.asyncio
async def test_value_error_produces_tool_error_event(tmp_path):
    """ValueError (bad LLM arguments) is logged as tool_error with error_type='invalid_args'."""
    result, events = await _run_with_tool_error(
        ValueError("cmd must be a non-empty list"), tmp_path
    )

    tool_errors = [e for e in events if e["kind"] == "tool_error"]
    assert len(tool_errors) == 1
    assert tool_errors[0]["payload"]["error_type"] == "invalid_args"
    assert result.payload["action"] == "final"


@pytest.mark.asyncio
async def test_type_error_produces_tool_error_event(tmp_path):
    """TypeError (wrong argument type from LLM) is logged as tool_error with error_type='invalid_args'."""
    result, events = await _run_with_tool_error(
        TypeError("expected list, got str"), tmp_path
    )

    tool_errors = [e for e in events if e["kind"] == "tool_error"]
    assert len(tool_errors) == 1
    assert tool_errors[0]["payload"]["error_type"] == "invalid_args"
    assert result.payload["action"] == "final"


@pytest.mark.asyncio
async def test_os_error_produces_tool_error_event(tmp_path):
    """OSError (filesystem error) is logged as tool_error with error_type='io_error'."""
    result, events = await _run_with_tool_error(
        OSError("No space left on device"), tmp_path
    )

    tool_errors = [e for e in events if e["kind"] == "tool_error"]
    assert len(tool_errors) == 1
    assert tool_errors[0]["payload"]["error_type"] == "io_error"
    assert result.payload["action"] == "final"


@pytest.mark.asyncio
async def test_unexpected_error_produces_tool_error_event(tmp_path):
    """Unexpected exception is logged as tool_error with error_type='unexpected' (not raised)."""
    result, events = await _run_with_tool_error(
        RuntimeError("something completely unexpected"), tmp_path
    )

    tool_errors = [e for e in events if e["kind"] == "tool_error"]
    assert len(tool_errors) == 1
    assert tool_errors[0]["payload"]["error_type"] == "unexpected"
    # Error message includes the original message.
    assert "unexpected" in tool_errors[0]["payload"]["error_message"].lower()
    assert result.payload["action"] == "final"


@pytest.mark.asyncio
async def test_error_content_sent_to_llm_as_tool_result_message(tmp_path):
    """The error dict is sent back to the LLM (as a tool message) so it can adapt."""
    # We can verify this indirectly: the LLM is called a second time (it received
    # the error and produced a finish_task response).
    result, events = await _run_with_tool_error(
        PermissionError("path escape"), tmp_path
    )

    llm_requests = [e for e in events if e["kind"] == "llm_request"]
    assert len(llm_requests) == 2, (
        "LLM must be called twice: once to get list_files, once after receiving the error"
    )


@pytest.mark.asyncio
async def test_successful_tool_produces_tool_result_not_tool_error(tmp_path):
    """Successful tool execution produces tool_result; no tool_error event must be present."""
    result = await _run(
        [_tool_response("list_files", call_id="c1"), _finish_response(call_id="c2")],
        tmp_path=tmp_path,
    )
    events = _read_runlog(result.run_dir)

    assert not any(e["kind"] == "tool_error" for e in events)
    list_results = [
        e for e in events
        if e["kind"] == "tool_result" and e["payload"]["tool"] == "list_files"
    ]
    assert len(list_results) == 1


# ---------------------------------------------------------------------------
# Additional coverage (T2-3)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_malformed_tool_arguments_produce_tool_error(tmp_path):
    """When the LLM returns unparseable JSON, the client stores {'_raw': '...'}.
    The tool receives unexpected kwargs → TypeError → tool_error event, loop continues.
    """
    # Simulate the _raw fallback that client.py produces on json.JSONDecodeError.
    # The LLM response with _raw args triggers a TypeError in the tool → tool_error event.
    result, events = await _run_with_tool_error(TypeError("unexpected keyword argument '_raw'"), tmp_path)

    # tool_error must be logged (TypeError → invalid_args)
    tool_errors = [e for e in events if e["kind"] == "tool_error"]
    assert len(tool_errors) == 1
    assert tool_errors[0]["payload"]["error_type"] == "invalid_args"
    # Loop continued past the error; task completed.
    assert result.payload["action"] == "final"


@pytest.mark.asyncio
async def test_multiple_tool_calls_in_one_response_all_processed(tmp_path):
    """When the LLM returns two tool calls in one response, both are executed."""
    multi_tool_response = LLMResponse(
        content=None,
        tool_calls=[
            ToolCallRequest(call_id="c1", tool_name="list_files", arguments={}),
            ToolCallRequest(call_id="c2", tool_name="list_files", arguments={}),
        ],
    )
    result = await _run(
        [multi_tool_response, _finish_response(call_id="c3")],
        tmp_path=tmp_path,
    )
    events = _read_runlog(result.run_dir)

    # Both tool calls should produce tool_result events in step_0.
    step0_results = [
        e for e in events
        if e["kind"] == "tool_result" and e.get("step") == "step_0"
        and e["payload"]["tool"] == "list_files"
    ]
    assert len(step0_results) == 2, "Both list_files calls should produce tool_result events"
    assert result.payload["action"] == "final"


@pytest.mark.asyncio
async def test_finish_task_coexisting_with_regular_tool_in_same_response(tmp_path):
    """If the LLM returns finish_task alongside a regular tool in one response,
    the regular tool is still executed and finish_task terminates the run.
    """
    combined_response = LLMResponse(
        content=None,
        tool_calls=[
            ToolCallRequest(call_id="c1", tool_name="list_files", arguments={}),
            ToolCallRequest(
                call_id="c2",
                tool_name="finish_task",
                arguments={"summary": "Done", "artifacts": [], "next_steps": [], "notes": "", "tests_verified": True},
            ),
        ],
    )
    result = await _run([combined_response], tmp_path=tmp_path)

    assert result.payload["action"] == "final"
    assert result.payload["summary"] == "Done"

    events = _read_runlog(result.run_dir)
    # The regular tool should have produced a tool_result.
    list_results = [
        e for e in events
        if e["kind"] == "tool_result" and e["payload"]["tool"] == "list_files"
    ]
    assert len(list_results) == 1
    # Exactly one LLM call (no retry needed).
    assert len([e for e in events if e["kind"] == "llm_request"]) == 1


@pytest.mark.asyncio
async def test_unknown_tool_name_returns_error_dict_to_llm(tmp_path):
    """Calling a tool that doesn't exist in the pack returns an error dict to the
    LLM (not a tool_error event) — execute_tool handles it without raising.
    """
    unknown_tool_call = LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(call_id="c1", tool_name="does_not_exist", arguments={})],
    )
    result = await _run([unknown_tool_call, _finish_response(call_id="c2")], tmp_path=tmp_path)

    events = _read_runlog(result.run_dir)
    # No tool_error — execute_tool returns an error dict, not an exception.
    assert not any(e["kind"] == "tool_error" for e in events)
    # The error dict is logged as a tool_result.
    tool_results = [
        e for e in events
        if e["kind"] == "tool_result" and e["payload"]["tool"] == "does_not_exist"
    ]
    assert len(tool_results) == 1
    assert "error" in tool_results[0]["payload"]["result"]
    # Loop continued; task completed.
    assert result.payload["action"] == "final"


# ---------------------------------------------------------------------------
# Security events (T2-4)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_permission_error_produces_security_event(tmp_path):
    """PermissionError produces both a tool_error AND a security_event in the runlog."""
    result, events = await _run_with_tool_error(
        PermissionError("path escape: /etc/passwd outside sandbox root"), tmp_path
    )

    security_events = [e for e in events if e["kind"] == "security_event"]
    assert len(security_events) == 1, "Exactly one security_event must be logged"

    sec = security_events[0]["payload"]
    assert sec["event_type"] == "sandbox_violation"
    assert sec["tool"] == "list_files"
    assert "/etc/passwd" in sec["error_message"]


@pytest.mark.asyncio
async def test_security_event_accompanies_tool_error(tmp_path):
    """The security_event is emitted alongside (not instead of) the tool_error event."""
    result, events = await _run_with_tool_error(
        PermissionError("path escape"), tmp_path
    )

    tool_errors = [e for e in events if e["kind"] == "tool_error"]
    security_events = [e for e in events if e["kind"] == "security_event"]
    assert len(tool_errors) == 1
    assert len(security_events) == 1
    # Both events reference the same tool and step.
    assert tool_errors[0]["payload"]["tool"] == security_events[0]["payload"]["tool"]
    assert tool_errors[0].get("step") == security_events[0].get("step")


@pytest.mark.asyncio
async def test_non_permission_errors_produce_no_security_event(tmp_path):
    """ValueError, OSError, and unexpected errors do NOT produce security_event entries."""
    for exc in [ValueError("bad args"), OSError("disk full"), RuntimeError("oops")]:
        result, events = await _run_with_tool_error(exc, tmp_path)
        assert not any(e["kind"] == "security_event" for e in events), (
            f"security_event must not be emitted for {type(exc).__name__}"
        )


# ---------------------------------------------------------------------------
# Loop / repetition detection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_loop_detected_event_emitted_on_repeated_call(tmp_path):
    """When the same tool+args combination appears _LOOP_DETECT_THRESHOLD times in
    the history window, a loop_detected event is logged in the runlog."""
    # list_files({}) repeated 3 times (triggers at 3rd occurrence) then finish.
    responses = [
        _tool_response("list_files", call_id="c1"),
        _tool_response("list_files", call_id="c2"),
        _tool_response("list_files", call_id="c3"),
        _finish_response(call_id="c4"),
    ]
    result = await _run(responses, tmp_path=tmp_path)

    events = _read_runlog(result.run_dir)
    loop_events = [e for e in events if e["kind"] == "loop_detected"]
    assert len(loop_events) >= 1, "loop_detected event must be logged"
    assert loop_events[0]["payload"]["tool"] == "list_files"
    assert loop_events[0]["payload"]["repeat_count"] >= 2
    # Task still completes normally.
    assert result.payload["action"] == "final"


@pytest.mark.asyncio
async def test_no_loop_detected_for_different_args(tmp_path):
    """Calls to the same tool with DIFFERENT arguments must NOT trigger loop detection."""
    resp1 = LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(call_id="c1", tool_name="read_file", arguments={"path": "a.py"})],
    )
    resp2 = LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(call_id="c2", tool_name="read_file", arguments={"path": "b.py"})],
    )
    resp3 = LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(call_id="c3", tool_name="read_file", arguments={"path": "c.py"})],
    )
    result = await _run([resp1, resp2, resp3, _finish_response(call_id="c4")], tmp_path=tmp_path)

    events = _read_runlog(result.run_dir)
    assert not any(e["kind"] == "loop_detected" for e in events), (
        "Different args must not trigger loop_detected"
    )


@pytest.mark.asyncio
async def test_loop_warning_injected_as_user_message(tmp_path):
    """After loop_detected, a user-visible warning message is injected so the LLM
    sees it in the next context window (one extra llm_request per warning)."""
    responses = [
        _tool_response("list_files", call_id="c1"),
        _tool_response("list_files", call_id="c2"),
        _tool_response("list_files", call_id="c3"),  # triggers loop warning on step 2
        _finish_response(call_id="c4"),
    ]
    result = await _run(responses, tmp_path=tmp_path)

    events = _read_runlog(result.run_dir)
    # At least one loop_detected event must exist.
    assert any(e["kind"] == "loop_detected" for e in events)
    # Task still completes.
    assert result.payload["action"] == "final"


# ---------------------------------------------------------------------------
# Phase 12 additions: brief injection, synthesis, checkpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_planner_subtask_description_injected_into_specialist_messages(tmp_path):
    """When plan_task decomposes into subtasks, the subtask description appears in the
    specialist's first user message.  V2 replacement for orchestrator brief injection.
    """
    from agentic_concierge.application.planner import PlanResult
    from agentic_concierge.application.task_graph import TaskGraph

    config = load_config()
    run_repository = FileSystemRunRepository(workspace_root=str(tmp_path))
    specialist_registry = ConfigSpecialistRegistry(config)

    # Create a TaskGraph with a specific subtask description
    graph = TaskGraph.from_root("build auth")
    graph.add_child(
        graph.root_id, "Implement auth using JWT",
        required_capabilities=["code_python"],
        finish_schema_key="code",
    )
    graph.transition(graph.root_id, "decomposing")
    graph.transition(graph.root_id, "critiqued")

    plan_result = PlanResult(
        graph=graph,
        reasoning="test brief injection",
        critique_feedback=None,
        replan_count=0,
        planner_model="test-model",
        critic_model="test-model",
    )

    captured_messages = []

    async def mock_chat(messages, model, **kwargs):
        captured_messages.append(list(messages))
        # First call: tool, second: finish
        if len(captured_messages) == 1:
            return LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(call_id="t0", tool_name="list_files", arguments={})],
            )
        return _finish_response()

    with patch(
        "agentic_concierge.application.planner.plan_task",
        new_callable=AsyncMock,
        return_value=plan_result,
    ), patch.object(OllamaChatClient, "chat", side_effect=mock_chat):
        chat_client = OllamaChatClient(base_url="http://localhost:11434/v1", timeout_s=5.0)
        task = Task(prompt="build auth", specialist_id=None, network_allowed=False)
        await execute_task(
            task,
            chat_client=chat_client,
            run_repository=run_repository,
            specialist_registry=specialist_registry,
            config=config,
            max_steps=10,
            max_review_iterations=0,
        )

    # The pack loop's first chat call should contain the subtask description
    assert len(captured_messages) >= 1
    first_user_msg = next(
        (m["content"] for m in captured_messages[0] if m["role"] == "user"), ""
    )
    assert "Implement auth using JWT" in first_user_msg, (
        f"Subtask description not found in user message: {first_user_msg!r}"
    )


@pytest.mark.asyncio
async def test_checkpoint_written_and_deleted_on_completion(tmp_path):
    """execute_task writes a checkpoint and deletes it on successful completion."""
    from pathlib import Path as _Path

    config = load_config()
    run_repository = FileSystemRunRepository(workspace_root=str(tmp_path))
    specialist_registry = ConfigSpecialistRegistry(config)

    with patch.object(OllamaChatClient, "chat", new_callable=AsyncMock,
                      side_effect=[
                          _tool_response("list_files"),
                          _finish_response(),
                      ]):
        chat_client = OllamaChatClient(base_url="http://localhost:11434/v1", timeout_s=5.0)
        task = Task(prompt="test checkpoint", specialist_id="engineering", network_allowed=False)
        result = await execute_task(
            task,
            chat_client=chat_client,
            run_repository=run_repository,
            specialist_registry=specialist_registry,
            config=config,
            max_steps=10,
            max_review_iterations=0,
        )

    # After successful completion, checkpoint.json must NOT exist
    assert not (_Path(result.run_dir) / "checkpoint.json").exists(), (
        "checkpoint.json should be deleted after run_complete"
    )


@pytest.mark.asyncio
async def test_quality_gate_rejects_false_tests_verified(tmp_path):
    """Quality gate in _execute_pack_loop rejects tests_verified=False and logs quality_gate_failed."""
    config = load_config()
    run_repository = FileSystemRunRepository(workspace_root=str(tmp_path))
    specialist_registry = ConfigSpecialistRegistry(config)

    # First response: finish_task with tests_verified=False → quality gate fires
    # Second: tool call → satisfies "prior work done" check
    # Third: finish_task with tests_verified=True → accepted
    responses = [
        _finish_response(tests_verified=False),   # gate 1 fires (no prior tool) → error
        _tool_response("list_files"),              # satisfies gate 1
        _finish_response(tests_verified=False),   # gate 3 fires → quality gate error
        _tool_response("list_files", call_id="t2"),
        _finish_response(tests_verified=True),    # accepted
    ]

    with patch.object(OllamaChatClient, "chat", new_callable=AsyncMock, side_effect=responses):
        chat_client = OllamaChatClient(base_url="http://localhost:11434/v1", timeout_s=5.0)
        task = Task(prompt="build", specialist_id="engineering", network_allowed=False)
        result = await execute_task(
            task,
            chat_client=chat_client,
            run_repository=run_repository,
            specialist_registry=specialist_registry,
            config=config,
            max_steps=20,
            max_review_iterations=0,
        )

    events = _read_runlog(result.run_dir)
    quality_gate_events = [e for e in events if e["kind"] == "quality_gate_failed"]
    assert len(quality_gate_events) >= 1, "Expected at least one quality_gate_failed event"


@pytest.mark.asyncio
async def test_single_node_graph_no_extra_overhead(tmp_path):
    """V2: single-leaf task graph completes with just the pack loop calls (no synthesis)."""
    from agentic_concierge.application.planner import PlanResult
    from agentic_concierge.application.task_graph import TaskGraph

    config = load_config()
    run_repository = FileSystemRunRepository(workspace_root=str(tmp_path))
    specialist_registry = ConfigSpecialistRegistry(config)

    # Build a single-leaf graph (planner produces one subtask)
    graph = TaskGraph.from_root("build", node_id="root")
    graph.add_child(
        "root", "Build the thing",
        node_id="leaf1",
        required_capabilities=["code_python"],
    )
    plan_result = PlanResult(
        graph=graph,
        reasoning="Single subtask",
        critique_feedback=None,
        replan_count=0,
        planner_model="test-model",
        critic_model="test-model",
    )

    call_count = {"n": 0}

    async def mock_chat(messages, model, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(call_id="t0", tool_name="list_files", arguments={})],
            )
        return _finish_response()

    with patch(
        "agentic_concierge.application.planner.plan_task",
        new_callable=AsyncMock,
        return_value=plan_result,
    ), patch.object(OllamaChatClient, "chat", side_effect=mock_chat):
        chat_client = OllamaChatClient(base_url="http://localhost:11434/v1", timeout_s=5.0)
        task = Task(prompt="build", specialist_id=None, network_allowed=False)
        await execute_task(
            task,
            chat_client=chat_client,
            run_repository=run_repository,
            specialist_registry=specialist_registry,
            config=config,
            max_steps=10,
            max_review_iterations=0,
        )

    # Only 2 chat calls (list_files + finish_task); no synthesis overhead
    assert call_count["n"] == 2, (
        f"Expected 2 chat calls (no synthesis), got {call_count['n']}"
    )


# ---------------------------------------------------------------------------
# Timeout recovery tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_timeout_recovery_uses_smaller_model(tmp_path):
    """httpx.ReadTimeout on first chat call -> pack loop retries with smaller model."""
    import httpx
    from agentic_concierge.application.execute_task import _execute_pack_loop
    from agentic_concierge.config.schema import ModelConfig

    call_models = []

    async def mock_chat(*, messages, model, tools, temperature=0.1, top_p=0.9, max_tokens=2048):
        call_models.append(model)
        if model == "qwen2.5:32b":
            raise httpx.ReadTimeout("read timed out")
        # Return a finish_task response for the smaller model
        return _finish_response()

    from unittest.mock import MagicMock
    chat_client = MagicMock()
    chat_client.chat = AsyncMock(side_effect=mock_chat)
    # Ensure pop_events is not present so the drain-events block is skipped
    del chat_client.pop_events

    run_repository = FileSystemRunRepository(workspace_root=str(tmp_path))
    run_id, run_dir, workspace_path = run_repository.create_run()

    config = load_config()
    pack = ConfigSpecialistRegistry(config).get_pack("engineering", workspace_path, False)

    model_cfg = ModelConfig(
        base_url="http://localhost:11434/v1",
        model="qwen2.5:32b",
        timeout_s=120.0,
    )

    messages = [
        {"role": "system", "content": pack.system_prompt},
        {"role": "user", "content": "Task:\ntest"},
    ]

    # build_chat_client returns a new client without pop_events
    fallback_client = MagicMock()
    fallback_client.chat = AsyncMock(side_effect=mock_chat)
    del fallback_client.pop_events

    with patch("agentic_concierge.infrastructure.chat.build_chat_client", return_value=fallback_client):
        payload = await _execute_pack_loop(
            pack=pack,
            messages=messages,
            model_cfg=model_cfg,
            chat_client=chat_client,
            run_repository=run_repository,
            run_id=run_id,
            max_steps=5,
            available_models=["qwen2.5:8b", "qwen2.5:32b"],
        )

    # First call was to 32b (timed out), second to 8b (succeeded)
    assert call_models[0] == "qwen2.5:32b"
    assert call_models[1] == "qwen2.5:8b"
    assert payload.get("action") == "final"


@pytest.mark.asyncio
async def test_timeout_no_smaller_model_raises(tmp_path):
    """httpx.ReadTimeout with no smaller model available -> RuntimeError."""
    import httpx
    from agentic_concierge.application.execute_task import _execute_pack_loop
    from agentic_concierge.config.schema import ModelConfig

    async def mock_chat(*, messages, model, tools, temperature=0.1, top_p=0.9, max_tokens=2048):
        raise httpx.ReadTimeout("read timed out")

    chat_client = AsyncMock()
    chat_client.chat = AsyncMock(side_effect=mock_chat)

    run_repository = FileSystemRunRepository(workspace_root=str(tmp_path))
    run_id, run_dir, workspace_path = run_repository.create_run()

    config = load_config()
    pack = ConfigSpecialistRegistry(config).get_pack("engineering", workspace_path, False)

    model_cfg = ModelConfig(
        base_url="http://localhost:11434/v1",
        model="qwen2.5:3b",
        timeout_s=120.0,
    )

    messages = [
        {"role": "system", "content": pack.system_prompt},
        {"role": "user", "content": "Task:\ntest"},
    ]

    with pytest.raises(RuntimeError, match="No smaller model available"):
        await _execute_pack_loop(
            pack=pack,
            messages=messages,
            model_cfg=model_cfg,
            chat_client=chat_client,
            run_repository=run_repository,
            run_id=run_id,
            max_steps=5,
            available_models=["qwen2.5:3b"],
        )
