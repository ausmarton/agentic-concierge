"""Tests for adaptive escalation (ADR-020).

Verify that the model-escalation mechanism in ``_execute_pack_loop`` works:
- Plain-text exhaustion triggers escalation to a larger model.
- Persistent loop detection triggers escalation.
- Review rejection at max triggers escalation and resets review counter.
- Escalation cap (``_MAX_ESCALATIONS``) prevents runaway escalation.
- No larger model available → fall through to existing behaviour.
- Each trigger type fires at most once.
- ``model_escalated`` runlog event is emitted with correct payload.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agentic_concierge.application.execute_task import (
    execute_task,
    _MAX_ESCALATIONS,
)
from agentic_concierge.config import load_config
from agentic_concierge.domain import LLMResponse, Task, ToolCallRequest
from agentic_concierge.infrastructure.ollama import OllamaChatClient
from agentic_concierge.infrastructure.specialists import ConfigSpecialistRegistry
from agentic_concierge.infrastructure.workspace import FileSystemRunRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finish_response(call_id: str = "c1", **kwargs) -> LLMResponse:
    defaults = {
        "summary": "Done",
        "artifacts": [],
        "next_steps": [],
        "notes": "",
        "tests_verified": True,
    }
    defaults.update(kwargs)
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(call_id=call_id, tool_name="finish_task", arguments=defaults)],
    )


def _tool_response(tool_name: str, call_id: str = "c1", **kwargs) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(call_id=call_id, tool_name=tool_name, arguments=kwargs)],
    )


def _plain_text_response(text: str = "I cannot use tools") -> LLMResponse:
    return LLMResponse(content=text, tool_calls=[])


def _approve_response(call_id: str = "r1", comment: str = "Looks good") -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            call_id=call_id, tool_name="approve_work",
            arguments={"comment": comment},
        )],
    )


def _reject_response(call_id: str = "r1", critique: str = "Needs work") -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            call_id=call_id, tool_name="request_revision",
            arguments={"critique": critique},
        )],
    )


def _read_runlog(run_dir: str) -> list[dict]:
    lines = Path(run_dir, "runlog.jsonl").read_text().strip().splitlines()
    return [json.loads(ln) for ln in lines if ln]


def _resolved_llm_with_models(current: str, available: list[str]):
    """Create a minimal ResolvedLLM-like object for threading available_models."""
    from agentic_concierge.infrastructure.llm_discovery import ResolvedLLM
    from agentic_concierge.config import ModelConfig
    return ResolvedLLM(
        base_url="http://localhost:11434/v1",
        model=current,
        model_config=ModelConfig(
            base_url="http://localhost:11434/v1",
            model=current,
        ),
        warnings=[],
        available_models=available,
    )


async def _run_escalation(
    chat_mock_responses,
    *,
    tmp_path,
    specialist_id="engineering",
    max_steps=20,
    max_review_iterations=0,
    max_escalations=2,
    available_models=None,
    start_model="qwen2.5:7b",
):
    """Execute a task with escalation enabled and available_models populated."""
    config = load_config()
    run_repository = FileSystemRunRepository(workspace_root=str(tmp_path))
    specialist_registry = ConfigSpecialistRegistry(config)

    resolved_llm = None
    resolved_model_cfg = None
    if available_models is not None:
        resolved_llm = _resolved_llm_with_models(start_model, available_models)
        resolved_model_cfg = resolved_llm.model_config

    with patch.object(
        OllamaChatClient, "chat", new_callable=AsyncMock,
        side_effect=chat_mock_responses,
    ):
        chat_client = OllamaChatClient(
            base_url="http://localhost:11434/v1", timeout_s=5.0,
        )
        task = Task(
            prompt="test task", specialist_id=specialist_id,
            network_allowed=False,
        )
        return await execute_task(
            task,
            chat_client=chat_client,
            run_repository=run_repository,
            specialist_registry=specialist_registry,
            config=config,
            max_steps=max_steps,
            max_review_iterations=max_review_iterations,
            max_escalations=max_escalations,
            resolved_llm=resolved_llm,
            resolved_model_cfg=resolved_model_cfg,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_plain_text_exhaustion_triggers_escalation(tmp_path):
    """After _MAX_PLAIN_TEXT_RETRIES plain texts, escalation to a larger model
    should be attempted before falling back to using text as payload."""
    responses = [
        # Step 0: tool call (satisfies Gate 1)
        _tool_response("list_files", call_id="c0"),
        # Steps 1-2: plain text (corrective re-prompts, 2 retries)
        _plain_text_response("plain 1"),
        _plain_text_response("plain 2"),
        # Step 3: third plain text → exhaustion → triggers escalation
        _plain_text_response("plain 3"),
        # Step 4-5: after escalation to larger model, LLM now uses tools
        _tool_response("list_files", call_id="c3"),
        _finish_response(call_id="c4"),
    ]
    result = await _run_escalation(
        responses,
        tmp_path=tmp_path,
        available_models=["qwen2.5:7b", "qwen2.5:14b"],
        max_escalations=2,
    )
    assert result.payload["action"] == "final"
    assert result.payload["summary"] == "Done"

    events = _read_runlog(result.run_dir)
    escalated = [e for e in events if e["kind"] == "model_escalated"]
    assert len(escalated) == 1
    assert escalated[0]["payload"]["trigger"] == "plain_text_exhaustion"
    assert escalated[0]["payload"]["from_model"] == "qwen2.5:7b"
    assert escalated[0]["payload"]["to_model"] == "qwen2.5:14b"
    assert escalated[0]["payload"]["escalation_count"] == 1


@pytest.mark.asyncio
async def test_plain_text_no_larger_model_falls_through(tmp_path):
    """When no larger model is available, plain text exhaustion falls through
    to the existing behaviour (text used as final payload)."""
    responses = [
        _tool_response("list_files", call_id="c0"),
        _plain_text_response("plain 1"),
        _plain_text_response("plain 2"),
        _plain_text_response("plain 3"),  # exhaustion; no larger model
    ]
    result = await _run_escalation(
        responses,
        tmp_path=tmp_path,
        available_models=["qwen2.5:7b"],  # no larger model
        max_escalations=2,
    )
    assert result.payload["action"] == "final"
    assert "plain text" in result.payload.get("notes", "").lower()

    events = _read_runlog(result.run_dir)
    escalated = [e for e in events if e["kind"] == "model_escalated"]
    assert len(escalated) == 0


@pytest.mark.asyncio
async def test_escalation_disabled_falls_through(tmp_path):
    """max_escalations=0 disables escalation entirely."""
    responses = [
        _tool_response("list_files", call_id="c0"),
        _plain_text_response("plain 1"),
        _plain_text_response("plain 2"),
        _plain_text_response("plain 3"),  # exhaustion; escalation disabled
    ]
    result = await _run_escalation(
        responses,
        tmp_path=tmp_path,
        available_models=["qwen2.5:7b", "qwen2.5:14b"],
        max_escalations=0,
    )
    assert result.payload["action"] == "final"
    assert "plain text" in result.payload.get("notes", "").lower()

    events = _read_runlog(result.run_dir)
    escalated = [e for e in events if e["kind"] == "model_escalated"]
    assert len(escalated) == 0


@pytest.mark.asyncio
async def test_review_rejection_max_triggers_escalation(tmp_path):
    """After max_review_iterations rejections, escalation should be attempted
    before accepting with _review_warning."""
    responses = [
        # Doer: tool + finish
        _tool_response("list_files", call_id="c0"),
        _finish_response(call_id="c1"),
        # Reviewer 1: reject
        _reject_response(call_id="r1"),
        # Doer: revise and finish again
        _tool_response("list_files", call_id="c2"),
        _finish_response(call_id="c3"),
        # Reviewer 2: reject (now at max_review_iterations)
        _reject_response(call_id="r2"),
        # Doer (escalated model): revise and finish
        _tool_response("list_files", call_id="c4"),
        _finish_response(call_id="c5"),
        # Reviewer (escalated): approve
        _approve_response(call_id="r3"),
    ]
    result = await _run_escalation(
        responses,
        tmp_path=tmp_path,
        available_models=["qwen2.5:7b", "qwen2.5:14b"],
        max_review_iterations=2,
        max_escalations=2,
    )
    assert result.payload["action"] == "final"
    assert result.payload["summary"] == "Done"
    assert "_review_warning" not in result.payload

    events = _read_runlog(result.run_dir)
    escalated = [e for e in events if e["kind"] == "model_escalated"]
    assert len(escalated) == 1
    assert escalated[0]["payload"]["trigger"] == "review_rejection_max"


@pytest.mark.asyncio
async def test_review_rejection_no_escalation_accepts_with_warning(tmp_path):
    """Without escalation (no larger model), max review rejections fall through
    to accepting with _review_warning."""
    responses = [
        _tool_response("list_files", call_id="c0"),
        _finish_response(call_id="c1"),
        _reject_response(call_id="r1"),
        _tool_response("list_files", call_id="c2"),
        _finish_response(call_id="c3"),
        _reject_response(call_id="r2"),  # max reached, no larger model
    ]
    result = await _run_escalation(
        responses,
        tmp_path=tmp_path,
        available_models=["qwen2.5:7b"],  # no larger model
        max_review_iterations=2,
        max_escalations=2,
    )
    assert result.payload["action"] == "final"
    assert "_review_warning" in result.payload


@pytest.mark.asyncio
async def test_each_trigger_fires_at_most_once(tmp_path):
    """A trigger type that already fired should not fire again."""
    responses = [
        _tool_response("list_files", call_id="c0"),
        # First plain text exhaustion → escalate to 14b (3 plain texts: 2 retries + exhaustion)
        _plain_text_response("p1"),  # retry 1/2
        _plain_text_response("p2"),  # retry 2/2
        _plain_text_response("p3"),  # exhaustion → escalation to 14b
        # After escalation: plain text exhaustion again (3 more: 2 retries + exhaustion)
        _plain_text_response("p4"),  # retry 1/2
        _plain_text_response("p5"),  # retry 2/2
        _plain_text_response("p6"),  # exhaustion; trigger already fired → falls through
    ]
    result = await _run_escalation(
        responses,
        tmp_path=tmp_path,
        available_models=["qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b"],
        max_escalations=2,
    )
    # Should fall through to plain-text-as-payload since trigger already fired
    assert result.payload["action"] == "final"
    assert "plain text" in result.payload.get("notes", "").lower()

    events = _read_runlog(result.run_dir)
    escalated = [e for e in events if e["kind"] == "model_escalated"]
    assert len(escalated) == 1  # only one escalation, not two


@pytest.mark.asyncio
async def test_escalation_cap_prevents_runaway(tmp_path):
    """After max_escalations total escalations, further triggers are ignored."""
    responses = [
        _tool_response("list_files", call_id="c0"),
        # Trigger 1: plain text exhaustion → escalate to 14b (3 texts)
        _plain_text_response("p1"),  # retry 1/2
        _plain_text_response("p2"),  # retry 2/2
        _plain_text_response("p3"),  # exhaustion → escalation #1 to 14b
        # After escalation to 14b: doer finishes, reviewer rejects × 2
        _tool_response("list_files", call_id="c1"),
        _finish_response(call_id="c2"),
        _reject_response(call_id="r1"),
        _tool_response("list_files", call_id="c3"),
        _finish_response(call_id="c4"),
        # Trigger 2: review_rejection_max → escalation #2 to 32b
        _reject_response(call_id="r2"),
        # After escalation to 32b: finish + approve
        _tool_response("list_files", call_id="c5"),
        _finish_response(call_id="c6"),
        _approve_response(call_id="r3"),
    ]
    result = await _run_escalation(
        responses,
        tmp_path=tmp_path,
        available_models=["qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b"],
        max_review_iterations=2,
        max_escalations=2,
    )
    assert result.payload["action"] == "final"
    assert result.payload["summary"] == "Done"

    events = _read_runlog(result.run_dir)
    escalated = [e for e in events if e["kind"] == "model_escalated"]
    assert len(escalated) == 2
    assert escalated[0]["payload"]["trigger"] == "plain_text_exhaustion"
    assert escalated[1]["payload"]["trigger"] == "review_rejection_max"


@pytest.mark.asyncio
async def test_model_escalated_event_payload(tmp_path):
    """The model_escalated runlog event should have the correct payload fields."""
    responses = [
        _tool_response("list_files", call_id="c0"),
        _plain_text_response("p1"),  # retry 1/2
        _plain_text_response("p2"),  # retry 2/2
        _plain_text_response("p3"),  # exhaustion → escalation
        # After escalation: finish normally
        _finish_response(call_id="c1"),
    ]
    result = await _run_escalation(
        responses,
        tmp_path=tmp_path,
        available_models=["qwen2.5:7b", "qwen2.5:14b"],
        max_escalations=1,
    )

    events = _read_runlog(result.run_dir)
    escalated = [e for e in events if e["kind"] == "model_escalated"]
    assert len(escalated) == 1
    ev = escalated[0]["payload"]
    assert ev["trigger"] == "plain_text_exhaustion"
    assert ev["from_model"] == "qwen2.5:7b"
    assert ev["to_model"] == "qwen2.5:14b"
    assert ev["escalation_count"] == 1


# ---------------------------------------------------------------------------
# pick_larger_model unit tests
# ---------------------------------------------------------------------------

def test_pick_larger_model_returns_smallest_larger():
    from agentic_concierge.infrastructure.llm_discovery import pick_larger_model
    result = pick_larger_model(
        ["qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b"],
        "qwen2.5:7b",
    )
    assert result == "qwen2.5:14b"


def test_pick_larger_model_prefers_same_family():
    from agentic_concierge.infrastructure.llm_discovery import pick_larger_model
    result = pick_larger_model(
        ["qwen2.5:7b", "llama3.1:8b", "qwen2.5:14b"],
        "qwen2.5:7b",
    )
    assert result == "qwen2.5:14b"


def test_pick_larger_model_falls_back_to_different_family():
    from agentic_concierge.infrastructure.llm_discovery import pick_larger_model
    result = pick_larger_model(
        ["qwen2.5:7b", "llama3.1:70b"],
        "qwen2.5:7b",
    )
    assert result == "llama3.1:70b"


def test_pick_larger_model_returns_none_when_no_larger():
    from agentic_concierge.infrastructure.llm_discovery import pick_larger_model
    result = pick_larger_model(
        ["qwen2.5:7b", "qwen2.5:0.5b"],
        "qwen2.5:7b",
    )
    assert result is None


def test_pick_larger_model_returns_none_for_empty_list():
    from agentic_concierge.infrastructure.llm_discovery import pick_larger_model
    result = pick_larger_model([], "qwen2.5:7b")
    assert result is None


def test_pick_larger_model_skips_unparseable_names():
    """Models with no parseable size (999.0 sentinel) should not be selected."""
    from agentic_concierge.infrastructure.llm_discovery import pick_larger_model
    # "custom-model" has no size tag → 999.0 → should be skipped
    result = pick_larger_model(
        ["qwen2.5:7b", "custom-model"],
        "qwen2.5:7b",
    )
    assert result is None


def test_pick_larger_model_returns_none_for_unparseable_current():
    """If the current model name is unparseable, escalation should not attempt."""
    from agentic_concierge.infrastructure.llm_discovery import pick_larger_model
    result = pick_larger_model(
        ["qwen2.5:7b", "qwen2.5:14b"],
        "custom-model",
    )
    assert result is None


def test_pick_larger_model_excludes_current_model():
    """Current model should not be returned even if it's in the list."""
    from agentic_concierge.infrastructure.llm_discovery import pick_larger_model
    result = pick_larger_model(
        ["qwen2.5:7b"],
        "qwen2.5:7b",
    )
    assert result is None


# ---------------------------------------------------------------------------
# Persistent loop escalation trigger tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_persistent_loop_triggers_escalation(tmp_path):
    """When the same tool+args is repeated beyond the threshold even after the
    loop warning, escalation to a larger model should be attempted."""
    # The loop detection threshold is 2 repeats in the last 8 calls.
    # First repeat at threshold → warning injected.
    # Second repeat (> threshold) → escalation triggered.
    responses = [
        # Step 0: initial tool call
        _tool_response("shell", call_id="c0", command="echo hello"),
        # Steps 1-2: same call repeated → loop detected at step 2 (threshold hit)
        _tool_response("shell", call_id="c1", command="echo hello"),
        _tool_response("shell", call_id="c2", command="echo hello"),
        # Step 3: same call again (> threshold) → triggers persistent_loop escalation
        _tool_response("shell", call_id="c3", command="echo hello"),
        # After escalation: different approach, then finish
        _tool_response("list_files", call_id="c4"),
        _finish_response(call_id="c5"),
    ]
    result = await _run_escalation(
        responses,
        tmp_path=tmp_path,
        available_models=["qwen2.5:7b", "qwen2.5:14b"],
        max_escalations=2,
    )
    assert result.payload["action"] == "final"
    assert result.payload["summary"] == "Done"

    events = _read_runlog(result.run_dir)
    escalated = [e for e in events if e["kind"] == "model_escalated"]
    assert len(escalated) == 1
    assert escalated[0]["payload"]["trigger"] == "persistent_loop"
    assert escalated[0]["payload"]["from_model"] == "qwen2.5:7b"
    assert escalated[0]["payload"]["to_model"] == "qwen2.5:14b"

    # Loop detected events should also be present
    loop_events = [e for e in events if e["kind"] == "loop_detected"]
    assert len(loop_events) >= 2


@pytest.mark.asyncio
async def test_persistent_loop_no_larger_model_continues(tmp_path):
    """When loop persists but no larger model is available, the loop warning is
    still injected but no escalation occurs."""
    responses = [
        _tool_response("shell", call_id="c0", command="echo hello"),
        _tool_response("shell", call_id="c1", command="echo hello"),
        _tool_response("shell", call_id="c2", command="echo hello"),
        _tool_response("shell", call_id="c3", command="echo hello"),
        # Eventually does something different
        _tool_response("list_files", call_id="c4"),
        _finish_response(call_id="c5"),
    ]
    result = await _run_escalation(
        responses,
        tmp_path=tmp_path,
        available_models=["qwen2.5:7b"],  # no larger model
        max_escalations=2,
    )
    assert result.payload["action"] == "final"

    events = _read_runlog(result.run_dir)
    escalated = [e for e in events if e["kind"] == "model_escalated"]
    assert len(escalated) == 0
    loop_events = [e for e in events if e["kind"] == "loop_detected"]
    assert len(loop_events) >= 2


# ---------------------------------------------------------------------------
# Cloud fallback wrapper preservation tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rebuild_chat_client_preserves_fallback_wrapper():
    """_rebuild_chat_client should preserve FallbackChatClient wrapping."""
    from agentic_concierge.application.execute_task import _rebuild_chat_client
    from agentic_concierge.infrastructure.chat.fallback import FallbackChatClient, FallbackPolicy
    from agentic_concierge.config import ModelConfig

    model_cfg = ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5:14b")
    local_mock = object()
    cloud_mock = object()
    policy = FallbackPolicy("no_tool_calls")
    wrapped = FallbackChatClient(local=local_mock, cloud=cloud_mock, cloud_model="gpt-4o", policy=policy)

    result = _rebuild_chat_client(model_cfg, wrapped)
    assert isinstance(result, FallbackChatClient)
    assert result._cloud is cloud_mock
    assert result._cloud_model == "gpt-4o"
    assert result._policy is policy
    # The local client should be a NEW client (not the old mock)
    assert result._local is not local_mock


@pytest.mark.asyncio
async def test_rebuild_chat_client_returns_plain_client_without_wrapper():
    """_rebuild_chat_client with a plain client should return a plain client."""
    from agentic_concierge.application.execute_task import _rebuild_chat_client
    from agentic_concierge.config import ModelConfig
    from agentic_concierge.infrastructure.ollama import OllamaChatClient

    model_cfg = ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5:14b")
    plain_client = OllamaChatClient(base_url="http://localhost:11434/v1", timeout_s=5.0)

    result = _rebuild_chat_client(model_cfg, plain_client)
    # Should NOT be wrapped in FallbackChatClient
    from agentic_concierge.infrastructure.chat.fallback import FallbackChatClient
    assert not isinstance(result, FallbackChatClient)


# ---------------------------------------------------------------------------
# _attempt_escalation unit tests
# ---------------------------------------------------------------------------

def test_attempt_escalation_returns_none_when_disabled():
    """max_escalations=0 should always return None."""
    from agentic_concierge.application.execute_task import _attempt_escalation
    from agentic_concierge.config import ModelConfig
    from unittest.mock import MagicMock

    model_cfg = ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5:7b")
    result = _attempt_escalation(
        trigger="plain_text_exhaustion",
        model_cfg=model_cfg,
        available_models=["qwen2.5:7b", "qwen2.5:14b"],
        escalation_count=0,
        escalated_triggers=set(),
        max_escalations=0,
        run_repository=MagicMock(),
        run_id=MagicMock(),
        event_queue=None,
        step_key="step_0",
    )
    assert result is None


def test_attempt_escalation_returns_none_at_cap():
    """When escalation_count >= max_escalations, should return None."""
    from agentic_concierge.application.execute_task import _attempt_escalation
    from agentic_concierge.config import ModelConfig
    from unittest.mock import MagicMock

    model_cfg = ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5:7b")
    result = _attempt_escalation(
        trigger="plain_text_exhaustion",
        model_cfg=model_cfg,
        available_models=["qwen2.5:7b", "qwen2.5:14b"],
        escalation_count=2,
        escalated_triggers=set(),
        max_escalations=2,
        run_repository=MagicMock(),
        run_id=MagicMock(),
        event_queue=None,
        step_key="step_0",
    )
    assert result is None


def test_attempt_escalation_returns_none_for_duplicate_trigger():
    """When the trigger has already fired, should return None."""
    from agentic_concierge.application.execute_task import _attempt_escalation
    from agentic_concierge.config import ModelConfig
    from unittest.mock import MagicMock

    model_cfg = ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5:7b")
    result = _attempt_escalation(
        trigger="plain_text_exhaustion",
        model_cfg=model_cfg,
        available_models=["qwen2.5:7b", "qwen2.5:14b"],
        escalation_count=1,
        escalated_triggers={"plain_text_exhaustion"},
        max_escalations=2,
        run_repository=MagicMock(),
        run_id=MagicMock(),
        event_queue=None,
        step_key="step_0",
    )
    assert result is None


def test_attempt_escalation_returns_new_config_on_success():
    """Successful escalation should return a new ModelConfig with the larger model."""
    from agentic_concierge.application.execute_task import _attempt_escalation
    from agentic_concierge.config import ModelConfig
    from unittest.mock import MagicMock

    model_cfg = ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5:7b")
    mock_repo = MagicMock()
    mock_run_id = MagicMock()

    result = _attempt_escalation(
        trigger="plain_text_exhaustion",
        model_cfg=model_cfg,
        available_models=["qwen2.5:7b", "qwen2.5:14b"],
        escalation_count=0,
        escalated_triggers=set(),
        max_escalations=2,
        run_repository=mock_repo,
        run_id=mock_run_id,
        event_queue=None,
        step_key="step_0",
    )
    assert result is not None
    assert result.model == "qwen2.5:14b"
    assert result.base_url == model_cfg.base_url
    assert result.backend == model_cfg.backend
    # Verify runlog event was emitted
    mock_repo.append_event.assert_called_once()
    call_args = mock_repo.append_event.call_args
    assert call_args[0][1] == "model_escalated"
    assert call_args[0][2]["trigger"] == "plain_text_exhaustion"
    assert call_args[0][2]["from_model"] == "qwen2.5:7b"
    assert call_args[0][2]["to_model"] == "qwen2.5:14b"


def test_attempt_escalation_returns_none_no_larger_model():
    """When no larger model is available, should return None."""
    from agentic_concierge.application.execute_task import _attempt_escalation
    from agentic_concierge.config import ModelConfig
    from unittest.mock import MagicMock

    model_cfg = ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5:14b")
    result = _attempt_escalation(
        trigger="plain_text_exhaustion",
        model_cfg=model_cfg,
        available_models=["qwen2.5:7b", "qwen2.5:14b"],
        escalation_count=0,
        escalated_triggers=set(),
        max_escalations=2,
        run_repository=MagicMock(),
        run_id=MagicMock(),
        event_queue=None,
        step_key="step_0",
    )
    assert result is None


def test_attempt_escalation_scales_timeout_for_larger_model():
    """Escalation to a larger model should scale the timeout proportionally."""
    from agentic_concierge.application.execute_task import _attempt_escalation
    from agentic_concierge.config import ModelConfig
    from unittest.mock import MagicMock

    model_cfg = ModelConfig(
        base_url="http://localhost:11434/v1", model="qwen2.5:7b",
        timeout_s=120.0,
    )
    result = _attempt_escalation(
        trigger="plain_text_exhaustion",
        model_cfg=model_cfg,
        available_models=["qwen2.5:7b", "qwen2.5:14b"],
        escalation_count=0,
        escalated_triggers=set(),
        max_escalations=2,
        run_repository=MagicMock(),
        run_id=MagicMock(),
        event_queue=None,
        step_key="step_0",
    )
    assert result is not None
    # 14b is 2x larger than 7b, so timeout should scale to 240s
    assert result.timeout_s == 240.0
