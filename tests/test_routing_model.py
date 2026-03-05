"""Tests for routing_model_key config field in ConciergeConfig.

Verifies that:
- execute_task uses the routing model (config.routing_model_key) for the
  llm_recruit_specialist call, not the task execution model.
- The correct model name reaches chat_client.chat() at each call index.
- Fallback to the task model occurs when routing_model_key is absent from config.models.
- The field default is "fast".
"""
from __future__ import annotations

from typing import List
from unittest.mock import AsyncMock, patch

import pytest

from agentic_concierge.application.execute_task import execute_task
from agentic_concierge.config import DEFAULT_CONFIG, ConciergeConfig
from agentic_concierge.config.schema import ModelConfig, SpecialistConfig
from agentic_concierge.domain import LLMResponse, Task, ToolCallRequest
from agentic_concierge.infrastructure.ollama import OllamaChatClient
from agentic_concierge.infrastructure.specialists import ConfigSpecialistRegistry
from agentic_concierge.infrastructure.workspace import FileSystemRunRepository


# ---------------------------------------------------------------------------
# Helpers (duplicated from test_task_force.py — small enough to inline)
# ---------------------------------------------------------------------------

def _routing_response(caps: list[str] | None = None) -> LLMResponse:
    """Mock LLM routing response for llm_recruit_specialist (single cap → engineering)."""
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            call_id="r0",
            tool_name="select_capabilities",
            arguments={"capabilities": caps or ["code_execution"]},
        )],
    )


def _create_plan_response(specialist_ids: list[str] | None = None, mode: str = "sequential") -> LLMResponse:
    """Mock orchestrator create_plan response (Phase 12: orchestrate_task uses this)."""
    sids = specialist_ids or ["engineering"]
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            call_id="orch0",
            tool_name="create_plan",
            arguments={
                "assignments": [{"specialist_id": sid, "brief": ""} for sid in sids],
                "mode": mode,
                "synthesis_required": len(sids) > 1,
                "reasoning": "test routing",
            },
        )],
    )


def _tool_resp(call_id: str = "t0") -> LLMResponse:
    """A list_files call to satisfy the 'prior tool call' structural requirement."""
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(call_id=call_id, tool_name="list_files", arguments={})],
    )


def _eng_finish(call_id: str = "c1") -> LLMResponse:
    """Engineering pack finish_task response."""
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            call_id=call_id,
            tool_name="finish_task",
            arguments={
                "summary": "done",
                "artifacts": [],
                "next_steps": [],
                "notes": "",
                "tests_verified": True,
            },
        )],
    )


def _make_capturing_side_effect(
    responses: list[LLMResponse],
    captured_models: list[str],
):
    """Return an async side_effect function that records the `model` arg of each call.

    llm_recruit_specialist calls chat(messages, model, tools=...) positionally,
    while the pack loop calls chat(messages=..., model=...) with kwargs.
    We handle both by checking args[1] first, then the kwarg.
    """
    responses_iter = iter(responses)

    async def _side_effect(*args, **kwargs):
        # Positional: chat(messages, model, ...) → args[1]
        # Keyword:    chat(messages=..., model=...) → kwargs["model"]
        model_val = args[1] if len(args) > 1 else kwargs.get("model", "")
        captured_models.append(model_val)
        return next(responses_iter)

    return _side_effect


async def _run_with_captured_models(
    config: ConciergeConfig,
    mock_responses: list[LLMResponse],
    prompt: str,
    tmp_path,
) -> tuple:
    """Run execute_task and return (result, call_models).

    call_models is a list of model names passed to chat_client.chat() in call order:
    index 0 = routing call, index 1+ = pack loop calls.
    """
    run_repository = FileSystemRunRepository(workspace_root=str(tmp_path))
    specialist_registry = ConfigSpecialistRegistry(config)
    captured_models: list[str] = []

    side_effect = _make_capturing_side_effect(mock_responses, captured_models)

    with patch.object(OllamaChatClient, "chat", new_callable=AsyncMock, side_effect=side_effect):
        chat_client = OllamaChatClient(base_url="http://localhost:11434/v1", timeout_s=5.0)
        task = Task(prompt=prompt, specialist_id=None, model_key="quality", network_allowed=False)
        result = await execute_task(
            task,
            chat_client=chat_client,
            run_repository=run_repository,
            specialist_registry=specialist_registry,
            config=config,
            max_steps=10,
            max_review_iterations=0,
        )

    return result, captured_models


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_routing_uses_fast_model_by_default(tmp_path):
    """DEFAULT_CONFIG routes via 'fast' model (qwen2.5:7b), runs task on 'quality' (qwen2.5:14b)."""
    # DEFAULT_CONFIG: routing_model_key="fast" → qwen2.5:7b; task model_key="quality" → qwen2.5:14b
    # Phase 12: orchestrate_task is called first (routing model), then the pack loop (task model).
    # _create_plan_response() satisfies the create_plan tool call so orchestrate_task succeeds
    # without falling back to llm_recruit_specialist.
    result, call_models = await _run_with_captured_models(
        DEFAULT_CONFIG,
        [_create_plan_response(["engineering"]), _tool_resp(), _eng_finish()],
        "build a Python service",
        tmp_path,
    )

    assert len(call_models) >= 2
    assert call_models[0] == "qwen2.5:7b",  f"routing call used {call_models[0]!r}, expected qwen2.5:7b"
    assert call_models[1] == "qwen2.5:14b", f"pack loop call used {call_models[1]!r}, expected qwen2.5:14b"


@pytest.mark.asyncio
async def test_routing_uses_explicit_routing_model_key(tmp_path):
    """A custom routing_model_key is used for the routing call."""
    config = ConciergeConfig(
        models={
            "fast": ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5:7b"),
            "quality": ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5:14b"),
            "routing": ModelConfig(base_url="http://localhost:11434/v1", model="llama3.1:8b"),
        },
        specialists={
            "engineering": SpecialistConfig(
                description="Engineering",
                workflow="engineering",
                capabilities=["code_execution", "file_io", "software_testing"],
            ),
        },
        routing_model_key="routing",
    )

    result, call_models = await _run_with_captured_models(
        config,
        [_create_plan_response(["engineering"]), _tool_resp(), _eng_finish()],
        "build a Python service",
        tmp_path,
    )

    assert len(call_models) >= 1
    assert call_models[0] == "llama3.1:8b", f"routing call used {call_models[0]!r}, expected llama3.1:8b"


@pytest.mark.asyncio
async def test_routing_falls_back_to_task_model_when_key_missing(tmp_path):
    """When routing_model_key is absent from config.models, the task model is used for routing."""
    config = ConciergeConfig(
        models={
            "quality": ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5:14b"),
        },
        specialists={
            "engineering": SpecialistConfig(
                description="Engineering",
                workflow="engineering",
                capabilities=["code_execution", "file_io", "software_testing"],
            ),
        },
        routing_model_key="nonexistent",
    )

    # routing_model_key='nonexistent' → execute_task falls back to task model for orchestrate_task.
    # _create_plan_response() satisfies the create_plan call so orchestrate_task succeeds.
    result, call_models = await _run_with_captured_models(
        config,
        [_create_plan_response(["engineering"]), _tool_resp(), _eng_finish()],
        "build a Python service",
        tmp_path,
    )

    assert len(call_models) >= 2
    # Both routing and pack loop should use the same (task) model.
    assert call_models[0] == call_models[1], (
        f"expected fallback: routing model {call_models[0]!r} == task model {call_models[1]!r}"
    )


def test_routing_model_key_field_default():
    """routing_model_key defaults to 'fast' in DEFAULT_CONFIG and in freshly constructed configs."""
    assert DEFAULT_CONFIG.routing_model_key == "fast"

    # A ConciergeConfig constructed without explicit routing_model_key also has "fast".
    config = ConciergeConfig(
        models={
            "fast": ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5:7b"),
        },
        specialists={
            "engineering": SpecialistConfig(
                description="Engineering",
                workflow="engineering",
                capabilities=["code_execution"],
            ),
        },
    )
    assert config.routing_model_key == "fast"
