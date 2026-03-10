"""Tests for routing_model_key config field in ConciergeConfig.

V2: The planner (plan_task) replaces the V1 orchestrator. The routing model
is used for the planner call; the task model is used for the pack loop.

Verifies that:
- execute_task uses the routing model for the planner call.
- The correct model name reaches the planner and the pack loop.
- Fallback to the task model occurs when routing_model_key is absent from config.models.
- The field default is "fast".
"""
from __future__ import annotations

from typing import List
from unittest.mock import AsyncMock, patch

import pytest

from agentic_concierge.application.execute_task import execute_task
from agentic_concierge.application.planner import PlanResult
from agentic_concierge.application.task_graph import TaskGraph
from agentic_concierge.config import DEFAULT_CONFIG, ConciergeConfig
from agentic_concierge.config.schema import ModelConfig, SpecialistConfig
from agentic_concierge.domain import LLMResponse, Task, ToolCallRequest
from agentic_concierge.infrastructure.ollama import OllamaChatClient
from agentic_concierge.infrastructure.specialists import ConfigSpecialistRegistry
from agentic_concierge.infrastructure.workspace import FileSystemRunRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_resp(call_id: str = "t0") -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(call_id=call_id, tool_name="list_files", arguments={})],
    )


def _eng_finish(call_id: str = "c1") -> LLMResponse:
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


def _make_single_leaf_graph() -> TaskGraph:
    graph = TaskGraph.from_root("build a Python service", node_id="root")
    graph.add_child(
        "root", "Build the service",
        node_id="leaf1",
        required_capabilities=["code_python"],
    )
    graph.transition("root", "decomposing")
    graph.transition("root", "critiqued")
    return graph


def _make_plan_result(planner_model: str) -> PlanResult:
    return PlanResult(
        graph=_make_single_leaf_graph(),
        reasoning="Single task",
        critique_feedback=None,
        replan_count=0,
        planner_model=planner_model,
        critic_model=planner_model,
    )


async def _run_with_captured_models(
    config: ConciergeConfig,
    mock_responses: list[LLMResponse],
    prompt: str,
    tmp_path,
) -> tuple:
    """Run execute_task, capturing the planner_model and pack loop models."""
    run_repository = FileSystemRunRepository(workspace_root=str(tmp_path))
    specialist_registry = ConfigSpecialistRegistry(config)
    captured_models: list[str] = []
    captured_planner_model: list[str] = []

    async def _mock_plan_task(prompt, chat_client, planner_model, critic_model, **kwargs):
        captured_planner_model.append(planner_model)
        return _make_plan_result(planner_model)

    responses_iter = iter(mock_responses)

    async def _side_effect(*args, **kwargs):
        model_val = args[1] if len(args) > 1 else kwargs.get("model", "")
        captured_models.append(model_val)
        return next(responses_iter)

    with patch(
        "agentic_concierge.application.planner.plan_task",
        new_callable=AsyncMock,
        side_effect=_mock_plan_task,
    ), patch.object(OllamaChatClient, "chat", new_callable=AsyncMock, side_effect=_side_effect):
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

    return result, captured_planner_model, captured_models


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_routing_uses_fast_model_by_default(tmp_path):
    """DEFAULT_CONFIG routes via 'fast' model (qwen2.5:7b), runs task on 'quality' (qwen2.5:14b)."""
    result, planner_models, pack_models = await _run_with_captured_models(
        DEFAULT_CONFIG,
        [_tool_resp(), _eng_finish()],
        "build a Python service",
        tmp_path,
    )

    assert len(planner_models) == 1
    assert planner_models[0] == "qwen2.5:7b", (
        f"planner used {planner_models[0]!r}, expected qwen2.5:7b (routing model)"
    )
    assert len(pack_models) >= 2
    assert pack_models[0] == "qwen2.5:14b", (
        f"pack loop used {pack_models[0]!r}, expected qwen2.5:14b (task model)"
    )


@pytest.mark.asyncio
async def test_routing_uses_explicit_routing_model_key(tmp_path):
    """A custom routing_model_key is used for the planner call."""
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

    result, planner_models, pack_models = await _run_with_captured_models(
        config,
        [_tool_resp(), _eng_finish()],
        "build a Python service",
        tmp_path,
    )

    assert len(planner_models) == 1
    assert planner_models[0] == "llama3.1:8b", (
        f"planner used {planner_models[0]!r}, expected llama3.1:8b (routing model)"
    )


@pytest.mark.asyncio
async def test_routing_falls_back_to_task_model_when_key_missing(tmp_path):
    """When routing_model_key is absent from config.models, the task model is used."""
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

    result, planner_models, pack_models = await _run_with_captured_models(
        config,
        [_tool_resp(), _eng_finish()],
        "build a Python service",
        tmp_path,
    )

    assert len(planner_models) == 1
    # Planner should fall back to the task model (quality → qwen2.5:14b)
    assert planner_models[0] == "qwen2.5:14b", (
        f"planner used {planner_models[0]!r}, expected qwen2.5:14b (fallback to task model)"
    )


def test_routing_model_key_field_default():
    """routing_model_key defaults to 'fast' in DEFAULT_CONFIG and in freshly constructed configs."""
    assert DEFAULT_CONFIG.routing_model_key == "fast"

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
