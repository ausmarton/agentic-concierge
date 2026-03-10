"""Tests for the orchestrator: orchestrate_task(), OrchestrationPlan.

All tests use AsyncMock to avoid real LLM calls.  They verify:
- Valid create_plan response → OrchestrationPlan with routing_method='orchestrator'.
- Brief assignment propagated to plan.
- synthesis_required forced True for multiple specialists.
- Fallback on no tool call / wrong tool / exception.
- Unknown specialist IDs filtered.
- No valid assignments after filtering → fallback.
- required_capabilities derived from specialist configs.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agentic_concierge.application.orchestrator import (
    OrchestrationPlan,
    SpecialistBrief,
    orchestrate_task,
)
from agentic_concierge.config import DEFAULT_CONFIG, ConciergeConfig
from agentic_concierge.config.schema import ModelConfig, SpecialistConfig
from agentic_concierge.domain import LLMResponse, ToolCallRequest
from agentic_concierge.infrastructure.ollama import OllamaChatClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plan_response(
    specialist_ids: list[str],
    mode: str = "sequential",
    synthesis_required: bool = False,
    reasoning: str = "test",
) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            call_id="orch0",
            tool_name="create_plan",
            arguments={
                "assignments": [
                    {"specialist_id": sid, "brief": f"brief for {sid}"}
                    for sid in specialist_ids
                ],
                "mode": mode,
                "synthesis_required": synthesis_required,
                "reasoning": reasoning,
            },
        )],
    )


def _config_with_two_specialists() -> ConciergeConfig:
    return ConciergeConfig(
        models={"fast": ModelConfig(base_url="http://localhost:11434/v1", model="m")},
        specialists={
            "engineering": SpecialistConfig(
                description="Engineering",
                workflow="engineering",
                capabilities=["code_execution", "file_io"],
            ),
            "research": SpecialistConfig(
                description="Research",
                workflow="research",
                capabilities=["systematic_review", "web_search"],
            ),
        },
    )


async def _call_orchestrate(mock_response: LLMResponse, prompt: str, config=None) -> OrchestrationPlan:
    cfg = config or DEFAULT_CONFIG
    with patch.object(OllamaChatClient, "chat", new_callable=AsyncMock, return_value=mock_response):
        client = OllamaChatClient(base_url="http://localhost:11434/v1", timeout_s=5.0)
        return await orchestrate_task(prompt, cfg, chat_client=client, model="m")


# ---------------------------------------------------------------------------
# Successful create_plan responses
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrate_single_specialist_plan():
    """Valid create_plan for one specialist returns routing_method='orchestrator'."""
    plan = await _call_orchestrate(
        _plan_response(["engineering"]),
        "build a Python service",
    )
    assert len(plan.specialist_assignments) == 1
    assert plan.specialist_assignments[0].specialist_id == "engineering"
    assert plan.routing_method == "orchestrator"


@pytest.mark.asyncio
async def test_orchestrate_brief_propagated():
    """Briefs from create_plan are stored in specialist assignments."""
    plan = await _call_orchestrate(
        _plan_response(["engineering"]),
        "build a REST API",
    )
    assert plan.specialist_assignments[0].brief == "brief for engineering"


@pytest.mark.asyncio
async def test_orchestrate_mode_sequential():
    plan = await _call_orchestrate(
        _plan_response(["engineering"], mode="sequential"),
        "build a service",
    )
    assert plan.mode == "sequential"


@pytest.mark.asyncio
async def test_orchestrate_mode_parallel():
    """Parallel mode is accepted from create_plan."""
    cfg = _config_with_two_specialists()
    plan = await _call_orchestrate(
        _plan_response(["engineering", "research"], mode="parallel"),
        "build and research",
        config=cfg,
    )
    assert plan.mode == "parallel"


@pytest.mark.asyncio
async def test_orchestrate_synthesis_required_forced_true_for_multi_specialist():
    """synthesis_required is forced True when >1 specialist assigned."""
    cfg = _config_with_two_specialists()
    # LLM returns synthesis_required=False, but the orchestrator forces True for multi-pack.
    plan = await _call_orchestrate(
        _plan_response(["engineering", "research"], synthesis_required=False),
        "build and research",
        config=cfg,
    )
    assert plan.synthesis_required is True


@pytest.mark.asyncio
async def test_orchestrate_synthesis_false_for_single_specialist():
    """synthesis_required stays False for a single-specialist plan."""
    plan = await _call_orchestrate(
        _plan_response(["engineering"], synthesis_required=False),
        "build a service",
    )
    assert plan.synthesis_required is False


@pytest.mark.asyncio
async def test_orchestrate_required_capabilities_derived():
    """required_capabilities is derived from the assigned specialists' config capabilities."""
    cfg = _config_with_two_specialists()
    plan = await _call_orchestrate(
        _plan_response(["engineering", "research"]),
        "build and research",
        config=cfg,
    )
    assert "code_execution" in plan.required_capabilities
    assert "systematic_review" in plan.required_capabilities


# ---------------------------------------------------------------------------
# Unknown / empty assignments → fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrate_unknown_specialist_id_filtered():
    """Unknown specialist IDs in assignments are silently filtered."""
    unknown_response = LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            call_id="o0",
            tool_name="create_plan",
            arguments={
                "assignments": [
                    {"specialist_id": "nonexistent", "brief": ""},
                    {"specialist_id": "engineering", "brief": "do stuff"},
                ],
                "mode": "sequential",
                "synthesis_required": False,
                "reasoning": "test",
            },
        )],
    )
    plan = await _call_orchestrate(unknown_response, "build something")
    # Only engineering should survive; nonexistent filtered out
    assert len(plan.specialist_assignments) == 1
    assert plan.specialist_assignments[0].specialist_id == "engineering"


@pytest.mark.asyncio
async def test_orchestrate_all_unknown_ids_falls_back():
    """When all assigned specialist IDs are unknown, fall back to template."""
    all_unknown = LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            call_id="o0",
            tool_name="create_plan",
            arguments={
                "assignments": [{"specialist_id": "ghost", "brief": ""}],
                "mode": "sequential",
                "synthesis_required": False,
                "reasoning": "test",
            },
        )],
    )
    with patch.object(
        OllamaChatClient, "chat", new_callable=AsyncMock,
        return_value=all_unknown,
    ):
        client = OllamaChatClient(base_url="http://localhost:11434/v1", timeout_s=5.0)
        plan = await orchestrate_task("build", DEFAULT_CONFIG, chat_client=client, model="m")

    # Fallback routes to first template
    assert len(plan.specialist_assignments) >= 1
    assert plan.routing_method == "template_fallback"


# ---------------------------------------------------------------------------
# Fallback paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrate_fallback_on_no_tool_call():
    """When LLM returns no tool call, fall back to template fallback."""
    no_tool = LLMResponse(content="I'll plan this", tool_calls=[])
    with patch.object(
        OllamaChatClient, "chat", new_callable=AsyncMock,
        return_value=no_tool,
    ):
        client = OllamaChatClient(base_url="http://localhost:11434/v1", timeout_s=5.0)
        plan = await orchestrate_task("build a service", DEFAULT_CONFIG, chat_client=client, model="m")

    assert plan.routing_method == "template_fallback"


@pytest.mark.asyncio
async def test_orchestrate_fallback_on_wrong_tool_name():
    """When LLM calls a different tool (not create_plan), fall back to template."""
    wrong_tool = LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(call_id="x", tool_name="select_capabilities",
                                    arguments={"capabilities": ["code_execution"]})],
    )
    with patch.object(
        OllamaChatClient, "chat", new_callable=AsyncMock,
        return_value=wrong_tool,
    ):
        client = OllamaChatClient(base_url="http://localhost:11434/v1", timeout_s=5.0)
        plan = await orchestrate_task("build a service", DEFAULT_CONFIG, chat_client=client, model="m")

    assert plan.routing_method == "template_fallback"


@pytest.mark.asyncio
async def test_orchestrate_fallback_on_exception():
    """When chat() raises, fall back to template gracefully."""
    with patch.object(
        OllamaChatClient, "chat", new_callable=AsyncMock,
        side_effect=RuntimeError("connection failed"),
    ):
        client = OllamaChatClient(base_url="http://localhost:11434/v1", timeout_s=5.0)
        plan = await orchestrate_task("build a service", DEFAULT_CONFIG, chat_client=client, model="m")

    assert len(plan.specialist_assignments) >= 1
    assert plan.routing_method == "template_fallback"
