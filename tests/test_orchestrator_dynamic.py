"""Tests for orchestrator dynamic pack composition (tools/role on SpecialistBrief)."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agentic_concierge.application.orchestrator import (
    OrchestrationPlan,
    SpecialistBrief,
    _collapse_redundant_dynamic,
    _dedup_same_id,
    orchestrate_task,
)
from agentic_concierge.config import DEFAULT_CONFIG
from agentic_concierge.domain import LLMResponse, ToolCallRequest
from agentic_concierge.infrastructure.ollama import OllamaChatClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dynamic_plan_response(
    tools: list[str],
    role: str = "A dynamic agent",
    specialist_id: str = "dynamic",
) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            call_id="orch0",
            tool_name="create_plan",
            arguments={
                "assignments": [
                    {
                        "specialist_id": specialist_id,
                        "brief": "do the work",
                        "tools": tools,
                        "role": role,
                    }
                ],
                "mode": "sequential",
                "synthesis_required": False,
                "reasoning": "dynamic composition",
            },
        )],
    )


async def _call_orchestrate(mock_response: LLMResponse, prompt: str) -> OrchestrationPlan:
    with patch.object(OllamaChatClient, "chat", new_callable=AsyncMock, return_value=mock_response):
        client = OllamaChatClient(base_url="http://localhost:11434/v1", timeout_s=5.0)
        return await orchestrate_task(prompt, DEFAULT_CONFIG, chat_client=client, model="m")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dynamic_pack_assignment_preserves_tools():
    """When create_plan includes tools, they are stored on the SpecialistBrief."""
    plan = await _call_orchestrate(
        _dynamic_plan_response(["shell", "web_search", "write_file"]),
        "search the web and write a script",
    )
    assert len(plan.specialist_assignments) == 1
    brief = plan.specialist_assignments[0]
    assert brief.tools == ["shell", "web_search", "write_file"]


@pytest.mark.asyncio
async def test_dynamic_pack_assignment_preserves_role():
    """When create_plan includes a role, it is stored on the SpecialistBrief."""
    plan = await _call_orchestrate(
        _dynamic_plan_response(["shell"], role="You are a deployment specialist"),
        "deploy the service",
    )
    brief = plan.specialist_assignments[0]
    assert brief.role == "You are a deployment specialist"


@pytest.mark.asyncio
async def test_dynamic_pack_specialist_id():
    """Dynamic assignments use the specialist_id from the plan."""
    plan = await _call_orchestrate(
        _dynamic_plan_response(["shell", "write_file"], specialist_id="dynamic"),
        "custom task",
    )
    assert plan.specialist_assignments[0].specialist_id == "dynamic"


@pytest.mark.asyncio
async def test_specialist_brief_tools_none_for_template():
    """Template-based assignments have tools=None by default."""
    resp = LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            call_id="orch0",
            tool_name="create_plan",
            arguments={
                "assignments": [{"specialist_id": "engineering", "brief": "build it"}],
                "mode": "sequential",
                "synthesis_required": False,
                "reasoning": "standard",
            },
        )],
    )
    plan = await _call_orchestrate(resp, "build a service")
    assert plan.specialist_assignments[0].tools is None
    assert plan.specialist_assignments[0].role is None


@pytest.mark.asyncio
async def test_specialist_brief_dataclass_fields():
    """SpecialistBrief has the expected optional fields."""
    brief = SpecialistBrief(
        specialist_id="dynamic",
        brief="test",
        tools=["shell", "write_file"],
        role="Test agent",
    )
    assert brief.tools == ["shell", "write_file"]
    assert brief.role == "Test agent"


@pytest.mark.asyncio
async def test_dynamic_routing_method_is_orchestrator():
    """Dynamic pack assignments still use routing_method='orchestrator'."""
    plan = await _call_orchestrate(
        _dynamic_plan_response(["shell", "web_search"]),
        "mixed task",
    )
    assert plan.routing_method == "orchestrator"


# ---------------------------------------------------------------------------
# Plan deduplication tests (_collapse_redundant_dynamic)
# ---------------------------------------------------------------------------


def test_collapse_noop_single_assignment():
    """Single-specialist plans are never collapsed, even if subset of a template."""
    assignments = [SpecialistBrief(specialist_id="dynamic", brief="do it", tools=["shell"])]
    result = _collapse_redundant_dynamic(assignments)
    assert len(result) == 1
    assert result[0].specialist_id == "dynamic"


def test_collapse_noop_non_overlapping():
    """Multi-specialist plan where dynamic tools don't overlap any assigned template."""
    assignments = [
        SpecialistBrief(specialist_id="research", brief="search for info"),
        SpecialistBrief(specialist_id="dynamic", brief="run shell", tools=["shell", "run_tests"]),
    ]
    result = _collapse_redundant_dynamic(assignments)
    assert len(result) == 2
    assert result[1].specialist_id == "dynamic"


def test_collapse_merges_redundant_dynamic():
    """Dynamic pack with tools subset of already-assigned template is merged."""
    assignments = [
        SpecialistBrief(specialist_id="research", brief="find stock prices"),
        SpecialistBrief(
            specialist_id="dynamic",
            brief="summarise the findings",
            tools=["web_search", "fetch_url"],
        ),
    ]
    result = _collapse_redundant_dynamic(assignments)
    assert len(result) == 1
    assert result[0].specialist_id == "research"
    assert "summarise the findings" in result[0].brief


def test_collapse_preserves_unrelated_template():
    """Non-dynamic assignments are always preserved."""
    assignments = [
        SpecialistBrief(specialist_id="engineering", brief="build it"),
        SpecialistBrief(specialist_id="research", brief="look it up"),
    ]
    result = _collapse_redundant_dynamic(assignments)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_collapse_applied_in_orchestrate_task():
    """Integration: orchestrate_task collapses redundant dynamic packs."""
    resp = LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            call_id="orch0",
            tool_name="create_plan",
            arguments={
                "assignments": [
                    {"specialist_id": "research", "brief": "look up AXP stock price"},
                    {
                        "specialist_id": "dynamic",
                        "brief": "format the result",
                        "tools": ["web_search"],
                        "role": "formatter",
                    },
                ],
                "mode": "sequential",
                "synthesis_required": True,
                "reasoning": "research then format",
            },
        )],
    )
    plan = await _call_orchestrate(resp, "What is AXP stock price?")
    # dynamic pack with ["web_search"] is subset of research — collapsed
    assert len(plan.specialist_assignments) == 1
    assert plan.specialist_assignments[0].specialist_id == "research"
    assert "format the result" in plan.specialist_assignments[0].brief


# ---------------------------------------------------------------------------
# Same-ID deduplication tests (_dedup_same_id)
# ---------------------------------------------------------------------------


def test_dedup_merges_same_id():
    """Two research assignments are merged into one with combined briefs."""
    assignments = [
        SpecialistBrief(specialist_id="research", brief="look up KO"),
        SpecialistBrief(specialist_id="research", brief="look up BAC"),
    ]
    result = _dedup_same_id(assignments)
    assert len(result) == 1
    assert result[0].specialist_id == "research"
    assert "KO" in result[0].brief
    assert "BAC" in result[0].brief


def test_dedup_preserves_different_ids():
    """Different specialist IDs are preserved."""
    assignments = [
        SpecialistBrief(specialist_id="research", brief="search"),
        SpecialistBrief(specialist_id="engineering", brief="build"),
    ]
    result = _dedup_same_id(assignments)
    assert len(result) == 2


def test_dedup_noop_single():
    """Single assignment is returned unchanged."""
    assignments = [SpecialistBrief(specialist_id="research", brief="search")]
    result = _dedup_same_id(assignments)
    assert len(result) == 1


@pytest.mark.asyncio
async def test_dedup_applied_in_orchestrate_task():
    """Integration: orchestrate_task deduplicates same-ID specialists."""
    resp = LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            call_id="orch0",
            tool_name="create_plan",
            arguments={
                "assignments": [
                    {"specialist_id": "research", "brief": "look up KO stock price"},
                    {"specialist_id": "research", "brief": "look up BAC stock price"},
                ],
                "mode": "parallel",
                "synthesis_required": True,
                "reasoning": "research both stocks",
            },
        )],
    )
    plan = await _call_orchestrate(resp, "KO and BAC stock prices")
    assert len(plan.specialist_assignments) == 1
    assert plan.specialist_assignments[0].specialist_id == "research"
    assert "KO" in plan.specialist_assignments[0].brief
    assert "BAC" in plan.specialist_assignments[0].brief
