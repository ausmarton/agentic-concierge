"""Tests for capability-driven orchestrator routing (ADR-028)."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agentic_concierge.application.orchestrator import (
    OrchestrationPlan,
    SpecialistBrief,
    _resolve_specialist_from_capabilities,
    orchestrate_task,
)
from agentic_concierge.config import DEFAULT_CONFIG
from agentic_concierge.domain import LLMResponse, ToolCallRequest
from agentic_concierge.infrastructure.ollama import OllamaChatClient


# ---------------------------------------------------------------------------
# _resolve_specialist_from_capabilities
# ---------------------------------------------------------------------------


def test_resolve_code_caps_to_engineering():
    """code_python capability should resolve to engineering template."""
    result = _resolve_specialist_from_capabilities(["code_python"])
    assert result == "engineering"


def test_resolve_web_caps_to_research():
    """web_comprehension should resolve to research template."""
    result = _resolve_specialist_from_capabilities(["web_comprehension"])
    assert result == "research"


def test_resolve_summarisation_to_research():
    """summarisation + instruction_following → research."""
    result = _resolve_specialist_from_capabilities(["summarisation", "instruction_following"])
    assert result == "research"


def test_resolve_mixed_code_and_web():
    """When both code and web needed, the template with highest total score wins."""
    result = _resolve_specialist_from_capabilities(["code_python", "web_comprehension"])
    # engineering has code_python=0.7 + web_comprehension=0 = 0.7
    # research has code_python=0 + web_comprehension=0.7 = 0.7
    # tie-breaking depends on iteration order; both are valid
    assert result in ("engineering", "research")


def test_resolve_empty_caps_returns_a_template():
    """Empty capabilities list still returns a valid template."""
    from agentic_concierge.infrastructure.specialists.dynamic_pack import PACK_TEMPLATES
    result = _resolve_specialist_from_capabilities([])
    assert result in PACK_TEMPLATES


def test_resolve_unknown_caps_returns_a_template():
    """Unknown capability names still produce a valid template (best-effort)."""
    from agentic_concierge.infrastructure.specialists.dynamic_pack import PACK_TEMPLATES
    result = _resolve_specialist_from_capabilities(["quantum_computing", "telepathy"])
    assert result in PACK_TEMPLATES


# ---------------------------------------------------------------------------
# SpecialistBrief.required_capabilities field
# ---------------------------------------------------------------------------


def test_specialist_brief_has_capabilities_field():
    """SpecialistBrief supports required_capabilities."""
    brief = SpecialistBrief(
        specialist_id="research",
        brief="test",
        required_capabilities=["web_comprehension", "summarisation"],
    )
    assert brief.required_capabilities == ["web_comprehension", "summarisation"]


def test_specialist_brief_capabilities_default_none():
    """required_capabilities defaults to None."""
    brief = SpecialistBrief(specialist_id="engineering", brief="test")
    assert brief.required_capabilities is None


# ---------------------------------------------------------------------------
# Integration: orchestrate_task with capability-driven routing
# ---------------------------------------------------------------------------


async def _call_orchestrate(mock_response: LLMResponse, prompt: str) -> OrchestrationPlan:
    with patch.object(OllamaChatClient, "chat", new_callable=AsyncMock, return_value=mock_response):
        client = OllamaChatClient(base_url="http://localhost:11434/v1", timeout_s=5.0)
        return await orchestrate_task(prompt, DEFAULT_CONFIG, chat_client=client, model="m")


@pytest.mark.asyncio
async def test_capabilities_resolve_to_specialist_id():
    """When LLM provides required_capabilities without specialist_id, system resolves it."""
    resp = LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            call_id="orch0",
            tool_name="create_plan",
            arguments={
                "assignments": [
                    {
                        "brief": "find the current price of AMZN stock",
                        "required_capabilities": ["web_comprehension", "summarisation"],
                    }
                ],
                "mode": "sequential",
                "synthesis_required": False,
                "reasoning": "simple web lookup",
            },
        )],
    )
    plan = await _call_orchestrate(resp, "What is AMZN stock price?")
    assert len(plan.specialist_assignments) == 1
    # web_comprehension + summarisation → research
    assert plan.specialist_assignments[0].specialist_id == "research"
    assert plan.specialist_assignments[0].required_capabilities == [
        "web_comprehension", "summarisation",
    ]


@pytest.mark.asyncio
async def test_capabilities_code_resolves_to_engineering():
    """code_python capability resolves to engineering template."""
    resp = LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            call_id="orch0",
            tool_name="create_plan",
            arguments={
                "assignments": [
                    {
                        "brief": "build a REST API",
                        "required_capabilities": ["code_python", "structured_output"],
                    }
                ],
                "mode": "sequential",
                "synthesis_required": False,
                "reasoning": "code generation task",
            },
        )],
    )
    plan = await _call_orchestrate(resp, "Build a REST API")
    assert plan.specialist_assignments[0].specialist_id == "engineering"


@pytest.mark.asyncio
async def test_explicit_specialist_id_overrides_capabilities():
    """When both specialist_id and capabilities are provided, specialist_id is used."""
    resp = LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            call_id="orch0",
            tool_name="create_plan",
            arguments={
                "assignments": [
                    {
                        "specialist_id": "engineering",
                        "brief": "test task",
                        "required_capabilities": ["web_comprehension"],
                    }
                ],
                "mode": "sequential",
                "synthesis_required": False,
                "reasoning": "explicit template",
            },
        )],
    )
    plan = await _call_orchestrate(resp, "test")
    # specialist_id is set so it takes precedence, not resolved from capabilities
    assert plan.specialist_assignments[0].specialist_id == "engineering"
    # capabilities are still stored
    assert plan.specialist_assignments[0].required_capabilities == ["web_comprehension"]


@pytest.mark.asyncio
async def test_create_plan_schema_has_capabilities():
    """The create_plan tool schema includes required_capabilities field."""
    from agentic_concierge.application.orchestrator import _build_orchestrator_tool_def
    tool_def = _build_orchestrator_tool_def()
    props = tool_def["function"]["parameters"]["properties"]["assignments"]["items"]["properties"]
    assert "required_capabilities" in props
    assert props["required_capabilities"]["type"] == "array"


@pytest.mark.asyncio
async def test_create_plan_brief_is_required_but_specialist_id_optional():
    """brief is required but specialist_id is now optional (capabilities can resolve it)."""
    from agentic_concierge.application.orchestrator import _build_orchestrator_tool_def
    tool_def = _build_orchestrator_tool_def()
    required = tool_def["function"]["parameters"]["properties"]["assignments"]["items"]["required"]
    assert "brief" in required
    # specialist_id is no longer required (can be resolved from capabilities)
    assert "specialist_id" not in required


@pytest.mark.asyncio
async def test_orchestrator_prompt_mentions_capabilities():
    """The orchestrator system prompt includes capability descriptions."""
    from agentic_concierge.application.orchestrator import _build_orchestrator_messages
    messages = _build_orchestrator_messages("test task", DEFAULT_CONFIG)
    system_msg = messages[0]["content"]
    assert "code_python" in system_msg
    assert "web_comprehension" in system_msg
    assert "reasoning" in system_msg
    assert "capabilities" in system_msg.lower()
