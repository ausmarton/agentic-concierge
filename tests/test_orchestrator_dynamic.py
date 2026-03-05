"""Tests for orchestrator dynamic pack composition (tools/role on SpecialistBrief)."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agentic_concierge.application.orchestrator import (
    OrchestrationPlan,
    SpecialistBrief,
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
