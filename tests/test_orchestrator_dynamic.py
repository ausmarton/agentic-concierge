"""Tests for orchestrator capability-driven routing (ADR-028) and dynamic packs."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agentic_concierge.application.orchestrator import (
    OrchestrationPlan,
    SpecialistBrief,
    _collapse_redundant_dynamic,
    _dedup_same_id,
    _resolve_pack_from_capabilities,
    _resolve_specialist_from_capabilities,
    orchestrate_task,
)
from agentic_concierge.config import DEFAULT_CONFIG
from agentic_concierge.domain import LLMResponse, ToolCallRequest
from agentic_concierge.infrastructure.ollama import OllamaChatClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capability_plan_response(
    required_capabilities: list[str],
    brief: str = "do the work",
    finish_schema: str | None = None,
    tools: list[str] | None = None,
    role: str | None = None,
) -> LLMResponse:
    """Build a create_plan response using capability-driven routing (no specialist_id)."""
    assignment: dict = {
        "brief": brief,
        "required_capabilities": required_capabilities,
    }
    if finish_schema:
        assignment["finish_schema"] = finish_schema
    if tools:
        assignment["tools"] = tools
    if role:
        assignment["role"] = role
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            call_id="orch0",
            tool_name="create_plan",
            arguments={
                "assignments": [assignment],
                "mode": "sequential",
                "synthesis_required": False,
                "reasoning": "capability routing",
            },
        )],
    )


def _dynamic_plan_response(
    tools: list[str],
    role: str = "A dynamic agent",
    specialist_id: str = "dynamic",
) -> LLMResponse:
    """Build a create_plan response with explicit tools (dynamic pack)."""
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
                        "required_capabilities": ["instruction_following"],
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
async def test_specialist_brief_tools_none_for_capability_routing():
    """Capability-routed assignments have tools=None by default."""
    plan = await _call_orchestrate(
        _capability_plan_response(["code_python"], brief="build it"),
        "build a service",
    )
    assert plan.specialist_assignments[0].tools is None
    assert plan.specialist_assignments[0].role is None
    assert plan.specialist_assignments[0].specialist_id == "engineering"


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
# Capability-driven routing tests (ADR-028)
# ---------------------------------------------------------------------------


def test_resolve_web_comprehension_to_research():
    """web_comprehension capability resolves to research template."""
    assert _resolve_specialist_from_capabilities(["web_comprehension"]) == "research"


def test_resolve_code_python_to_engineering():
    """code_python capability resolves to engineering template."""
    assert _resolve_specialist_from_capabilities(["code_python"]) == "engineering"


def test_resolve_summarisation_to_research():
    """summarisation capability resolves to research template."""
    assert _resolve_specialist_from_capabilities(["summarisation"]) == "research"


def test_resolve_unknown_capability_defaults():
    """Unknown capability falls back to research (degenerate base-tools-only case)."""
    result = _resolve_specialist_from_capabilities(["telekinesis"])
    # Unknown caps → base tools only → degenerate → research fallback
    assert result == "research"


# ---------------------------------------------------------------------------
# _resolve_pack_from_capabilities (full resolution)
# ---------------------------------------------------------------------------


def test_pack_resolution_web_returns_template():
    """web_comprehension tools match research template → (research, None, None, None)."""
    sid, tools, role, fs = _resolve_pack_from_capabilities(["web_comprehension"])
    assert sid == "research"
    assert tools is None
    assert role is None
    assert fs is None


def test_pack_resolution_code_returns_template():
    """code_python tools match engineering template → (engineering, None, None, None)."""
    sid, tools, role, fs = _resolve_pack_from_capabilities(["code_python"])
    assert sid == "engineering"
    assert tools is None
    assert role is None
    assert fs is None


def test_pack_resolution_mixed_returns_dynamic():
    """code_python + web_comprehension → no template match → dynamic pack."""
    sid, tools, role, fs = _resolve_pack_from_capabilities(["code_python", "web_comprehension"])
    assert sid == "dynamic"
    assert tools is not None
    assert "shell" in tools
    assert "web_search" in tools
    assert "write_file" in tools
    assert "fetch_url" in tools
    assert role is not None
    assert "python" in role.lower()
    assert "web" in role.lower()


def test_pack_resolution_degenerate_falls_back_to_research():
    """Only model capabilities (no tool caps) → degenerate → research fallback."""
    sid, tools, role, fs = _resolve_pack_from_capabilities(["reasoning"])
    assert sid == "research"
    assert tools is None


def test_pack_resolution_infers_finish_schema():
    """Dynamic pack infers finish schema from capabilities."""
    _sid, _tools, _role, fs = _resolve_pack_from_capabilities(
        ["code_python", "web_comprehension"]
    )
    # Mixed → None (no clear inference)
    assert fs is None


@pytest.mark.asyncio
async def test_capability_routing_web_comprehension():
    """Integration: web_comprehension → research template via orchestrate_task."""
    plan = await _call_orchestrate(
        _capability_plan_response(["web_comprehension"]),
        "What is the AXP stock price?",
    )
    assert len(plan.specialist_assignments) == 1
    assert plan.specialist_assignments[0].specialist_id == "research"


@pytest.mark.asyncio
async def test_capability_routing_code_python():
    """Integration: code_python → engineering template via orchestrate_task."""
    plan = await _call_orchestrate(
        _capability_plan_response(["code_python"], finish_schema="code"),
        "Build a REST API with Flask",
    )
    assert len(plan.specialist_assignments) == 1
    assert plan.specialist_assignments[0].specialist_id == "engineering"
    assert plan.specialist_assignments[0].finish_schema == "code"


@pytest.mark.asyncio
async def test_capability_routing_with_finish_schema():
    """Capability routing preserves finish_schema from the plan."""
    plan = await _call_orchestrate(
        _capability_plan_response(
            ["web_comprehension"], finish_schema="quick_answer",
        ),
        "How much does Amex Platinum cost?",
    )
    brief = plan.specialist_assignments[0]
    assert brief.specialist_id == "research"
    assert brief.finish_schema == "quick_answer"


@pytest.mark.asyncio
async def test_capability_routing_dynamic_with_tools():
    """When tools are provided without specialist_id, creates a dynamic pack."""
    plan = await _call_orchestrate(
        _capability_plan_response(
            ["web_comprehension", "code_python"],
            tools=["web_search", "fetch_url", "shell", "write_file"],
            role="You are a deployment specialist",
        ),
        "Search the web and write a deployment script",
    )
    assert len(plan.specialist_assignments) == 1
    brief = plan.specialist_assignments[0]
    assert brief.specialist_id == "dynamic"
    assert brief.tools == ["web_search", "fetch_url", "shell", "write_file"]
    assert brief.role == "You are a deployment specialist"


@pytest.mark.asyncio
async def test_capability_routing_mixed_creates_dynamic():
    """Integration: code_python + web_comprehension → dynamic pack (no template match)."""
    plan = await _call_orchestrate(
        _capability_plan_response(["code_python", "web_comprehension"]),
        "Search the web for API docs and implement a client",
    )
    assert len(plan.specialist_assignments) == 1
    brief = plan.specialist_assignments[0]
    assert brief.specialist_id == "dynamic"
    assert brief.tools is not None
    assert "shell" in brief.tools
    assert "web_search" in brief.tools
    assert "write_file" in brief.tools
    assert "fetch_url" in brief.tools
    assert brief.role is not None


@pytest.mark.asyncio
async def test_backward_compat_specialist_id_still_works():
    """Legacy: explicit specialist_id is accepted as an override."""
    resp = LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            call_id="orch0",
            tool_name="create_plan",
            arguments={
                "assignments": [{
                    "specialist_id": "engineering",
                    "brief": "build it",
                    "required_capabilities": ["code_python"],
                }],
                "mode": "sequential",
                "synthesis_required": False,
                "reasoning": "backward compat",
            },
        )],
    )
    plan = await _call_orchestrate(resp, "build a service")
    assert plan.specialist_assignments[0].specialist_id == "engineering"


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
                    {
                        "specialist_id": "research",
                        "brief": "look up AXP stock price",
                        "required_capabilities": ["web_comprehension"],
                    },
                    {
                        "specialist_id": "dynamic",
                        "brief": "format the result",
                        "tools": ["web_search"],
                        "role": "formatter",
                        "required_capabilities": ["summarisation"],
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


def test_dedup_preserves_dynamic_with_different_tools():
    """Two dynamic packs with different tool sets are NOT merged."""
    assignments = [
        SpecialistBrief(
            specialist_id="dynamic", brief="write code",
            tools=["shell", "write_file"], role="Code writer",
        ),
        SpecialistBrief(
            specialist_id="dynamic", brief="search web",
            tools=["web_search", "fetch_url"], role="Researcher",
        ),
    ]
    result = _dedup_same_id(assignments)
    assert len(result) == 2
    assert result[0].tools == ["shell", "write_file"]
    assert result[1].tools == ["web_search", "fetch_url"]


def test_dedup_merges_dynamic_with_same_tools():
    """Two dynamic packs with identical tool sets ARE merged."""
    assignments = [
        SpecialistBrief(
            specialist_id="dynamic", brief="task A",
            tools=["web_search", "fetch_url"], role="Agent",
        ),
        SpecialistBrief(
            specialist_id="dynamic", brief="task B",
            tools=["web_search", "fetch_url"], role="Agent",
        ),
    ]
    result = _dedup_same_id(assignments)
    assert len(result) == 1
    assert "task A" in result[0].brief
    assert "task B" in result[0].brief


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
                    {
                        "specialist_id": "research",
                        "brief": "look up KO stock price",
                        "required_capabilities": ["web_comprehension"],
                    },
                    {
                        "specialist_id": "research",
                        "brief": "look up BAC stock price",
                        "required_capabilities": ["web_comprehension"],
                    },
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


# ---------------------------------------------------------------------------
# Issue #1: Capability validation — _filter_known_capabilities
# ---------------------------------------------------------------------------


def test_filter_known_capabilities_all_valid():
    from agentic_concierge.application.orchestrator import _filter_known_capabilities
    result = _filter_known_capabilities(["web_comprehension", "code_python"])
    assert result == ["web_comprehension", "code_python"]


def test_filter_known_capabilities_all_hallucinated():
    from agentic_concierge.application.orchestrator import _filter_known_capabilities
    result = _filter_known_capabilities(["systematic_review", "citation_extraction"])
    assert result == ["instruction_following"]


def test_filter_known_capabilities_mixed():
    from agentic_concierge.application.orchestrator import _filter_known_capabilities
    result = _filter_known_capabilities(["web_comprehension", "fabricated_cap"])
    assert result == ["web_comprehension"]


def test_filter_known_capabilities_empty_input():
    from agentic_concierge.application.orchestrator import _filter_known_capabilities
    result = _filter_known_capabilities([])
    assert result == ["instruction_following"]


@pytest.mark.asyncio
async def test_hallucinated_capabilities_filtered_in_routing():
    """LLM returns hallucinated caps → filtered to defaults → correct resolution."""
    resp = _capability_plan_response(
        required_capabilities=["systematic_review", "web_search", "file_io"],
        brief="stock price lookup",
    )
    plan = await _call_orchestrate(resp, "What is KO stock price?")
    # Hallucinated caps filtered → defaults to instruction_following → degenerate → research
    assert plan.specialist_assignments[0].specialist_id == "research"
    # Filtered caps should only contain known values
    caps = plan.specialist_assignments[0].required_capabilities or []
    from agentic_concierge.infrastructure.model_profiles import KNOWN_CAPABILITIES
    for c in caps:
        assert c in KNOWN_CAPABILITIES, f"Unknown capability {c!r} leaked through"


@pytest.mark.asyncio
async def test_mixed_valid_and_invalid_caps_preserves_valid():
    """LLM returns mix of valid and invalid caps → valid ones preserved."""
    resp = _capability_plan_response(
        required_capabilities=["web_comprehension", "fabricated_cap"],
        brief="look up info",
    )
    plan = await _call_orchestrate(resp, "Look up info")
    # web_comprehension preserved → research template
    assert plan.specialist_assignments[0].specialist_id == "research"


# ---------------------------------------------------------------------------
# Issue #4: Degenerate fallback uses quick_answer schema
# ---------------------------------------------------------------------------


def test_degenerate_fallback_uses_quick_answer_schema():
    """Model-only caps (reasoning) → degenerate → research with quick_answer schema."""
    sid, tools, role, finish_key = _resolve_pack_from_capabilities(["reasoning"])
    assert sid == "research"
    assert finish_key == "quick_answer"


def test_degenerate_fallback_instruction_following():
    """instruction_following only → degenerate → research with quick_answer."""
    sid, tools, role, finish_key = _resolve_pack_from_capabilities(["instruction_following"])
    assert sid == "research"
    assert finish_key == "quick_answer"
