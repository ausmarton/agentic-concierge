"""Tests for planner and critic agents — task decomposition loop.

Verifies:
- Planner decomposes tasks into TaskGraph via decompose_task tool call
- Critic approves or rejects plans via critique_plan tool call
- Re-planning loop with feedback
- Max replans respected
- Fallback on planner failure
- Adaptive depth control (simple tasks skip critic)
- Fail-open on critic failure
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from agentic_concierge.application.planner import (
    MAX_DECOMPOSITION_DEPTH,
    MAX_REPLANS,
    PlanResult,
    _call_critic,
    _call_planner,
    _parse_decomposition,
    plan_task,
    should_decompose,
)
from agentic_concierge.application.task_graph import TaskGraph
from agentic_concierge.domain import LLMResponse, ToolCallRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tc(name: str, arguments: Dict[str, Any]) -> ToolCallRequest:
    return ToolCallRequest(call_id="1", tool_name=name, arguments=arguments)


def _resp(
    content: str = "",
    tool_calls: Optional[List[ToolCallRequest]] = None,
) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=tool_calls or [])


def _planner_response(
    subtasks: List[Dict[str, Any]],
    reasoning: str = "test reasoning",
) -> LLMResponse:
    """Build a valid planner decompose_task response."""
    return _resp(tool_calls=[_tc("decompose_task", {
        "subtasks": subtasks,
        "reasoning": reasoning,
    })])


def _critic_response(approved: bool, feedback: str = "") -> LLMResponse:
    """Build a valid critic critique_plan response."""
    return _resp(tool_calls=[_tc("critique_plan", {
        "approved": approved,
        "feedback": feedback,
    })])


def _simple_subtask(
    desc: str = "Do the thing",
    caps: Optional[List[str]] = None,
    fs: str = "general",
) -> Dict[str, Any]:
    return {
        "description": desc,
        "required_capabilities": caps or ["instruction_following"],
        "finish_schema": fs,
    }


class SequentialClient:
    """Fake ChatClient that returns responses in sequence."""

    def __init__(self, responses: List[LLMResponse]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    async def chat(self, messages, model, **kwargs) -> LLMResponse:
        if self._call_count >= len(self._responses):
            raise RuntimeError("No more responses")
        resp = self._responses[self._call_count]
        self._call_count += 1
        return resp


# ---------------------------------------------------------------------------
# _parse_decomposition
# ---------------------------------------------------------------------------


class TestParseDecomposition:

    def test_single_subtask(self):
        args = {"subtasks": [_simple_subtask("Find the answer")]}
        graph = _parse_decomposition(args, "What is 2+2?")
        assert graph.node_count() == 2  # root + 1 child
        children = graph.children_of(graph.root_id)
        assert len(children) == 1
        assert children[0].description == "Find the answer"

    def test_multiple_subtasks(self):
        args = {"subtasks": [
            _simple_subtask("Step 1"),
            _simple_subtask("Step 2"),
            _simple_subtask("Step 3"),
        ]}
        graph = _parse_decomposition(args, "Complex task")
        assert graph.node_count() == 4
        assert len(graph.children_of(graph.root_id)) == 3

    def test_capabilities_and_tools(self):
        args = {"subtasks": [{
            "description": "Write SQL",
            "required_capabilities": ["code_sql"],
            "required_tools": ["run_sql"],
            "finish_schema": "code",
        }]}
        graph = _parse_decomposition(args, "SQL task")
        child = graph.children_of(graph.root_id)[0]
        assert child.required_capabilities == ["code_sql"]
        assert child.required_tools == ["run_sql"]
        assert child.finish_schema_key == "code"

    def test_empty_subtasks_raises(self):
        with pytest.raises(ValueError, match="no subtasks"):
            _parse_decomposition({"subtasks": []}, "Task")

    def test_missing_subtasks_raises(self):
        with pytest.raises(ValueError, match="no subtasks"):
            _parse_decomposition({}, "Task")

    def test_unknown_finish_schema_defaults_to_general(self):
        args = {"subtasks": [{
            "description": "Task",
            "required_capabilities": ["reasoning"],
            "finish_schema": "unknown_schema",
        }]}
        graph = _parse_decomposition(args, "Task")
        child = graph.children_of(graph.root_id)[0]
        assert child.finish_schema_key == "general"

    def test_skips_empty_description(self):
        args = {"subtasks": [
            {"description": "", "required_capabilities": ["reasoning"]},
            _simple_subtask("Valid task"),
        ]}
        graph = _parse_decomposition(args, "Root")
        assert graph.node_count() == 2  # root + 1 valid child


# ---------------------------------------------------------------------------
# _call_planner
# ---------------------------------------------------------------------------


class TestCallPlanner:

    @pytest.mark.asyncio
    async def test_returns_graph_and_reasoning(self):
        client = SequentialClient([
            _planner_response([_simple_subtask("Do it")], "because reasons"),
        ])
        graph, reasoning = await _call_planner(client, "test-model", "My task")
        assert graph.node_count() == 2
        assert reasoning == "because reasons"

    @pytest.mark.asyncio
    async def test_no_tool_call_raises(self):
        client = SequentialClient([_resp(content="Here is my plan...")])
        with pytest.raises(RuntimeError, match="no tool call"):
            await _call_planner(client, "test-model", "Task")

    @pytest.mark.asyncio
    async def test_wrong_tool_raises(self):
        client = SequentialClient([
            _resp(tool_calls=[_tc("wrong_tool", {})]),
        ])
        with pytest.raises(RuntimeError, match="unexpected tool"):
            await _call_planner(client, "test-model", "Task")

    @pytest.mark.asyncio
    async def test_critique_feedback_included_in_messages(self):
        """Verify that critique feedback is passed to the planner."""
        call_log = []

        async def recording_chat(messages, model, **kwargs):
            call_log.append(messages)
            return _planner_response([_simple_subtask()])

        client = AsyncMock()
        client.chat = recording_chat

        await _call_planner(client, "m", "Task", critique_feedback="Fix step 2")
        assert any("Fix step 2" in str(m) for m in call_log[0])


# ---------------------------------------------------------------------------
# _call_critic
# ---------------------------------------------------------------------------


class TestCallCritic:

    @pytest.mark.asyncio
    async def test_approved(self):
        client = SequentialClient([_critic_response(True, "Looks good")])
        graph = TaskGraph.from_root("Root", node_id="r")
        graph.add_child("r", "Step 1")
        approved, feedback = await _call_critic(client, "critic-model", "Task", graph)
        assert approved is True
        assert feedback == "Looks good"

    @pytest.mark.asyncio
    async def test_rejected(self):
        client = SequentialClient([_critic_response(False, "Missing step 2")])
        graph = TaskGraph.from_root("Root", node_id="r")
        graph.add_child("r", "Step 1")
        approved, feedback = await _call_critic(client, "critic-model", "Task", graph)
        assert approved is False
        assert feedback == "Missing step 2"

    @pytest.mark.asyncio
    async def test_no_tool_call_assumes_approved(self):
        """Fail-open: no tool call → assume approved."""
        client = SequentialClient([_resp(content="Plan looks fine")])
        graph = TaskGraph.from_root("Root", node_id="r")
        graph.add_child("r", "Step 1")
        approved, _ = await _call_critic(client, "m", "Task", graph)
        assert approved is True

    @pytest.mark.asyncio
    async def test_wrong_tool_assumes_approved(self):
        client = SequentialClient([
            _resp(tool_calls=[_tc("other_tool", {})]),
        ])
        graph = TaskGraph.from_root("Root", node_id="r")
        graph.add_child("r", "Step 1")
        approved, _ = await _call_critic(client, "m", "Task", graph)
        assert approved is True


# ---------------------------------------------------------------------------
# should_decompose
# ---------------------------------------------------------------------------


class TestShouldDecompose:

    def test_max_depth_prevents_decomposition(self):
        assert should_decompose(3, 2, max_depth=3) is False

    def test_beyond_max_depth(self):
        assert should_decompose(5, 2, max_depth=3) is False

    def test_single_subtask_at_root_no_decompose(self):
        assert should_decompose(0, 1) is False

    def test_multiple_subtasks_at_root_allows_decompose(self):
        assert should_decompose(0, 3) is True

    def test_deeper_node_allows_decompose(self):
        assert should_decompose(1, 2) is True

    def test_zero_subtasks_no_decompose(self):
        assert should_decompose(0, 0) is False

    def test_custom_max_depth(self):
        assert should_decompose(1, 2, max_depth=2) is True
        assert should_decompose(2, 2, max_depth=2) is False


# ---------------------------------------------------------------------------
# plan_task (integration)
# ---------------------------------------------------------------------------


class TestPlanTask:

    @pytest.mark.asyncio
    async def test_simple_task_approved_first_try(self):
        """Simple task: planner returns 1 subtask, critic skipped."""
        client = SequentialClient([
            _planner_response([_simple_subtask("Answer the question", fs="quick_answer")]),
            # No critic call expected (single subtask → skip critic)
        ])
        result = await plan_task("What is 2+2?", client, "planner-m", "critic-m")

        assert result.graph.node_count() == 2
        assert result.replan_count == 0
        assert result.planner_model == "planner-m"
        assert result.critic_model == "critic-m"

    @pytest.mark.asyncio
    async def test_complex_task_approved(self):
        """Complex task: planner returns multiple subtasks, critic approves."""
        client = SequentialClient([
            _planner_response([
                _simple_subtask("Plan architecture"),
                _simple_subtask("Write code", ["code_python"], "code"),
                _simple_subtask("Run tests", ["code_python"], "code"),
            ]),
            _critic_response(True, "Good plan"),
        ])
        result = await plan_task("Build a REST API", client, "p", "c")

        assert result.graph.node_count() == 4  # root + 3 children
        assert result.replan_count == 0
        assert result.critique_feedback is None

    @pytest.mark.asyncio
    async def test_replan_after_critique_rejection(self):
        """Critic rejects → planner re-plans → critic approves."""
        client = SequentialClient([
            # Attempt 1: planner
            _planner_response([
                _simple_subtask("Step 1"),
                _simple_subtask("Step 2"),
            ]),
            # Attempt 1: critic rejects
            _critic_response(False, "Missing error handling step"),
            # Attempt 2: planner re-plans
            _planner_response([
                _simple_subtask("Step 1"),
                _simple_subtask("Step 2"),
                _simple_subtask("Handle errors"),
            ]),
            # Attempt 2: critic approves
            _critic_response(True, "Better"),
        ])
        result = await plan_task("Build a thing", client, "p", "c")

        assert result.graph.node_count() == 4  # root + 3 (revised plan)
        assert result.replan_count == 1
        assert result.critique_feedback == "Missing error handling step"

    @pytest.mark.asyncio
    async def test_max_replans_reached(self):
        """Critic rejects all attempts → uses last plan."""
        client = SequentialClient([
            # Attempt 1
            _planner_response([_simple_subtask("A"), _simple_subtask("B")]),
            _critic_response(False, "Bad 1"),
            # Attempt 2
            _planner_response([_simple_subtask("A2"), _simple_subtask("B2")]),
            _critic_response(False, "Bad 2"),
            # Attempt 3 (last)
            _planner_response([_simple_subtask("A3"), _simple_subtask("B3")]),
            _critic_response(False, "Still bad"),
        ])
        result = await plan_task("Task", client, "p", "c", max_replans=2)

        assert result.replan_count == 2
        # Should use the last plan despite rejection
        children = result.graph.children_of(result.graph.root_id)
        assert children[0].description == "A3"

    @pytest.mark.asyncio
    async def test_planner_failure_fallback(self):
        """Planner returns no tool call → fallback graph created."""
        client = SequentialClient([
            _resp(content="I can't decompose this"),
        ])
        result = await plan_task("Task", client, "p", "c")

        assert result.graph.node_count() == 2  # root + fallback child
        assert "planner_fallback" in result.reasoning

    @pytest.mark.asyncio
    async def test_critic_failure_accepts_plan(self):
        """Critic raises exception → fail-open, plan accepted."""
        call_count = 0

        async def failing_critic_chat(messages, model, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Planner response
                return _planner_response([
                    _simple_subtask("A"),
                    _simple_subtask("B"),
                ])
            # Critic call raises
            raise RuntimeError("Critic model unavailable")

        client = AsyncMock()
        client.chat = failing_critic_chat

        result = await plan_task("Task", client, "p", "c")
        assert result.graph.node_count() == 3
        assert result.replan_count == 0

    @pytest.mark.asyncio
    async def test_graph_transitions_applied(self):
        """After planning, root is in 'critiqued' state."""
        client = SequentialClient([
            _planner_response([
                _simple_subtask("Step 1"),
                _simple_subtask("Step 2"),
            ]),
            _critic_response(True, "OK"),
        ])
        result = await plan_task("Task", client, "p", "c")

        assert result.graph.root.status == "critiqued"

    @pytest.mark.asyncio
    async def test_zero_replans_allowed(self):
        """max_replans=0: critic rejection → use plan anyway."""
        client = SequentialClient([
            _planner_response([_simple_subtask("A"), _simple_subtask("B")]),
            _critic_response(False, "Rejected"),
        ])
        result = await plan_task("Task", client, "p", "c", max_replans=0)

        assert result.replan_count == 0
        assert result.graph.node_count() == 3

    @pytest.mark.asyncio
    async def test_single_subtask_skips_critic(self):
        """Single subtask → should_decompose returns False → critic not called."""
        responses_used = []

        async def tracking_chat(messages, model, **kwargs):
            resp = _planner_response([_simple_subtask("Just do it")])
            responses_used.append(model)
            return resp

        client = AsyncMock()
        client.chat = tracking_chat

        result = await plan_task("Simple Q", client, "planner", "critic")
        # Only planner should be called, not critic
        assert len(responses_used) == 1
        assert responses_used[0] == "planner"
        assert result.replan_count == 0
