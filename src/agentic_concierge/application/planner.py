"""Planner and critic agents for recursive task decomposition.

The planner decomposes a task into a ``TaskGraph``.  The critic reviews
the plan for quality.  If the critique fails, the planner re-plans with
feedback (up to ``max_replans`` times).

See DESIGN_V2.md §8.2 and §8.3 for design rationale.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agentic_concierge.application.task_graph import TaskGraph

if TYPE_CHECKING:
    from agentic_concierge.application.ports import ChatClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_REPLANS = 2
MAX_DECOMPOSITION_DEPTH = 3

_KNOWN_FINISH_SCHEMAS = {"quick_answer", "research_report", "code", "general"}


# ---------------------------------------------------------------------------
# Tool definitions for the planner LLM
# ---------------------------------------------------------------------------

_DECOMPOSE_TOOL_DEF: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "decompose_task",
        "description": (
            "Decompose the task into subtasks. Each subtask specifies what "
            "capabilities are needed and what tools to use. For simple tasks, "
            "return a single subtask with no children."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subtasks": {
                    "type": "array",
                    "description": "Ordered list of subtasks.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "Clear description of what this subtask accomplishes.",
                            },
                            "required_capabilities": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Capabilities needed: web_comprehension, summarisation, "
                                    "code_python, code_rust, code_sql, reasoning, "
                                    "structured_output, instruction_following."
                                ),
                            },
                            "required_tools": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific tools needed (optional).",
                            },
                            "finish_schema": {
                                "type": "string",
                                "enum": ["quick_answer", "research_report", "code", "general"],
                                "description": "Output format for this subtask.",
                            },
                        },
                        "required": ["description", "required_capabilities"],
                    },
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of the decomposition strategy.",
                },
            },
            "required": ["subtasks", "reasoning"],
        },
    },
}


# ---------------------------------------------------------------------------
# Tool definition for the critic LLM
# ---------------------------------------------------------------------------

_CRITIQUE_TOOL_DEF: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "critique_plan",
        "description": (
            "Review the task decomposition plan. Approve if subtasks are "
            "achievable, complete, and not over-decomposed. Reject with "
            "specific feedback if the plan needs improvement."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "approved": {
                    "type": "boolean",
                    "description": "True if the plan is good enough to execute.",
                },
                "feedback": {
                    "type": "string",
                    "description": (
                        "If rejected: specific actionable feedback for the planner. "
                        "If approved: brief summary of why the plan is good."
                    ),
                },
            },
            "required": ["approved", "feedback"],
        },
    },
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class PlanResult:
    """Result of the plan_task decomposition loop."""

    graph: TaskGraph
    reasoning: str
    critique_feedback: Optional[str]  # last critique feedback, None if approved first try
    replan_count: int  # number of re-plans performed (0 = approved first try)
    planner_model: str
    critic_model: str


# ---------------------------------------------------------------------------
# Planner system prompt
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = (
    "You are a task decomposition planner. Given a task, break it down into "
    "subtasks that can each be executed by a single specialist agent.\n\n"
    "Guidelines:\n"
    "1. SIMPLE TASKS (factual questions, single lookups, quick calculations) "
    "should be a SINGLE subtask. Do not over-decompose.\n"
    "2. COMPLEX TASKS (multi-step engineering, research with multiple sources, "
    "build-and-test workflows) should be decomposed into sequential subtasks.\n"
    "3. Each subtask must specify the capabilities it needs.\n"
    "4. Subtasks are executed in order — later subtasks can depend on earlier ones.\n"
    "5. Use finish_schema='quick_answer' for simple factual answers, 'code' for "
    "engineering output, 'research_report' for deep research, 'general' otherwise.\n\n"
    "Call decompose_task with your plan."
)

_PLANNER_REPLAN_TEMPLATE = (
    "Your previous plan was rejected by the critic.\n\n"
    "Critic feedback: {feedback}\n\n"
    "Please revise your plan addressing the feedback. Call decompose_task again."
)


# ---------------------------------------------------------------------------
# Critic system prompt
# ---------------------------------------------------------------------------

_CRITIC_SYSTEM = (
    "You are a plan critic. Review the task decomposition plan and decide "
    "whether it is good enough to execute.\n\n"
    "Evaluate:\n"
    "1. Are subtasks achievable by a single agent with the specified capabilities?\n"
    "2. Are there missing steps that would cause execution to fail?\n"
    "3. Is the plan over-decomposed? (Simple tasks split into too many subtasks)\n"
    "4. Are the specified capabilities realistic for each subtask?\n"
    "5. Is the finish_schema appropriate for each subtask?\n\n"
    "Call critique_plan with your decision."
)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _build_planner_messages(
    prompt: str,
    *,
    critique_feedback: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build messages for the planner LLM call."""
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _PLANNER_SYSTEM},
        {"role": "user", "content": f"Task: {prompt}"},
    ]
    if critique_feedback:
        messages.append({
            "role": "user",
            "content": _PLANNER_REPLAN_TEMPLATE.format(feedback=critique_feedback),
        })
    return messages


def _build_critic_messages(prompt: str, subtasks_json: str) -> List[Dict[str, Any]]:
    """Build messages for the critic LLM call."""
    return [
        {"role": "system", "content": _CRITIC_SYSTEM},
        {"role": "user", "content": (
            f"Original task: {prompt}\n\n"
            f"Proposed plan:\n{subtasks_json}"
        )},
    ]


def _parse_decomposition(
    arguments: Dict[str, Any],
    root_description: str,
) -> TaskGraph:
    """Parse a decompose_task tool call into a TaskGraph."""
    subtasks = arguments.get("subtasks", [])
    if not subtasks:
        raise ValueError("decompose_task returned no subtasks")

    graph = TaskGraph.from_root(root_description)

    for st in subtasks:
        desc = st.get("description", "")
        if not desc:
            continue
        caps = st.get("required_capabilities", [])
        tools = st.get("required_tools", [])
        fs = st.get("finish_schema", "general")
        if fs not in _KNOWN_FINISH_SCHEMAS:
            fs = "general"
        graph.add_child(
            graph.root_id,
            desc,
            required_capabilities=caps if isinstance(caps, list) else [],
            required_tools=tools if isinstance(tools, list) else [],
            finish_schema_key=fs,
        )

    return graph


async def _call_planner(
    chat_client: "ChatClient",
    model: str,
    prompt: str,
    *,
    critique_feedback: Optional[str] = None,
) -> tuple[TaskGraph, str]:
    """Call the planner LLM and parse the result into a TaskGraph.

    Returns (graph, reasoning).
    Raises RuntimeError on failure.
    """
    messages = _build_planner_messages(prompt, critique_feedback=critique_feedback)
    response = await chat_client.chat(
        messages, model, tools=[_DECOMPOSE_TOOL_DEF], temperature=0.0, max_tokens=1024,
    )

    if not response.tool_calls:
        raise RuntimeError("Planner returned no tool call")

    tc = response.tool_calls[0]
    if tc.tool_name != "decompose_task":
        raise RuntimeError(f"Planner called unexpected tool {tc.tool_name!r}")

    args = tc.arguments
    reasoning = args.get("reasoning", "")
    graph = _parse_decomposition(args, prompt)
    return graph, reasoning


async def _call_critic(
    chat_client: "ChatClient",
    model: str,
    prompt: str,
    graph: TaskGraph,
) -> tuple[bool, str]:
    """Call the critic LLM to review the plan.

    Returns (approved, feedback).
    """
    # Build a JSON summary of subtasks for the critic
    subtasks_summary = []
    for child in graph.children_of(graph.root_id):
        subtasks_summary.append({
            "description": child.description,
            "required_capabilities": child.required_capabilities,
            "required_tools": child.required_tools,
            "finish_schema": child.finish_schema_key,
        })
    subtasks_json = json.dumps(subtasks_summary, indent=2)

    messages = _build_critic_messages(prompt, subtasks_json)
    response = await chat_client.chat(
        messages, model, tools=[_CRITIQUE_TOOL_DEF], temperature=0.0, max_tokens=512,
    )

    if not response.tool_calls:
        # Fail-open: no tool call → assume approved
        logger.warning("Critic returned no tool call; assuming approved")
        return True, "No critique provided"

    tc = response.tool_calls[0]
    if tc.tool_name != "critique_plan":
        logger.warning("Critic called unexpected tool %r; assuming approved", tc.tool_name)
        return True, "Unexpected tool call"

    approved = bool(tc.arguments.get("approved", True))
    feedback = tc.arguments.get("feedback", "")
    return approved, feedback


def should_decompose(
    node_depth: int,
    subtask_count: int,
    *,
    max_depth: int = MAX_DECOMPOSITION_DEPTH,
) -> bool:
    """Decide whether further decomposition is appropriate.

    Simple heuristic based on depth and existing subtask count.
    More sophisticated checks (model capability matching) can be
    added when the model runtime integration is wired.

    Args:
        node_depth: Current depth of the node being considered.
        subtask_count: Number of subtasks in the current plan.
        max_depth: Maximum allowed decomposition depth.
    """
    if node_depth >= max_depth:
        return False
    # Single subtask at root level = simple task, don't decompose further
    if node_depth == 0 and subtask_count <= 1:
        return False
    return True


async def plan_task(
    prompt: str,
    chat_client: "ChatClient",
    planner_model: str,
    critic_model: str,
    *,
    max_replans: int = MAX_REPLANS,
    max_depth: int = MAX_DECOMPOSITION_DEPTH,
) -> PlanResult:
    """Decompose a task into a TaskGraph using planner + critic loop.

    1. Planner decomposes the task
    2. Critic reviews the plan
    3. If rejected, planner re-plans with feedback (up to max_replans times)
    4. Returns the final approved (or last attempted) plan

    Args:
        prompt: User task description.
        chat_client: LLM chat interface (planner and critic may use different models).
        planner_model: Model name for the planner.
        critic_model: Model name for the critic.
        max_replans: Maximum re-plan attempts after critique rejection.
        max_depth: Maximum task decomposition depth.

    Returns:
        PlanResult with the TaskGraph and metadata.
    """
    critique_feedback: Optional[str] = None
    last_rejection: Optional[str] = None  # tracks the rejection that caused re-planning
    replan_count = 0
    graph: Optional[TaskGraph] = None
    reasoning = ""

    for attempt in range(1 + max_replans):
        try:
            graph, reasoning = await _call_planner(
                chat_client, planner_model, prompt,
                critique_feedback=critique_feedback,
            )
        except RuntimeError as exc:
            logger.warning("Planner failed (attempt %d): %s", attempt + 1, exc)
            if graph is None:
                # First attempt failed — create a minimal fallback graph
                graph = TaskGraph.from_root(prompt)
                graph.add_child(
                    graph.root_id, prompt,
                    required_capabilities=["instruction_following"],
                    finish_schema_key="general",
                )
                reasoning = f"planner_fallback: {exc}"
            break

        # Check depth control — if decomposition is too shallow to critique, skip
        subtask_count = len(graph.children_of(graph.root_id))
        if not should_decompose(0, subtask_count, max_depth=max_depth):
            logger.info("Simple task (single subtask); skipping critic")
            break

        # Call critic
        try:
            approved, feedback = await _call_critic(
                chat_client, critic_model, prompt, graph,
            )
        except Exception as exc:
            # Fail-open: critic error → accept the plan
            logger.warning("Critic failed (%s); accepting plan as-is", exc)
            break

        if approved:
            logger.info("Critic approved plan (attempt %d)", attempt + 1)
            critique_feedback = feedback
            break

        # Rejected — will we re-plan?
        logger.info("Critic rejected plan (attempt %d): %s", attempt + 1, feedback)
        critique_feedback = feedback
        last_rejection = feedback

        if attempt >= max_replans:
            # No more re-plans allowed — use the plan anyway
            logger.warning("Max re-plans reached; using last plan despite rejection")
            break

        # Will re-plan on next iteration
        replan_count += 1

    assert graph is not None  # At least the fallback creates one

    # Mark root as decomposing → critiqued
    if len(graph.children_of(graph.root_id)) > 0:
        graph.transition(graph.root_id, "decomposing")
        graph.transition(graph.root_id, "critiqued")
        if critique_feedback:
            graph.root.critique = critique_feedback

    return PlanResult(
        graph=graph,
        reasoning=reasoning,
        critique_feedback=last_rejection,
        replan_count=replan_count,
        planner_model=planner_model,
        critic_model=critic_model,
    )
