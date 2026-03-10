"""Affinity-aware leaf executor — wires agent roles into graph execution.

Wraps the graph executor's ``LeafExecutor`` callback with model assignment:
each leaf node gets the best model for its inferred agent role, and the
model handle is released after execution.

Also issues ``preload_hint`` calls for upcoming nodes so models are loaded
in the background while the current step executes.

See DESIGN_V2.md §9.2 and §7.5 for design rationale.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from agentic_concierge.application.agent_roles import (
    ModelAssignment,
    assign_model,
    determine_role,
    get_role,
)
from agentic_concierge.application.graph_executor import (
    EventCallback,
    ExecutionResult,
    execute_graph,
)
from agentic_concierge.application.task_graph import TaskGraph, TaskNode

logger = logging.getLogger(__name__)


# Type for the actual work callback.
# Takes (node, graph, model_assignment) → result dict.
WorkExecutor = Callable[
    [TaskNode, TaskGraph, ModelAssignment],
    Awaitable[Dict[str, Any]],
]


@dataclass
class AffinityContext:
    """Tracks model assignments across the task graph execution.

    Used to enforce must_differ_from constraints and release handles.
    """

    # role → model_id used (for must_differ_from)
    role_model_map: Dict[str, str] = field(default_factory=dict)

    # node_id → ModelAssignment (for handle release)
    node_assignments: Dict[str, ModelAssignment] = field(default_factory=dict)


async def execute_graph_with_affinity(
    graph: TaskGraph,
    runtime: Any,  # ModelRuntime protocol
    work_executor: WorkExecutor,
    *,
    max_steps: int = 50,
    on_event: Optional[EventCallback] = None,
    prior_results: Optional[Dict[str, Dict[str, Any]]] = None,
) -> ExecutionResult:
    """Execute a task graph with automatic model assignment per node.

    For each leaf node:
    1. Determine the agent role from task capabilities
    2. Assign the best model using ``assign_model``
    3. Execute the work via ``work_executor``
    4. Release the model handle

    Args:
        graph: Task graph to execute.
        runtime: ModelRuntime for model acquisition.
        work_executor: Async callback ``(node, graph, assignment) -> result``.
        max_steps: Safety limit.
        on_event: Optional event callback.

    Returns:
        ExecutionResult with the completed graph.
    """
    ctx = AffinityContext()

    async def _leaf_executor(node: TaskNode, graph: TaskGraph) -> Dict[str, Any]:
        # 1. Determine role
        role_name = determine_role(node.required_capabilities)

        # 2. Detect concurrent execution: count nodes currently executing
        #    (including self — already transitioned to "executing" by graph_executor)
        concurrent_count = sum(
            1 for n in graph.nodes.values() if n.status == "executing"
        )
        prefer_concurrent = concurrent_count > 1

        # 3. Assign model with constraints
        assignment = await assign_model(
            role_name,
            runtime,
            task_capabilities=node.required_capabilities,
            node_model_map=ctx.role_model_map,
            prefer_concurrent=prefer_concurrent,
        )

        ctx.node_assignments[node.id] = assignment
        ctx.role_model_map[role_name] = assignment.model_id

        logger.info(
            "Node %s (%s): role=%s model=%s concurrent=%s",
            node.id, node.description[:40], role_name, assignment.model_id,
            prefer_concurrent,
        )

        if on_event:
            try:
                on_event("model_assigned", {
                    "node_id": node.id,
                    "role": role_name,
                    "model_id": assignment.model_id,
                    "prefer_concurrent": prefer_concurrent,
                })
            except Exception:
                pass

        # 4. Preload hint for upcoming siblings
        _issue_preload_hints(node, graph, runtime)

        # 5. Execute
        try:
            return await work_executor(node, graph, assignment)
        finally:
            # 6. Release handle
            try:
                await runtime.release(assignment.handle)
            except Exception as exc:
                logger.warning("Failed to release handle for node %s: %s", node.id, exc)

    return await execute_graph(
        graph,
        _leaf_executor,
        max_steps=max_steps,
        on_event=on_event,
        prior_results=prior_results,
    )


def _issue_preload_hints(
    current_node: TaskNode,
    graph: TaskGraph,
    runtime: Any,
) -> None:
    """Fire-and-forget preload hints for nodes likely to execute next.

    Looks at pending siblings that follow the current node.
    Non-blocking — failures are silently ignored.
    """
    if current_node.parent_id is None:
        return
    parent = graph.nodes.get(current_node.parent_id)
    if parent is None:
        return

    # Find siblings after the current node
    found_current = False
    for sib_id in parent.children:
        if sib_id == current_node.id:
            found_current = True
            continue
        if not found_current:
            continue
        sib = graph.nodes.get(sib_id)
        if sib is None or sib.status != "pending":
            continue
        # Issue preload hint
        role_name = determine_role(sib.required_capabilities)
        try:
            role = get_role(role_name)
            requirements = dict(role.requirements)
            for cap in sib.required_capabilities:
                if cap not in requirements:
                    requirements[cap] = 0.6
            # Fire and forget — don't await
            asyncio.ensure_future(_safe_preload(runtime, requirements, role.require_tool_calling))
        except Exception:
            pass  # Never let preload hints break execution
        break  # Only hint for the immediate next sibling


async def _safe_preload(
    runtime: Any,
    requirements: Dict[str, float],
    require_tool_calling: bool,
) -> None:
    """Safely call preload_hint, catching all exceptions."""
    try:
        await runtime.preload_hint(
            requirements, require_tool_calling=require_tool_calling,
        )
    except Exception:
        pass  # Preload is best-effort
