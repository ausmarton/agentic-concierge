"""Task graph executor — runs leaf nodes and propagates completion.

Drives a ``TaskGraph`` to completion by repeatedly finding ready nodes,
executing them (potentially in parallel), and propagating results upward.

The executor is agnostic to *how* leaf nodes are executed — it accepts an
``execute_leaf`` async callback and delegates to it.  This allows the same
executor to work with the existing pack loop, a mock, or future backends.

See DESIGN_V2.md §8.4 for design rationale.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from agentic_concierge.application.task_graph import TaskGraph, TaskNode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# Callback signature for leaf execution.
# Takes (node, graph) → result dict (or raises on failure).
LeafExecutor = Callable[[TaskNode, TaskGraph], Awaitable[Dict[str, Any]]]

# Callback for event notifications.
EventCallback = Callable[[str, Dict[str, Any]], None]


@dataclass
class ExecutionResult:
    """Result of executing a task graph."""

    graph: TaskGraph
    completed: bool  # True if root is done
    results: Dict[str, Dict[str, Any]]  # node_id → result
    failures: Dict[str, str]  # node_id → error message
    steps_executed: int  # total leaf executions attempted


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_EXECUTION_STEPS = 50  # safety limit to prevent infinite loops


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

async def execute_graph(
    graph: TaskGraph,
    execute_leaf: LeafExecutor,
    *,
    max_steps: int = MAX_EXECUTION_STEPS,
    on_event: Optional[EventCallback] = None,
    prior_results: Optional[Dict[str, Dict[str, Any]]] = None,
) -> ExecutionResult:
    """Execute a task graph by running ready leaves until completion.

    1. Find ready nodes (pending leaves whose preceding siblings are done)
    2. Execute them in parallel via ``execute_leaf``
    3. Mark done/failed, propagate completion
    4. Repeat until root is done or no more ready nodes

    Args:
        graph: The task graph to execute.
        execute_leaf: Async callback ``(node, graph) -> result_dict``.
        max_steps: Safety limit on total leaf executions.
        on_event: Optional callback for execution events.
        prior_results: Pre-populated results from a previous (checkpointed)
            execution.  These are included in the returned ``results`` dict
            but the corresponding nodes are NOT re-executed (they should
            already be marked ``done`` in the graph by
            ``prepare_graph_for_resume``).

    Returns:
        ExecutionResult with the final graph state and collected results.
    """
    results: Dict[str, Dict[str, Any]] = dict(prior_results or {})
    failures: Dict[str, str] = {}
    steps = 0

    def emit(kind: str, payload: Dict[str, Any]) -> None:
        if on_event:
            try:
                on_event(kind, payload)
            except Exception:
                pass  # never let event callbacks break execution

    emit("graph_execution_start", {
        "node_count": graph.node_count(),
        "root_id": graph.root_id,
    })

    while not graph.all_done() and steps < max_steps:
        ready = graph.ready_nodes()
        if not ready:
            # No ready nodes but not done — either all remaining nodes
            # are failed or there's a dependency issue
            logger.warning("No ready nodes but graph not done; breaking")
            emit("graph_stalled", {"summary": graph.summary()})
            break

        emit("graph_round_start", {
            "step": steps,
            "ready_count": len(ready),
            "ready_ids": [n.id for n in ready],
        })

        # Execute all ready nodes in parallel
        tasks = []
        for node in ready:
            graph.transition(node.id, "executing")
            tasks.append(_execute_node(node, graph, execute_leaf))

        node_results = await asyncio.gather(*tasks, return_exceptions=True)

        for node, result in zip(ready, node_results):
            steps += 1
            if isinstance(result, Exception):
                error_msg = str(result)
                logger.warning("Node %s failed: %s", node.id, error_msg)
                graph.mark_failed(node.id, error_msg)
                failures[node.id] = error_msg
                emit("node_failed", {
                    "node_id": node.id,
                    "description": node.description,
                    "error": error_msg,
                })
            else:
                graph.mark_done(node.id, result)
                results[node.id] = result
                emit("node_done", {
                    "node_id": node.id,
                    "description": node.description,
                })

    completed = graph.all_done()
    emit("graph_execution_end", {
        "completed": completed,
        "steps": steps,
        "failures": len(failures),
    })

    return ExecutionResult(
        graph=graph,
        completed=completed,
        results=results,
        failures=failures,
        steps_executed=steps,
    )


async def _execute_node(
    node: TaskNode,
    graph: TaskGraph,
    execute_leaf: LeafExecutor,
) -> Dict[str, Any]:
    """Execute a single leaf node via the callback.

    Wraps the call to add logging and ensure exceptions propagate cleanly.
    """
    logger.info("Executing node %s: %s", node.id, node.description[:80])
    result = await execute_leaf(node, graph)
    return result if result is not None else {}
