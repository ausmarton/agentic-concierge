"""Graph checkpoint serialization — serialize/deserialize TaskGraph state.

Provides pure-data serialization of a ``TaskGraph`` to/from dicts (suitable
for JSON).  Also provides ``prepare_graph_for_resume`` which resets in-flight
nodes so the graph can be safely re-executed.

This module contains NO I/O — file persistence is in
``infrastructure.workspace.graph_checkpoint``.

See ADR-032 in docs/DECISIONS.md for design rationale.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agentic_concierge.application.task_graph import TaskGraph, TaskNode

# Schema version for forward compatibility.  Bump when the serialized shape
# changes in a way that old code cannot read.
SCHEMA_VERSION = 1


def serialize_node(node: TaskNode) -> Dict[str, Any]:
    """Serialize a single TaskNode to a plain dict."""
    return {
        "id": node.id,
        "description": node.description,
        "parent_id": node.parent_id,
        "status": node.status,
        "children": list(node.children),
        "required_capabilities": list(node.required_capabilities),
        "required_tools": list(node.required_tools),
        "finish_schema_key": node.finish_schema_key,
        "is_synthesis": node.is_synthesis,
        "depends_on": list(node.depends_on),
        "result": node.result,
        "critique": node.critique,
        "depth": node.depth,
    }


def deserialize_node(data: Dict[str, Any]) -> TaskNode:
    """Reconstruct a TaskNode from a plain dict.

    Raises ``ValueError`` if required fields are missing.
    """
    missing = [f for f in ("id", "description", "status") if f not in data]
    if missing:
        raise ValueError(f"Missing required fields in node data: {missing}")
    return TaskNode(
        id=data["id"],
        description=data["description"],
        parent_id=data.get("parent_id"),
        status=data["status"],
        children=list(data.get("children", [])),
        required_capabilities=list(data.get("required_capabilities", [])),
        required_tools=list(data.get("required_tools", [])),
        finish_schema_key=data.get("finish_schema_key", "general"),
        is_synthesis=bool(data.get("is_synthesis", False)),
        depends_on=list(data.get("depends_on", [])),
        result=data.get("result"),
        critique=data.get("critique"),
        depth=data.get("depth", 0),
    )


def serialize_graph(graph: TaskGraph) -> Dict[str, Any]:
    """Serialize a TaskGraph to a JSON-compatible dict.

    The output includes:
    - ``schema_version``: for forward compatibility
    - ``root_id``: the root node id
    - ``nodes``: list of serialized node dicts

    Returns:
        Dict suitable for ``json.dumps``.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "root_id": graph.root_id,
        "nodes": [serialize_node(n) for n in graph.nodes.values()],
    }


def deserialize_graph(data: Dict[str, Any]) -> TaskGraph:
    """Reconstruct a TaskGraph from a serialized dict.

    Validates:
    - Schema version is supported
    - All nodes are present
    - ``root_id`` references an existing node
    - Parent/child references are internally consistent

    Raises:
        ValueError: on validation failures.
    """
    version = data.get("schema_version", 0)
    if version < 1 or version > SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema version {version} "
            f"(supported: 1..{SCHEMA_VERSION})"
        )

    root_id = data.get("root_id", "")
    raw_nodes = data.get("nodes")
    if not raw_nodes:
        raise ValueError("No nodes in checkpoint data")

    nodes: Dict[str, TaskNode] = {}
    for raw in raw_nodes:
        node = deserialize_node(raw)
        if node.id in nodes:
            raise ValueError(f"Duplicate node id: {node.id!r}")
        nodes[node.id] = node

    if root_id not in nodes:
        raise ValueError(
            f"root_id {root_id!r} not found in nodes "
            f"(available: {sorted(nodes.keys())})"
        )

    # Validate parent/child consistency
    for node in nodes.values():
        if node.parent_id is not None and node.parent_id not in nodes:
            raise ValueError(
                f"Node {node.id!r} references parent {node.parent_id!r} "
                f"which does not exist"
            )
        for child_id in node.children:
            if child_id not in nodes:
                raise ValueError(
                    f"Node {node.id!r} references child {child_id!r} "
                    f"which does not exist"
                )

    return TaskGraph(nodes=nodes, root_id=root_id)


# Statuses that indicate the node was in-flight when the checkpoint was taken.
_IN_FLIGHT_STATUSES = frozenset({"executing", "reviewing", "decomposing"})


def prepare_graph_for_resume(
    graph: TaskGraph,
    *,
    prior_results: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Prepare a deserialized graph for re-execution.

    1. Resets in-flight nodes (``executing``, ``reviewing``, ``decomposing``)
       back to ``pending`` so they are eligible for re-execution.
    2. Collects results from all ``done`` nodes into a ``prior_results`` dict
       that the graph executor can use to skip already-completed work.

    This is an *administrative* operation — it bypasses the state machine's
    ``transition()`` validation because the graph was persisted mid-flight
    and the normal transition rules don't apply to recovery.

    Args:
        graph: The deserialized graph (modified **in place**).
        prior_results: Optional dict to merge prior results into.
            If provided, done-node results are added to this dict.

    Returns:
        Dict mapping ``node_id → result`` for all ``done`` nodes.
    """
    collected: Dict[str, Dict[str, Any]] = dict(prior_results or {})

    for node in graph.nodes.values():
        if node.status in _IN_FLIGHT_STATUSES:
            # Administrative reset — bypass state machine
            node.status = "pending"
            node.result = None
        elif node.status == "done" and node.result is not None:
            collected[node.id] = node.result

    return collected
