"""Task graph for recursive task decomposition.

Replaces the flat ``OrchestrationPlan`` as the execution plan.  The graph
is a DAG of ``TaskNode`` objects, where the root is the user's original
request and leaves are executable sub-tasks.

See DESIGN_V2.md §8.1 for rationale.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


# Valid status transitions.  Key = current status, value = set of allowed next statuses.
_VALID_TRANSITIONS: Dict[str, set] = {
    "pending": {"decomposing", "executing", "failed"},
    "decomposing": {"critiqued", "failed"},
    "critiqued": {"decomposing", "executing", "failed"},
    "executing": {"reviewing", "done", "failed"},
    "reviewing": {"executing", "done", "failed"},
    "done": set(),
    "failed": set(),
}

Status = Literal[
    "pending", "decomposing", "critiqued", "executing", "reviewing", "done", "failed"
]


def _make_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class TaskNode:
    """A node in the task decomposition graph."""

    id: str
    description: str
    parent_id: Optional[str] = None

    # Decomposition state
    status: Status = "pending"
    children: List[str] = field(default_factory=list)

    # Execution requirements (set during decomposition)
    required_capabilities: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    finish_schema_key: str = "general"

    # Results
    result: Optional[Dict[str, Any]] = None
    critique: Optional[str] = None

    # Depth in the graph (root = 0)
    depth: int = 0

    @property
    def is_leaf(self) -> bool:
        """True when this node has no children."""
        return len(self.children) == 0

    @property
    def is_terminal(self) -> bool:
        """True when this node is done or failed."""
        return self.status in ("done", "failed")


@dataclass
class TaskGraph:
    """DAG of task nodes.  Root is the user's original request."""

    nodes: Dict[str, TaskNode] = field(default_factory=dict)
    root_id: str = ""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_root(cls, description: str, *, node_id: Optional[str] = None) -> "TaskGraph":
        """Create a graph with a single root node."""
        nid = node_id or _make_id()
        root = TaskNode(id=nid, description=description, depth=0)
        return cls(nodes={nid: root}, root_id=nid)

    def add_child(
        self,
        parent_id: str,
        description: str,
        *,
        node_id: Optional[str] = None,
        required_capabilities: Optional[List[str]] = None,
        required_tools: Optional[List[str]] = None,
        finish_schema_key: str = "general",
    ) -> TaskNode:
        """Add a child node under *parent_id*.  Returns the new node."""
        parent = self.nodes.get(parent_id)
        if parent is None:
            raise KeyError(f"Parent node {parent_id!r} not found in graph")

        nid = node_id or _make_id()
        child = TaskNode(
            id=nid,
            description=description,
            parent_id=parent_id,
            required_capabilities=required_capabilities or [],
            required_tools=required_tools or [],
            finish_schema_key=finish_schema_key,
            depth=parent.depth + 1,
        )
        parent.children.append(nid)
        self.nodes[nid] = child
        return child

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    @property
    def root(self) -> TaskNode:
        """Return the root node."""
        return self.nodes[self.root_id]

    def leaves(self) -> List[TaskNode]:
        """Return leaf nodes (no children) that are not terminal."""
        return [n for n in self.nodes.values() if n.is_leaf and not n.is_terminal]

    def ready_nodes(self) -> List[TaskNode]:
        """Return leaf nodes whose sibling dependencies are satisfied.

        A node is ready when:
        - It is a leaf (no children)
        - Its status is ``"pending"``
        - All sibling nodes that appear *before* it in the parent's children
          list and are marked sequential (i.e. the default) are ``"done"``

        For the root node (no parent), it is ready if pending and a leaf.
        """
        ready = []
        for node in self.nodes.values():
            if node.status != "pending" or not node.is_leaf:
                continue
            if node.parent_id is None:
                # Root node — always ready if pending + leaf
                ready.append(node)
                continue
            parent = self.nodes[node.parent_id]
            # Check all siblings that precede this node
            preceding_done = True
            for sib_id in parent.children:
                if sib_id == node.id:
                    break
                sib = self.nodes[sib_id]
                if not sib.is_terminal:
                    preceding_done = False
                    break
            if preceding_done:
                ready.append(node)
        return ready

    def children_of(self, node_id: str) -> List[TaskNode]:
        """Return direct children of *node_id*."""
        node = self.nodes.get(node_id)
        if node is None:
            raise KeyError(f"Node {node_id!r} not found in graph")
        return [self.nodes[cid] for cid in node.children]

    def ancestors(self, node_id: str) -> List[TaskNode]:
        """Return ancestors from node to root (exclusive of node itself)."""
        result = []
        current = self.nodes.get(node_id)
        if current is None:
            raise KeyError(f"Node {node_id!r} not found in graph")
        while current.parent_id is not None:
            current = self.nodes[current.parent_id]
            result.append(current)
        return result

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def transition(self, node_id: str, new_status: Status) -> None:
        """Transition a node to *new_status*.

        Raises ``ValueError`` if the transition is invalid.
        """
        node = self.nodes.get(node_id)
        if node is None:
            raise KeyError(f"Node {node_id!r} not found in graph")
        allowed = _VALID_TRANSITIONS.get(node.status, set())
        if new_status not in allowed:
            raise ValueError(
                f"Invalid transition: {node.status!r} -> {new_status!r} "
                f"(allowed: {sorted(allowed)})"
            )
        node.status = new_status

    def mark_done(self, node_id: str, result: Optional[Dict[str, Any]] = None) -> None:
        """Mark node as done and propagate completion upward.

        If all children of the parent are done, the parent is also marked done.
        """
        node = self.nodes.get(node_id)
        if node is None:
            raise KeyError(f"Node {node_id!r} not found in graph")
        # Allow transition from executing or reviewing
        if node.status not in ("executing", "reviewing"):
            raise ValueError(
                f"Cannot mark_done from status {node.status!r} "
                f"(must be 'executing' or 'reviewing')"
            )
        node.status = "done"
        node.result = result
        self._propagate_done(node)

    def mark_failed(self, node_id: str, error: str = "") -> None:
        """Mark node as failed.  Does not propagate — parent stays in current state."""
        node = self.nodes.get(node_id)
        if node is None:
            raise KeyError(f"Node {node_id!r} not found in graph")
        if node.is_terminal:
            return  # idempotent
        node.status = "failed"
        node.result = {"error": error} if error else None

    def _propagate_done(self, node: TaskNode) -> None:
        """If all siblings are done, mark parent done too (recursively)."""
        if node.parent_id is None:
            return
        parent = self.nodes[node.parent_id]
        all_done = all(
            self.nodes[cid].status == "done" for cid in parent.children
        )
        if all_done:
            # Merge child results
            merged = {}
            for cid in parent.children:
                child_result = self.nodes[cid].result
                if child_result:
                    merged[cid] = child_result
            parent.status = "done"
            parent.result = merged
            self._propagate_done(parent)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def all_done(self) -> bool:
        """True when the root node is done."""
        return self.root.status == "done"

    def has_failures(self) -> bool:
        """True if any node has failed."""
        return any(n.status == "failed" for n in self.nodes.values())

    def node_count(self) -> int:
        """Total number of nodes in the graph."""
        return len(self.nodes)

    def max_depth(self) -> int:
        """Maximum depth of any node in the graph."""
        if not self.nodes:
            return 0
        return max(n.depth for n in self.nodes.values())

    def summary(self) -> Dict[str, Any]:
        """Return a summary of graph state for logging."""
        by_status: Dict[str, int] = {}
        for n in self.nodes.values():
            by_status[n.status] = by_status.get(n.status, 0) + 1
        return {
            "node_count": self.node_count(),
            "max_depth": self.max_depth(),
            "all_done": self.all_done(),
            "has_failures": self.has_failures(),
            "by_status": by_status,
        }
