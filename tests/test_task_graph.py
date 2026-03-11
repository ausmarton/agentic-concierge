"""Tests for TaskGraph and TaskNode — recursive task decomposition.

Verifies:
- Graph construction (root, add_child, nesting)
- Traversal (leaves, ready_nodes, children_of, ancestors)
- State transitions (valid, invalid, terminal)
- Completion propagation (mark_done, mark_failed)
- Inspection (all_done, has_failures, summary)
"""

from __future__ import annotations

import pytest

from agentic_concierge.application.task_graph import TaskGraph, TaskNode


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:

    def test_from_root_creates_single_node(self):
        g = TaskGraph.from_root("Build a web app", node_id="root")
        assert g.node_count() == 1
        assert g.root_id == "root"
        assert g.root.description == "Build a web app"
        assert g.root.status == "pending"
        assert g.root.depth == 0

    def test_from_root_generates_id(self):
        g = TaskGraph.from_root("Task")
        assert len(g.root_id) == 12
        assert g.root_id == g.root.id

    def test_add_child(self):
        g = TaskGraph.from_root("Root", node_id="r")
        child = g.add_child("r", "Sub-task A", node_id="a")
        assert child.id == "a"
        assert child.parent_id == "r"
        assert child.depth == 1
        assert "a" in g.root.children
        assert g.node_count() == 2

    def test_add_child_with_capabilities(self):
        g = TaskGraph.from_root("Root", node_id="r")
        child = g.add_child(
            "r", "Write SQL",
            node_id="sql",
            required_capabilities=["code_sql"],
            required_tools=["run_sql"],
            finish_schema_key="code",
        )
        assert child.required_capabilities == ["code_sql"]
        assert child.required_tools == ["run_sql"]
        assert child.finish_schema_key == "code"

    def test_nested_children(self):
        g = TaskGraph.from_root("Root", node_id="r")
        g.add_child("r", "A", node_id="a")
        g.add_child("a", "A1", node_id="a1")
        g.add_child("a", "A2", node_id="a2")
        assert g.node_count() == 4
        assert g.max_depth() == 2
        assert g.nodes["a1"].depth == 2
        assert g.nodes["a"].children == ["a1", "a2"]

    def test_add_child_unknown_parent_raises(self):
        g = TaskGraph.from_root("Root", node_id="r")
        with pytest.raises(KeyError, match="not-here"):
            g.add_child("not-here", "Orphan")

    def test_multiple_children_at_root(self):
        g = TaskGraph.from_root("Root", node_id="r")
        g.add_child("r", "A", node_id="a")
        g.add_child("r", "B", node_id="b")
        g.add_child("r", "C", node_id="c")
        assert g.root.children == ["a", "b", "c"]
        assert g.node_count() == 4


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------


class TestTraversal:

    def _sample_graph(self) -> TaskGraph:
        """Build a small graph:
           root
           ├── a
           │   ├── a1
           │   └── a2
           └── b
        """
        g = TaskGraph.from_root("Root", node_id="root")
        g.add_child("root", "A", node_id="a")
        g.add_child("root", "B", node_id="b")
        g.add_child("a", "A1", node_id="a1")
        g.add_child("a", "A2", node_id="a2")
        return g

    def test_leaves_returns_leaf_nodes(self):
        g = self._sample_graph()
        leaf_ids = {n.id for n in g.leaves()}
        assert leaf_ids == {"a1", "a2", "b"}

    def test_leaves_excludes_terminal(self):
        g = self._sample_graph()
        g.nodes["a1"].status = "done"
        leaf_ids = {n.id for n in g.leaves()}
        assert "a1" not in leaf_ids
        assert "a2" in leaf_ids

    def test_root_is_leaf_when_no_children(self):
        g = TaskGraph.from_root("Solo", node_id="r")
        assert g.leaves() == [g.root]

    def test_ready_nodes_parallel_siblings(self):
        """Siblings without depends_on are all ready in parallel."""
        g = self._sample_graph()
        ready_ids = {n.id for n in g.ready_nodes()}
        # All leaf siblings are ready in parallel (no depends_on)
        assert "a1" in ready_ids
        assert "a2" in ready_ids
        assert "b" in ready_ids

    def test_ready_nodes_after_sibling_done(self):
        g = self._sample_graph()
        g.transition("a1", "executing")
        g.mark_done("a1", {"answer": "ok"})
        ready_ids = {n.id for n in g.ready_nodes()}
        # a2 still ready (parallel by default), a1 is done so not in ready
        assert "a2" in ready_ids
        assert "a1" not in ready_ids

    def test_ready_nodes_root_pending_leaf(self):
        g = TaskGraph.from_root("Solo", node_id="r")
        ready = g.ready_nodes()
        assert len(ready) == 1
        assert ready[0].id == "r"

    def test_children_of(self):
        g = self._sample_graph()
        kids = g.children_of("a")
        assert [k.id for k in kids] == ["a1", "a2"]

    def test_children_of_leaf(self):
        g = self._sample_graph()
        assert g.children_of("b") == []

    def test_children_of_unknown_raises(self):
        g = self._sample_graph()
        with pytest.raises(KeyError):
            g.children_of("missing")

    def test_ancestors(self):
        g = self._sample_graph()
        anc = g.ancestors("a1")
        assert [n.id for n in anc] == ["a", "root"]

    def test_ancestors_of_root(self):
        g = self._sample_graph()
        assert g.ancestors("root") == []

    def test_ancestors_unknown_raises(self):
        g = self._sample_graph()
        with pytest.raises(KeyError):
            g.ancestors("nope")

    def test_ready_nodes_parallel_by_default(self):
        """Three siblings with no depends_on should all be ready immediately."""
        g = TaskGraph.from_root("Root", node_id="r")
        g.add_child("r", "A", node_id="a")
        g.add_child("r", "B", node_id="b")
        g.add_child("r", "C", node_id="c")
        ready_ids = {n.id for n in g.ready_nodes()}
        assert ready_ids == {"a", "b", "c"}

    def test_ready_nodes_with_depends_on(self):
        """Subtask with depends_on waits for specified dependencies."""
        g = TaskGraph.from_root("Root", node_id="r")
        g.add_child("r", "A", node_id="a")
        g.add_child("r", "B", node_id="b", depends_on=["a"])
        g.add_child("r", "C", node_id="c")

        ready_ids = {n.id for n in g.ready_nodes()}
        # a and c are ready (no deps), b waits for a
        assert "a" in ready_ids
        assert "c" in ready_ids
        assert "b" not in ready_ids

        # Complete a → b should now be ready
        g.transition("a", "executing")
        g.mark_done("a", {"result": "ok"})
        ready_ids = {n.id for n in g.ready_nodes()}
        assert "b" in ready_ids

    def test_ready_nodes_synthesis_auto_depends(self):
        """Synthesis node waits for all non-synthesis siblings."""
        g = TaskGraph.from_root("Root", node_id="r")
        g.add_child("r", "Research A", node_id="a")
        g.add_child("r", "Research B", node_id="b")
        g.add_child("r", "Combine results", node_id="s", is_synthesis=True)

        ready_ids = {n.id for n in g.ready_nodes()}
        # a and b are ready, synthesis node s is not
        assert "a" in ready_ids
        assert "b" in ready_ids
        assert "s" not in ready_ids

        # Complete a only → s still not ready (b is not done)
        g.transition("a", "executing")
        g.mark_done("a", {"result": "A done"})
        ready_ids = {n.id for n in g.ready_nodes()}
        assert "s" not in ready_ids

        # Complete b → s should now be ready
        g.transition("b", "executing")
        g.mark_done("b", {"result": "B done"})
        ready_ids = {n.id for n in g.ready_nodes()}
        assert "s" in ready_ids

    def test_ready_nodes_depends_on_multiple(self):
        """Subtask depending on multiple predecessors waits for all."""
        g = TaskGraph.from_root("Root", node_id="r")
        g.add_child("r", "A", node_id="a")
        g.add_child("r", "B", node_id="b")
        g.add_child("r", "C", node_id="c", depends_on=["a", "b"])

        ready_ids = {n.id for n in g.ready_nodes()}
        assert "a" in ready_ids
        assert "b" in ready_ids
        assert "c" not in ready_ids

        # Complete a only → c still not ready
        g.transition("a", "executing")
        g.mark_done("a", {"r": 1})
        ready_ids = {n.id for n in g.ready_nodes()}
        assert "c" not in ready_ids

        # Complete b → c is ready
        g.transition("b", "executing")
        g.mark_done("b", {"r": 2})
        ready_ids = {n.id for n in g.ready_nodes()}
        assert "c" in ready_ids

    def test_add_child_with_depends_on(self):
        """add_child wires depends_on correctly."""
        g = TaskGraph.from_root("Root", node_id="r")
        a = g.add_child("r", "A", node_id="a")
        b = g.add_child("r", "B", node_id="b", depends_on=["a"])
        assert a.depends_on == []
        assert b.depends_on == ["a"]


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


class TestStateTransitions:

    def test_valid_pending_to_executing(self):
        g = TaskGraph.from_root("T", node_id="r")
        g.transition("r", "executing")
        assert g.root.status == "executing"

    def test_valid_pending_to_decomposing(self):
        g = TaskGraph.from_root("T", node_id="r")
        g.transition("r", "decomposing")
        assert g.root.status == "decomposing"

    def test_valid_executing_to_reviewing(self):
        g = TaskGraph.from_root("T", node_id="r")
        g.transition("r", "executing")
        g.transition("r", "reviewing")
        assert g.root.status == "reviewing"

    def test_valid_reviewing_to_done(self):
        g = TaskGraph.from_root("T", node_id="r")
        g.transition("r", "executing")
        g.transition("r", "reviewing")
        g.transition("r", "done")
        assert g.root.status == "done"

    def test_critique_cycle(self):
        g = TaskGraph.from_root("T", node_id="r")
        g.transition("r", "decomposing")
        g.transition("r", "critiqued")
        g.transition("r", "decomposing")  # re-plan after critique
        g.transition("r", "critiqued")
        g.transition("r", "executing")
        assert g.root.status == "executing"

    def test_invalid_transition_raises(self):
        g = TaskGraph.from_root("T", node_id="r")
        with pytest.raises(ValueError, match="Invalid transition"):
            g.transition("r", "done")  # pending -> done is invalid

    def test_terminal_states_no_exit(self):
        g = TaskGraph.from_root("T", node_id="r")
        g.transition("r", "executing")
        g.transition("r", "done")
        with pytest.raises(ValueError, match="Invalid transition"):
            g.transition("r", "executing")

    def test_failed_is_terminal(self):
        g = TaskGraph.from_root("T", node_id="r")
        g.transition("r", "failed")
        with pytest.raises(ValueError, match="Invalid transition"):
            g.transition("r", "pending")

    def test_transition_unknown_node_raises(self):
        g = TaskGraph.from_root("T", node_id="r")
        with pytest.raises(KeyError):
            g.transition("missing", "executing")

    def test_any_status_can_fail(self):
        """Every non-terminal status can transition to failed."""
        for status in ("pending", "decomposing", "critiqued", "executing", "reviewing"):
            g = TaskGraph.from_root("T", node_id="r")
            g.nodes["r"].status = status  # force status for test
            g.transition("r", "failed")
            assert g.root.status == "failed"


# ---------------------------------------------------------------------------
# Completion propagation
# ---------------------------------------------------------------------------


class TestCompletion:

    def test_mark_done_leaf(self):
        g = TaskGraph.from_root("T", node_id="r")
        g.transition("r", "executing")
        g.mark_done("r", {"answer": "42"})
        assert g.root.status == "done"
        assert g.root.result == {"answer": "42"}

    def test_mark_done_requires_executing_or_reviewing(self):
        g = TaskGraph.from_root("T", node_id="r")
        with pytest.raises(ValueError, match="Cannot mark_done"):
            g.mark_done("r")

    def test_mark_done_propagates_to_parent(self):
        g = TaskGraph.from_root("Root", node_id="r")
        g.add_child("r", "A", node_id="a")
        g.add_child("r", "B", node_id="b")

        g.transition("a", "executing")
        g.mark_done("a", {"a": 1})
        assert g.root.status != "done"  # b still pending

        g.transition("b", "executing")
        g.mark_done("b", {"b": 2})
        assert g.root.status == "done"
        assert g.root.result == {"a": {"a": 1}, "b": {"b": 2}}

    def test_mark_done_propagates_recursively(self):
        g = TaskGraph.from_root("Root", node_id="r")
        g.add_child("r", "A", node_id="a")
        g.add_child("a", "A1", node_id="a1")

        g.transition("a1", "executing")
        g.mark_done("a1", {"val": "ok"})

        # a should be done (only child a1 is done)
        assert g.nodes["a"].status == "done"
        # root should be done (only child a is done)
        assert g.root.status == "done"

    def test_mark_done_does_not_propagate_with_pending_sibling(self):
        g = TaskGraph.from_root("Root", node_id="r")
        g.add_child("r", "A", node_id="a")
        g.add_child("r", "B", node_id="b")

        g.transition("a", "executing")
        g.mark_done("a")
        assert g.root.status != "done"

    def test_mark_failed(self):
        g = TaskGraph.from_root("T", node_id="r")
        g.transition("r", "executing")
        g.mark_failed("r", "timeout")
        assert g.root.status == "failed"
        assert g.root.result == {"error": "timeout"}

    def test_mark_failed_idempotent(self):
        g = TaskGraph.from_root("T", node_id="r")
        g.transition("r", "failed")
        g.mark_failed("r", "again")  # no error
        assert g.root.status == "failed"

    def test_mark_failed_does_not_propagate(self):
        g = TaskGraph.from_root("Root", node_id="r")
        g.add_child("r", "A", node_id="a")
        g.add_child("r", "B", node_id="b")
        g.mark_failed("a", "oops")
        assert g.root.status == "pending"  # parent not affected

    def test_mark_done_unknown_node_raises(self):
        g = TaskGraph.from_root("T", node_id="r")
        with pytest.raises(KeyError):
            g.mark_done("missing")

    def test_mark_failed_unknown_node_raises(self):
        g = TaskGraph.from_root("T", node_id="r")
        with pytest.raises(KeyError):
            g.mark_failed("missing")


# ---------------------------------------------------------------------------
# TaskNode properties
# ---------------------------------------------------------------------------


class TestNodeProperties:

    def test_is_leaf(self):
        node = TaskNode(id="x", description="leaf")
        assert node.is_leaf is True

    def test_is_not_leaf(self):
        node = TaskNode(id="x", description="parent", children=["c1"])
        assert node.is_leaf is False

    def test_is_terminal_done(self):
        node = TaskNode(id="x", description="d", status="done")
        assert node.is_terminal is True

    def test_is_terminal_failed(self):
        node = TaskNode(id="x", description="d", status="failed")
        assert node.is_terminal is True

    def test_not_terminal(self):
        for s in ("pending", "decomposing", "critiqued", "executing", "reviewing"):
            node = TaskNode(id="x", description="d", status=s)
            assert node.is_terminal is False


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


class TestInspection:

    def test_all_done(self):
        g = TaskGraph.from_root("T", node_id="r")
        g.transition("r", "executing")
        g.mark_done("r")
        assert g.all_done() is True

    def test_not_all_done(self):
        g = TaskGraph.from_root("T", node_id="r")
        assert g.all_done() is False

    def test_has_failures(self):
        g = TaskGraph.from_root("Root", node_id="r")
        g.add_child("r", "A", node_id="a")
        g.mark_failed("a")
        assert g.has_failures() is True

    def test_no_failures(self):
        g = TaskGraph.from_root("T", node_id="r")
        assert g.has_failures() is False

    def test_max_depth(self):
        g = TaskGraph.from_root("Root", node_id="r")
        g.add_child("r", "A", node_id="a")
        g.add_child("a", "A1", node_id="a1")
        g.add_child("a1", "A1a", node_id="a1a")
        assert g.max_depth() == 3

    def test_summary(self):
        g = TaskGraph.from_root("Root", node_id="r")
        g.add_child("r", "A", node_id="a")
        g.add_child("r", "B", node_id="b")
        g.transition("a", "executing")
        g.mark_done("a")

        s = g.summary()
        assert s["node_count"] == 3
        assert s["all_done"] is False
        assert s["has_failures"] is False
        assert s["by_status"]["done"] == 1
        assert s["by_status"]["pending"] == 2  # root + b

    def test_node_count(self):
        g = TaskGraph.from_root("T", node_id="r")
        assert g.node_count() == 1
        g.add_child("r", "A", node_id="a")
        assert g.node_count() == 2


# ---------------------------------------------------------------------------
# Synthesis flag
# ---------------------------------------------------------------------------


class TestSynthesisFlag:

    def test_default_is_false(self):
        node = TaskNode(id="n", description="test")
        assert node.is_synthesis is False

    def test_depends_on_default_is_empty(self):
        node = TaskNode(id="n", description="test")
        assert node.depends_on == []

    def test_add_child_with_synthesis(self):
        g = TaskGraph.from_root("Task", node_id="r")
        g.add_child("r", "Research A", node_id="a")
        synth = g.add_child("r", "Compare A and B", node_id="s", is_synthesis=True)
        assert synth.is_synthesis is True
        assert g.nodes["a"].is_synthesis is False

    def test_add_child_without_synthesis(self):
        g = TaskGraph.from_root("Task", node_id="r")
        child = g.add_child("r", "Work node", node_id="w")
        assert child.is_synthesis is False
