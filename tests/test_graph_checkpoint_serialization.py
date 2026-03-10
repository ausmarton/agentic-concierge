"""Tests for graph checkpoint serialization (application layer).

Tests serialize_graph / deserialize_graph round-trips, validation,
and prepare_graph_for_resume logic.  No I/O — pure data transformation.
"""

from __future__ import annotations

import copy
import json

import pytest

from agentic_concierge.application.graph_checkpoint import (
    SCHEMA_VERSION,
    deserialize_graph,
    deserialize_node,
    prepare_graph_for_resume,
    serialize_graph,
    serialize_node,
)
from agentic_concierge.application.task_graph import TaskGraph, TaskNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_graph() -> TaskGraph:
    """Root with two leaf children."""
    g = TaskGraph.from_root("parent task", node_id="root")
    g.add_child(
        "root", "child A",
        node_id="a",
        required_capabilities=["code_python"],
        required_tools=["shell"],
        finish_schema_key="code",
    )
    g.add_child(
        "root", "child B",
        node_id="b",
        required_capabilities=["web_comprehension"],
        required_tools=["web_search"],
        finish_schema_key="quick_answer",
    )
    return g


def _deep_graph() -> TaskGraph:
    """Root → mid → leaf_1, leaf_2."""
    g = TaskGraph.from_root("root task", node_id="root")
    g.add_child("root", "mid node", node_id="mid")
    g.add_child("mid", "leaf 1", node_id="leaf1", required_capabilities=["reasoning"])
    g.add_child("mid", "leaf 2", node_id="leaf2")
    return g


# ---------------------------------------------------------------------------
# serialize_node / deserialize_node
# ---------------------------------------------------------------------------

class TestNodeSerialization:
    def test_round_trip_defaults(self):
        node = TaskNode(id="n1", description="a task")
        data = serialize_node(node)
        restored = deserialize_node(data)
        assert restored.id == node.id
        assert restored.description == node.description
        assert restored.parent_id is None
        assert restored.status == "pending"
        assert restored.children == []
        assert restored.required_capabilities == []
        assert restored.required_tools == []
        assert restored.finish_schema_key == "general"
        assert restored.is_synthesis is False
        assert restored.result is None
        assert restored.critique is None
        assert restored.depth == 0

    def test_round_trip_all_fields(self):
        node = TaskNode(
            id="n2",
            description="complex task",
            parent_id="p1",
            status="done",
            children=["c1", "c2"],
            required_capabilities=["code_python", "reasoning"],
            required_tools=["shell", "run_tests"],
            finish_schema_key="code",
            is_synthesis=True,
            result={"summary": "done", "files_modified": ["a.py"]},
            critique="Looks good",
            depth=3,
        )
        data = serialize_node(node)
        restored = deserialize_node(data)
        assert restored.id == "n2"
        assert restored.description == "complex task"
        assert restored.parent_id == "p1"
        assert restored.status == "done"
        assert restored.children == ["c1", "c2"]
        assert restored.required_capabilities == ["code_python", "reasoning"]
        assert restored.required_tools == ["shell", "run_tests"]
        assert restored.finish_schema_key == "code"
        assert restored.is_synthesis is True
        assert restored.result == {"summary": "done", "files_modified": ["a.py"]}
        assert restored.critique == "Looks good"
        assert restored.depth == 3

    def test_round_trip_synthesis_false(self):
        """is_synthesis=False is preserved through serialization."""
        node = TaskNode(id="n", description="work node", is_synthesis=False)
        data = serialize_node(node)
        assert data["is_synthesis"] is False
        restored = deserialize_node(data)
        assert restored.is_synthesis is False

    def test_round_trip_synthesis_true(self):
        """is_synthesis=True is preserved through serialization."""
        node = TaskNode(id="n", description="compare results", is_synthesis=True)
        data = serialize_node(node)
        assert data["is_synthesis"] is True
        restored = deserialize_node(data)
        assert restored.is_synthesis is True

    def test_deserialize_missing_is_synthesis_defaults_false(self):
        """Legacy data without is_synthesis field defaults to False."""
        data = {"id": "n", "description": "old node", "status": "pending"}
        node = deserialize_node(data)
        assert node.is_synthesis is False

    def test_serialized_is_json_compatible(self):
        node = TaskNode(id="n1", description="task", result={"key": [1, 2, 3]})
        data = serialize_node(node)
        # Must round-trip through JSON without error
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)
        restored = deserialize_node(restored_data)
        assert restored.result == {"key": [1, 2, 3]}

    def test_deserialize_missing_id_raises(self):
        with pytest.raises(ValueError, match="Missing required fields.*id"):
            deserialize_node({"description": "task", "status": "pending"})

    def test_deserialize_missing_description_raises(self):
        with pytest.raises(ValueError, match="Missing required fields.*description"):
            deserialize_node({"id": "n1", "status": "pending"})

    def test_deserialize_missing_status_raises(self):
        with pytest.raises(ValueError, match="Missing required fields.*status"):
            deserialize_node({"id": "n1", "description": "task"})

    def test_deserialize_defaults_for_optional_fields(self):
        """Only id, description, status are required; everything else has defaults."""
        node = deserialize_node({"id": "n1", "description": "t", "status": "pending"})
        assert node.parent_id is None
        assert node.children == []
        assert node.required_capabilities == []
        assert node.required_tools == []
        assert node.finish_schema_key == "general"
        assert node.depth == 0

    def test_lists_are_copies(self):
        """Deserialized lists should be independent copies."""
        data = serialize_node(TaskNode(id="n1", description="t", children=["c1"]))
        restored = deserialize_node(data)
        restored.children.append("c2")
        # Original data should not be modified
        assert data["children"] == ["c1"]


# ---------------------------------------------------------------------------
# serialize_graph / deserialize_graph
# ---------------------------------------------------------------------------

class TestGraphSerialization:
    def test_round_trip_simple(self):
        g = _simple_graph()
        data = serialize_graph(g)
        restored = deserialize_graph(data)
        assert restored.root_id == "root"
        assert set(restored.nodes.keys()) == {"root", "a", "b"}
        assert restored.nodes["a"].description == "child A"
        assert restored.nodes["b"].required_tools == ["web_search"]
        assert restored.nodes["root"].children == ["a", "b"]

    def test_round_trip_deep(self):
        g = _deep_graph()
        data = serialize_graph(g)
        restored = deserialize_graph(data)
        assert restored.root_id == "root"
        assert restored.nodes["mid"].parent_id == "root"
        assert restored.nodes["leaf1"].parent_id == "mid"
        assert restored.nodes["leaf2"].parent_id == "mid"
        assert restored.nodes["leaf1"].depth == 2
        assert restored.nodes["mid"].children == ["leaf1", "leaf2"]

    def test_round_trip_with_done_nodes(self):
        g = _simple_graph()
        g.transition("a", "executing")
        g.mark_done("a", {"summary": "done A"})
        data = serialize_graph(g)
        restored = deserialize_graph(data)
        assert restored.nodes["a"].status == "done"
        assert restored.nodes["a"].result == {"summary": "done A"}

    def test_round_trip_with_failed_nodes(self):
        g = _simple_graph()
        g.transition("a", "executing")
        g.mark_failed("a", "timeout")
        data = serialize_graph(g)
        restored = deserialize_graph(data)
        assert restored.nodes["a"].status == "failed"
        assert restored.nodes["a"].result == {"error": "timeout"}

    def test_round_trip_preserves_critique(self):
        g = _simple_graph()
        g.nodes["a"].critique = "Needs more tests"
        data = serialize_graph(g)
        restored = deserialize_graph(data)
        assert restored.nodes["a"].critique == "Needs more tests"

    def test_schema_version_included(self):
        data = serialize_graph(_simple_graph())
        assert data["schema_version"] == SCHEMA_VERSION

    def test_json_round_trip(self):
        """Full JSON serialization round-trip."""
        g = _simple_graph()
        g.transition("a", "executing")
        g.mark_done("a", {"summary": "ok", "nested": {"key": [1, 2]}})
        json_str = json.dumps(serialize_graph(g))
        data = json.loads(json_str)
        restored = deserialize_graph(data)
        assert restored.nodes["a"].result == {"summary": "ok", "nested": {"key": [1, 2]}}

    def test_graph_methods_work_after_deserialize(self):
        """The deserialized graph should support normal TaskGraph methods."""
        g = _simple_graph()
        data = serialize_graph(g)
        restored = deserialize_graph(data)
        # ready_nodes should return both children (preceding sibling for 'b' is 'a' which is pending)
        # Actually, ready_nodes returns pending leaves whose preceding siblings are done.
        # 'a' has no preceding siblings → ready.  'b' has preceding 'a' which is pending → NOT ready.
        ready = restored.ready_nodes()
        assert len(ready) == 1
        assert ready[0].id == "a"

    def test_leaves_work_after_deserialize(self):
        g = _simple_graph()
        data = serialize_graph(g)
        restored = deserialize_graph(data)
        leaves = restored.leaves()
        leaf_ids = {n.id for n in leaves}
        assert leaf_ids == {"a", "b"}


# ---------------------------------------------------------------------------
# deserialize_graph validation
# ---------------------------------------------------------------------------

class TestGraphDeserializationValidation:
    def test_unsupported_schema_version_zero(self):
        data = {"schema_version": 0, "root_id": "r", "nodes": []}
        with pytest.raises(ValueError, match="Unsupported schema version"):
            deserialize_graph(data)

    def test_unsupported_schema_version_future(self):
        data = {"schema_version": SCHEMA_VERSION + 1, "root_id": "r", "nodes": []}
        with pytest.raises(ValueError, match="Unsupported schema version"):
            deserialize_graph(data)

    def test_no_nodes(self):
        data = {"schema_version": 1, "root_id": "r", "nodes": []}
        with pytest.raises(ValueError, match="No nodes"):
            deserialize_graph(data)

    def test_missing_nodes_key(self):
        data = {"schema_version": 1, "root_id": "r"}
        with pytest.raises(ValueError, match="No nodes"):
            deserialize_graph(data)

    def test_root_id_not_found(self):
        data = {
            "schema_version": 1,
            "root_id": "missing",
            "nodes": [{"id": "n1", "description": "t", "status": "pending"}],
        }
        with pytest.raises(ValueError, match="root_id.*not found"):
            deserialize_graph(data)

    def test_duplicate_node_id(self):
        data = {
            "schema_version": 1,
            "root_id": "n1",
            "nodes": [
                {"id": "n1", "description": "t1", "status": "pending"},
                {"id": "n1", "description": "t2", "status": "pending"},
            ],
        }
        with pytest.raises(ValueError, match="Duplicate node id"):
            deserialize_graph(data)

    def test_dangling_parent_ref(self):
        data = {
            "schema_version": 1,
            "root_id": "n1",
            "nodes": [
                {"id": "n1", "description": "root", "status": "pending"},
                {"id": "n2", "description": "orphan", "status": "pending", "parent_id": "ghost"},
            ],
        }
        with pytest.raises(ValueError, match="parent.*ghost.*does not exist"):
            deserialize_graph(data)

    def test_dangling_child_ref(self):
        data = {
            "schema_version": 1,
            "root_id": "n1",
            "nodes": [
                {"id": "n1", "description": "root", "status": "pending", "children": ["ghost"]},
            ],
        }
        with pytest.raises(ValueError, match="child.*ghost.*does not exist"):
            deserialize_graph(data)


# ---------------------------------------------------------------------------
# prepare_graph_for_resume
# ---------------------------------------------------------------------------

class TestPrepareGraphForResume:
    def test_resets_executing_to_pending(self):
        g = _simple_graph()
        g.transition("a", "executing")
        results = prepare_graph_for_resume(g)
        assert g.nodes["a"].status == "pending"
        assert results == {}

    def test_resets_reviewing_to_pending(self):
        g = _simple_graph()
        g.transition("a", "executing")
        g.transition("a", "reviewing")
        prepare_graph_for_resume(g)
        assert g.nodes["a"].status == "pending"

    def test_resets_decomposing_to_pending(self):
        g = TaskGraph.from_root("task", node_id="root")
        g.transition("root", "decomposing")
        prepare_graph_for_resume(g)
        assert g.nodes["root"].status == "pending"

    def test_clears_result_on_reset(self):
        g = _simple_graph()
        g.transition("a", "executing")
        g.nodes["a"].result = {"partial": True}
        prepare_graph_for_resume(g)
        assert g.nodes["a"].result is None

    def test_preserves_done_nodes(self):
        g = _simple_graph()
        g.transition("a", "executing")
        g.mark_done("a", {"summary": "ok"})
        results = prepare_graph_for_resume(g)
        assert g.nodes["a"].status == "done"
        assert g.nodes["a"].result == {"summary": "ok"}
        assert results == {"a": {"summary": "ok"}}

    def test_preserves_failed_nodes(self):
        g = _simple_graph()
        g.transition("a", "executing")
        g.mark_failed("a", "crash")
        results = prepare_graph_for_resume(g)
        assert g.nodes["a"].status == "failed"
        # Failed nodes are not in results
        assert results == {}

    def test_mixed_states(self):
        """Graph with done, executing, pending, and failed nodes."""
        g = _simple_graph()
        # a = done
        g.transition("a", "executing")
        g.mark_done("a", {"summary": "done A"})
        # b = executing (in-flight)
        g.transition("b", "executing")
        results = prepare_graph_for_resume(g)
        # a stays done, b is reset
        assert g.nodes["a"].status == "done"
        assert g.nodes["b"].status == "pending"
        assert results == {"a": {"summary": "done A"}}

    def test_ready_nodes_after_resume_prep(self):
        """After prepare, the graph should have correct ready nodes."""
        g = _simple_graph()
        g.transition("a", "executing")
        g.mark_done("a", {"summary": "done A"})
        g.transition("b", "executing")  # was in-flight
        prepare_graph_for_resume(g)
        # a is done, b was reset to pending → b should be ready
        ready = g.ready_nodes()
        assert len(ready) == 1
        assert ready[0].id == "b"

    def test_prior_results_merge(self):
        """prior_results from caller are merged with done-node results."""
        g = _simple_graph()
        g.transition("a", "executing")
        g.mark_done("a", {"summary": "ok"})
        existing = {"prev": {"data": 42}}
        results = prepare_graph_for_resume(g, prior_results=existing)
        assert results == {"prev": {"data": 42}, "a": {"summary": "ok"}}
        # Caller's dict should not be modified
        assert existing == {"prev": {"data": 42}}

    def test_done_node_without_result_not_in_results(self):
        """A done node with result=None should not appear in results."""
        g = _simple_graph()
        g.transition("a", "executing")
        g.mark_done("a")  # result=None
        results = prepare_graph_for_resume(g)
        assert "a" not in results

    def test_idempotent(self):
        """Calling prepare_graph_for_resume twice should be safe."""
        g = _simple_graph()
        g.transition("a", "executing")
        r1 = prepare_graph_for_resume(g)
        r2 = prepare_graph_for_resume(g)
        assert g.nodes["a"].status == "pending"
        assert r1 == r2
