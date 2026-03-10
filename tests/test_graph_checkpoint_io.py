"""Tests for graph checkpoint I/O (infrastructure layer).

Covers save/load/delete, atomic writes, find_resumable_runs, and edge cases.
All tests use tmp_path so no real workspace is touched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agentic_concierge.application.graph_checkpoint import serialize_graph
from agentic_concierge.application.task_graph import TaskGraph
from agentic_concierge.infrastructure.workspace.graph_checkpoint import (
    GraphCheckpoint,
    ResumableRun,
    _CHECKPOINT_FILENAME,
    delete_checkpoint,
    find_resumable_runs,
    load_checkpoint,
    load_graph_from_checkpoint,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph() -> TaskGraph:
    """Root with two children."""
    g = TaskGraph.from_root("test task", node_id="root")
    g.add_child("root", "child A", node_id="a", required_capabilities=["code_python"])
    g.add_child("root", "child B", node_id="b")
    return g


def _make_checkpoint(graph: TaskGraph, run_id: str = "run-001") -> GraphCheckpoint:
    return GraphCheckpoint(
        run_id=run_id,
        prompt="test task",
        model_name="llama3.1:8b",
        specialist_ids=["engineering"],
        graph_data=serialize_graph(graph),
        routing_method="planner",
    )


def _setup_run_dir(tmp_path: Path, run_id: str = "run-001") -> Path:
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    return run_dir


# ---------------------------------------------------------------------------
# save_checkpoint / load_checkpoint
# ---------------------------------------------------------------------------

class TestSaveLoadCheckpoint:
    def test_save_and_load_round_trip(self, tmp_path):
        run_dir = _setup_run_dir(tmp_path)
        graph = _make_graph()
        ckpt = _make_checkpoint(graph)
        save_checkpoint(run_dir, ckpt)
        loaded = load_checkpoint(run_dir)
        assert loaded is not None
        assert loaded.run_id == "run-001"
        assert loaded.prompt == "test task"
        assert loaded.model_name == "llama3.1:8b"
        assert loaded.specialist_ids == ["engineering"]
        assert loaded.routing_method == "planner"
        assert loaded.graph_data["root_id"] == "root"
        assert len(loaded.graph_data["nodes"]) == 3

    def test_sets_timestamps(self, tmp_path):
        run_dir = _setup_run_dir(tmp_path)
        ckpt = _make_checkpoint(_make_graph())
        assert ckpt.created_at == 0.0
        save_checkpoint(run_dir, ckpt)
        assert ckpt.created_at > 0
        assert ckpt.updated_at > 0
        assert ckpt.created_at == ckpt.updated_at

    def test_updates_timestamp_on_resave(self, tmp_path):
        run_dir = _setup_run_dir(tmp_path)
        ckpt = _make_checkpoint(_make_graph())
        save_checkpoint(run_dir, ckpt)
        first_ts = ckpt.updated_at
        # Save again — updated_at should change, created_at should not
        import time
        time.sleep(0.01)  # ensure different timestamp
        save_checkpoint(run_dir, ckpt)
        assert ckpt.updated_at >= first_ts
        assert ckpt.created_at == first_ts  # created_at preserved (non-zero)

    def test_file_is_valid_json(self, tmp_path):
        run_dir = _setup_run_dir(tmp_path)
        save_checkpoint(run_dir, _make_checkpoint(_make_graph()))
        path = run_dir / _CHECKPOINT_FILENAME
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["run_id"] == "run-001"

    def test_load_nonexistent_returns_none(self, tmp_path):
        run_dir = _setup_run_dir(tmp_path)
        assert load_checkpoint(run_dir) is None

    def test_overwrite_existing(self, tmp_path):
        run_dir = _setup_run_dir(tmp_path)
        save_checkpoint(run_dir, _make_checkpoint(_make_graph(), run_id="first"))
        save_checkpoint(run_dir, _make_checkpoint(_make_graph(), run_id="second"))
        loaded = load_checkpoint(run_dir)
        assert loaded.run_id == "second"

    def test_creates_parent_dirs(self, tmp_path):
        run_dir = tmp_path / "deep" / "nested" / "run-dir"
        # run_dir doesn't exist yet
        save_checkpoint(run_dir, _make_checkpoint(_make_graph()))
        loaded = load_checkpoint(run_dir)
        assert loaded is not None

    def test_completed_node_results_persist(self, tmp_path):
        run_dir = _setup_run_dir(tmp_path)
        ckpt = _make_checkpoint(_make_graph())
        ckpt.completed_node_results = {"a": {"summary": "done"}}
        save_checkpoint(run_dir, ckpt)
        loaded = load_checkpoint(run_dir)
        assert loaded.completed_node_results == {"a": {"summary": "done"}}


# ---------------------------------------------------------------------------
# delete_checkpoint
# ---------------------------------------------------------------------------

class TestDeleteCheckpoint:
    def test_delete_existing(self, tmp_path):
        run_dir = _setup_run_dir(tmp_path)
        save_checkpoint(run_dir, _make_checkpoint(_make_graph()))
        assert (run_dir / _CHECKPOINT_FILENAME).exists()
        delete_checkpoint(run_dir)
        assert not (run_dir / _CHECKPOINT_FILENAME).exists()

    def test_delete_nonexistent_no_error(self, tmp_path):
        run_dir = _setup_run_dir(tmp_path)
        delete_checkpoint(run_dir)  # should not raise

    def test_delete_dir_nonexistent_no_error(self, tmp_path):
        run_dir = tmp_path / "nonexistent"
        delete_checkpoint(run_dir)  # should not raise


# ---------------------------------------------------------------------------
# load_graph_from_checkpoint
# ---------------------------------------------------------------------------

class TestLoadGraphFromCheckpoint:
    def test_loads_graph(self, tmp_path):
        run_dir = _setup_run_dir(tmp_path)
        graph = _make_graph()
        save_checkpoint(run_dir, _make_checkpoint(graph))
        restored = load_graph_from_checkpoint(run_dir)
        assert restored is not None
        assert restored.root_id == "root"
        assert set(restored.nodes.keys()) == {"root", "a", "b"}

    def test_returns_none_when_no_checkpoint(self, tmp_path):
        run_dir = _setup_run_dir(tmp_path)
        assert load_graph_from_checkpoint(run_dir) is None

    def test_returns_none_when_empty_graph_data(self, tmp_path):
        run_dir = _setup_run_dir(tmp_path)
        ckpt = GraphCheckpoint(
            run_id="r1", prompt="t", model_name="m",
            graph_data={},
        )
        save_checkpoint(run_dir, ckpt)
        assert load_graph_from_checkpoint(run_dir) is None

    def test_graph_methods_work(self, tmp_path):
        run_dir = _setup_run_dir(tmp_path)
        graph = _make_graph()
        graph.transition("a", "executing")
        graph.mark_done("a", {"ok": True})
        save_checkpoint(run_dir, _make_checkpoint(graph))
        restored = load_graph_from_checkpoint(run_dir)
        assert restored.nodes["a"].status == "done"
        ready = restored.ready_nodes()
        assert len(ready) == 1
        assert ready[0].id == "b"


# ---------------------------------------------------------------------------
# find_resumable_runs
# ---------------------------------------------------------------------------

class TestFindResumableRuns:
    def test_empty_workspace(self, tmp_path):
        assert find_resumable_runs(tmp_path) == []

    def test_no_runs_dir(self, tmp_path):
        assert find_resumable_runs(tmp_path / "nonexistent") == []

    def test_finds_incomplete_run(self, tmp_path):
        run_dir = _setup_run_dir(tmp_path, "run-001")
        graph = _make_graph()
        graph.transition("a", "executing")
        graph.mark_done("a", {"ok": True})
        # b is still pending, root not done
        save_checkpoint(run_dir, _make_checkpoint(graph, "run-001"))
        results = find_resumable_runs(tmp_path)
        assert len(results) == 1
        assert results[0].run_id == "run-001"
        assert results[0].done_nodes == 1  # 'a' is done
        assert results[0].pending_nodes == 2  # root + 'b' are pending
        assert results[0].total_nodes == 3  # root + a + b

    def test_excludes_completed_run(self, tmp_path):
        run_dir = _setup_run_dir(tmp_path, "run-done")
        graph = _make_graph()
        graph.transition("a", "executing")
        graph.mark_done("a", {"ok": True})
        graph.transition("b", "executing")
        graph.mark_done("b", {"ok": True})
        # Root should auto-propagate to done
        assert graph.root.status == "done"
        save_checkpoint(run_dir, _make_checkpoint(graph, "run-done"))
        results = find_resumable_runs(tmp_path)
        assert len(results) == 0

    def test_excludes_runs_without_checkpoint(self, tmp_path):
        # Create a run dir without a checkpoint
        (tmp_path / "runs" / "run-no-ckpt").mkdir(parents=True)
        results = find_resumable_runs(tmp_path)
        assert len(results) == 0

    def test_sorted_by_updated_at(self, tmp_path):
        # Create two resumable runs
        for run_id in ["run-old", "run-new"]:
            run_dir = _setup_run_dir(tmp_path, run_id)
            graph = _make_graph()
            ckpt = _make_checkpoint(graph, run_id)
            save_checkpoint(run_dir, ckpt)

        results = find_resumable_runs(tmp_path)
        assert len(results) == 2
        # Most recent first
        assert results[0].updated_at >= results[1].updated_at

    def test_multiple_mixed(self, tmp_path):
        # run-1: incomplete (resumable)
        run_dir1 = _setup_run_dir(tmp_path, "run-1")
        g1 = _make_graph()
        g1.transition("a", "executing")
        g1.mark_done("a", {"ok": True})
        save_checkpoint(run_dir1, _make_checkpoint(g1, "run-1"))

        # run-2: complete (not resumable)
        run_dir2 = _setup_run_dir(tmp_path, "run-2")
        g2 = _make_graph()
        g2.transition("a", "executing")
        g2.mark_done("a", {"ok": True})
        g2.transition("b", "executing")
        g2.mark_done("b", {"ok": True})
        save_checkpoint(run_dir2, _make_checkpoint(g2, "run-2"))

        # run-3: failed node (still resumable — root not done)
        run_dir3 = _setup_run_dir(tmp_path, "run-3")
        g3 = _make_graph()
        g3.transition("a", "executing")
        g3.mark_failed("a", "crash")
        save_checkpoint(run_dir3, _make_checkpoint(g3, "run-3"))

        results = find_resumable_runs(tmp_path)
        resumable_ids = {r.run_id for r in results}
        assert "run-1" in resumable_ids
        assert "run-2" not in resumable_ids
        assert "run-3" in resumable_ids

    def test_failed_node_counts(self, tmp_path):
        run_dir = _setup_run_dir(tmp_path, "run-fail")
        graph = _make_graph()
        graph.transition("a", "executing")
        graph.mark_failed("a", "error")
        save_checkpoint(run_dir, _make_checkpoint(graph, "run-fail"))
        results = find_resumable_runs(tmp_path)
        assert len(results) == 1
        assert results[0].failed_nodes == 1
        assert results[0].pending_nodes == 2  # root + 'b' still pending

    def test_skips_invalid_checkpoint(self, tmp_path):
        run_dir = _setup_run_dir(tmp_path, "run-bad")
        # Write garbage JSON
        (run_dir / _CHECKPOINT_FILENAME).write_text('{"invalid": true}', encoding="utf-8")
        # Should not crash, just skip
        results = find_resumable_runs(tmp_path)
        assert len(results) == 0

    def test_skips_non_directory_entries(self, tmp_path):
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir(parents=True)
        # Create a file (not a directory) in runs/
        (runs_dir / "not-a-dir.txt").write_text("hello")
        results = find_resumable_runs(tmp_path)
        assert len(results) == 0

    def test_resumable_run_fields(self, tmp_path):
        run_dir = _setup_run_dir(tmp_path, "run-fields")
        graph = _make_graph()
        ckpt = GraphCheckpoint(
            run_id="run-fields",
            prompt="build me a thing",
            model_name="qwen2.5:7b",
            specialist_ids=["engineering", "research"],
            graph_data=serialize_graph(graph),
            routing_method="planner",
        )
        save_checkpoint(run_dir, ckpt)
        results = find_resumable_runs(tmp_path)
        r = results[0]
        assert r.run_id == "run-fields"
        assert r.prompt == "build me a thing"
        assert r.model_name == "qwen2.5:7b"
        assert r.specialist_ids == ["engineering", "research"]
        assert r.routing_method == "planner"


# ---------------------------------------------------------------------------
# Atomic write safety
# ---------------------------------------------------------------------------

class TestAtomicWrite:
    def test_no_temp_files_left(self, tmp_path):
        run_dir = _setup_run_dir(tmp_path)
        save_checkpoint(run_dir, _make_checkpoint(_make_graph()))
        # Only the checkpoint file should exist (no .tmp files)
        files = list(run_dir.iterdir())
        assert len(files) == 1
        assert files[0].name == _CHECKPOINT_FILENAME

    def test_concurrent_save_overwrites_cleanly(self, tmp_path):
        """Simulate two rapid saves — the last one should win."""
        run_dir = _setup_run_dir(tmp_path)
        ckpt1 = _make_checkpoint(_make_graph(), "first")
        ckpt2 = _make_checkpoint(_make_graph(), "second")
        save_checkpoint(run_dir, ckpt1)
        save_checkpoint(run_dir, ckpt2)
        loaded = load_checkpoint(run_dir)
        assert loaded.run_id == "second"
