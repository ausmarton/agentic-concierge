"""Tests for resume_execute_task and checkpoint wiring in execute_task.

Tests the full resume flow: create checkpoint → simulate interruption →
resume from checkpoint → verify completed nodes are skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_concierge.application.execute_task import (
    _create_graph_checkpoint,
    _delete_graph_checkpoint,
    _update_graph_checkpoint,
    resume_execute_task,
)
from agentic_concierge.application.graph_checkpoint import serialize_graph
from agentic_concierge.application.task_graph import TaskGraph
from agentic_concierge.config import ConciergeConfig, ModelConfig
from agentic_concierge.domain import RunId, RunResult
from agentic_concierge.infrastructure.workspace.graph_checkpoint import (
    GraphCheckpoint,
    delete_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_cfg() -> ModelConfig:
    return ModelConfig(base_url="http://localhost:11434", model="llama3.1:8b")


def _config(**overrides) -> ConciergeConfig:
    from agentic_concierge.config.schema import SpecialistConfig
    models = overrides.pop("models", {"quality": _model_cfg()})
    specialists = overrides.pop("specialists", {
        "research": SpecialistConfig(description="Research specialist"),
        "engineering": SpecialistConfig(description="Engineering specialist"),
    })
    return ConciergeConfig(models=models, specialists=specialists, **overrides)


def _run_repo(tmp_path: Path):
    """Minimal RunRepository mock."""
    m = MagicMock()
    m.create_run.return_value = (RunId("test-run"), str(tmp_path / "runs" / "test-run"), str(tmp_path / "runs" / "test-run" / "workspace"))
    m.append_event = MagicMock()
    return m


def _simple_graph() -> TaskGraph:
    g = TaskGraph.from_root("test task", node_id="root")
    g.add_child("root", "child A", node_id="a", required_capabilities=["code_python"])
    g.add_child("root", "child B", node_id="b", required_capabilities=["web_comprehension"])
    g.transition("root", "decomposing")
    g.transition("root", "critiqued")
    return g


def _setup_run_dir(tmp_path: Path) -> str:
    run_dir = tmp_path / "runs" / "test-run"
    run_dir.mkdir(parents=True)
    workspace = run_dir / "workspace"
    workspace.mkdir()
    return str(run_dir)


# ---------------------------------------------------------------------------
# Checkpoint helper tests
# ---------------------------------------------------------------------------


class TestCreateGraphCheckpoint:
    def test_creates_checkpoint_file(self, tmp_path):
        run_dir = _setup_run_dir(tmp_path)
        graph = _simple_graph()
        _create_graph_checkpoint(
            run_dir, RunId("test-run"), "test task", "llama3.1:8b",
            ["engineering"], "planner", graph,
        )
        ckpt = load_checkpoint(run_dir)
        assert ckpt is not None
        assert ckpt.run_id == "test-run"
        assert ckpt.prompt == "test task"
        assert ckpt.model_name == "llama3.1:8b"
        assert ckpt.specialist_ids == ["engineering"]

    def test_create_checkpoint_failure_logged_not_raised(self, tmp_path):
        """Checkpoint failures should not crash execute_task."""
        # Pass a bogus run_dir that can't be written to
        with patch(
            "agentic_concierge.application.execute_task.logger"
        ) as mock_logger:
            _create_graph_checkpoint(
                "/nonexistent/path/run-dir", RunId("x"), "p", "m",
                [], "planner", _simple_graph(),
            )
            # Should have logged a warning
            mock_logger.warning.assert_called()


class TestUpdateGraphCheckpoint:
    def test_updates_graph_data(self, tmp_path):
        run_dir = _setup_run_dir(tmp_path)
        graph = _simple_graph()
        _create_graph_checkpoint(
            run_dir, RunId("test-run"), "task", "llama3.1:8b",
            [], "planner", graph,
        )
        # Simulate node completion
        graph.transition("a", "executing")
        graph.mark_done("a", {"summary": "ok"})
        _update_graph_checkpoint(run_dir, graph, {"a": {"summary": "ok"}})

        ckpt = load_checkpoint(run_dir)
        assert ckpt.completed_node_results == {"a": {"summary": "ok"}}
        # The graph_data should reflect 'a' as done
        from agentic_concierge.application.graph_checkpoint import deserialize_graph
        restored = deserialize_graph(ckpt.graph_data)
        assert restored.nodes["a"].status == "done"

    def test_update_noop_when_no_checkpoint(self, tmp_path):
        run_dir = _setup_run_dir(tmp_path)
        # Should not raise
        _update_graph_checkpoint(run_dir, _simple_graph(), {})


class TestDeleteGraphCheckpoint:
    def test_deletes_checkpoint(self, tmp_path):
        run_dir = _setup_run_dir(tmp_path)
        _create_graph_checkpoint(
            run_dir, RunId("r"), "t", "m", [], "planner", _simple_graph(),
        )
        assert load_checkpoint(run_dir) is not None
        _delete_graph_checkpoint(run_dir)
        assert load_checkpoint(run_dir) is None

    def test_delete_noop_when_absent(self, tmp_path):
        run_dir = _setup_run_dir(tmp_path)
        _delete_graph_checkpoint(run_dir)  # no error


# ---------------------------------------------------------------------------
# resume_execute_task
# ---------------------------------------------------------------------------


class TestResumeExecuteTask:
    @pytest.mark.asyncio
    async def test_no_checkpoint_raises_value_error(self, tmp_path):
        run_dir = _setup_run_dir(tmp_path)
        with pytest.raises(ValueError, match="No checkpoint"):
            await resume_execute_task(
                RunId("test-run"),
                run_dir,
                chat_client=MagicMock(),
                run_repository=_run_repo(tmp_path),
                specialist_registry=MagicMock(),
                config=_config(),
            )

    @pytest.mark.asyncio
    async def test_empty_graph_data_raises_value_error(self, tmp_path):
        run_dir = _setup_run_dir(tmp_path)
        ckpt = GraphCheckpoint(
            run_id="test-run", prompt="t", model_name="m",
            graph_data={},
        )
        save_checkpoint(run_dir, ckpt)
        with pytest.raises(ValueError, match="no graph data"):
            await resume_execute_task(
                RunId("test-run"),
                run_dir,
                chat_client=MagicMock(),
                run_repository=_run_repo(tmp_path),
                specialist_registry=MagicMock(),
                config=_config(),
            )

    @pytest.mark.asyncio
    async def test_resume_skips_done_nodes(self, tmp_path):
        """Resumed execution should not re-execute nodes that are already done."""
        run_dir = _setup_run_dir(tmp_path)
        graph = _simple_graph()
        # Mark 'a' as done
        graph.transition("a", "executing")
        graph.mark_done("a", {"summary": "done A"})
        ckpt = GraphCheckpoint(
            run_id="test-run",
            prompt="test task",
            model_name="llama3.1:8b",
            specialist_ids=["engineering"],
            graph_data=serialize_graph(graph),
            routing_method="planner",
            completed_node_results={"a": {"summary": "done A"}},
        )
        save_checkpoint(run_dir, ckpt)

        executed_nodes: List[str] = []

        # Mock the entire execution pipeline
        async def _mock_leaf(node, graph, assignment):
            executed_nodes.append(node.id)
            return {"summary": f"done {node.id}"}

        mock_assignment = MagicMock()
        mock_assignment.model_id = "llama3.1:8b"
        mock_assignment.handle.chat_client = MagicMock()
        mock_assignment.handle.model_id = "llama3.1:8b"
        mock_assignment.handle.slot = None
        mock_assignment.role = "worker"

        from agentic_concierge.application.graph_executor import ExecutionResult
        from agentic_concierge.application.graph_checkpoint import deserialize_graph, prepare_graph_for_resume

        # The resume function will call execute_graph_with_affinity.
        # We mock it to simulate executing only pending nodes.
        async def mock_execute_graph_with_affinity(
            g, runtime, work_executor, *, max_steps=50, on_event=None, prior_results=None,
        ):
            # Simulate: 'a' is already done, only 'b' should be executed
            results = dict(prior_results or {})
            for leaf in g.ready_nodes():
                g.transition(leaf.id, "executing")
                executed_nodes.append(leaf.id)
                g.mark_done(leaf.id, {"summary": f"done {leaf.id}"})
                results[leaf.id] = {"summary": f"done {leaf.id}"}
            return ExecutionResult(
                graph=g,
                completed=g.all_done(),
                results=results,
                failures={},
                steps_executed=len(executed_nodes),
            )

        registry = MagicMock()
        registry.get_pack.return_value = MagicMock(system_prompt="You are a helper.")

        with patch(
            "agentic_concierge.application.affinity_executor.execute_graph_with_affinity",
            mock_execute_graph_with_affinity,
        ):
            result = await resume_execute_task(
                RunId("test-run"),
                run_dir,
                chat_client=MagicMock(),
                run_repository=_run_repo(tmp_path),
                specialist_registry=registry,
                config=_config(),
            )

        # 'a' was already done — only 'b' should have been executed
        assert "a" not in executed_nodes
        assert "b" in executed_nodes
        assert isinstance(result, RunResult)
        assert result.run_id.value == "test-run"

    @pytest.mark.asyncio
    async def test_resume_deletes_checkpoint_on_completion(self, tmp_path):
        """Checkpoint should be deleted after successful completion."""
        run_dir = _setup_run_dir(tmp_path)
        graph = _simple_graph()
        graph.transition("a", "executing")
        graph.mark_done("a", {"ok": True})
        ckpt = GraphCheckpoint(
            run_id="test-run",
            prompt="task",
            model_name="llama3.1:8b",
            specialist_ids=["engineering"],
            graph_data=serialize_graph(graph),
            completed_node_results={"a": {"ok": True}},
        )
        save_checkpoint(run_dir, ckpt)

        from agentic_concierge.application.graph_executor import ExecutionResult

        async def mock_execute(g, runtime, work_executor, *, max_steps=50, on_event=None, prior_results=None):
            # Complete all remaining nodes
            results = dict(prior_results or {})
            for leaf in g.ready_nodes():
                g.transition(leaf.id, "executing")
                g.mark_done(leaf.id, {"ok": True})
                results[leaf.id] = {"ok": True}
            return ExecutionResult(
                graph=g, completed=g.all_done(),
                results=results, failures={}, steps_executed=1,
            )

        with patch(
            "agentic_concierge.application.affinity_executor.execute_graph_with_affinity",
            mock_execute,
        ):
            await resume_execute_task(
                RunId("test-run"), run_dir,
                chat_client=MagicMock(),
                run_repository=_run_repo(tmp_path),
                specialist_registry=MagicMock(),
                config=_config(),
            )

        # Checkpoint should be deleted after completion
        assert load_checkpoint(run_dir) is None

    @pytest.mark.asyncio
    async def test_resume_preserves_checkpoint_on_partial(self, tmp_path):
        """Checkpoint should be preserved if execution is incomplete."""
        run_dir = _setup_run_dir(tmp_path)
        graph = _simple_graph()
        ckpt = GraphCheckpoint(
            run_id="test-run",
            prompt="task",
            model_name="llama3.1:8b",
            specialist_ids=["engineering"],
            graph_data=serialize_graph(graph),
        )
        save_checkpoint(run_dir, ckpt)

        from agentic_concierge.application.graph_executor import ExecutionResult

        async def mock_execute(g, runtime, work_executor, *, max_steps=50, on_event=None, prior_results=None):
            # Only execute 'a', leave 'b' pending
            results = dict(prior_results or {})
            for leaf in g.ready_nodes():
                g.transition(leaf.id, "executing")
                g.mark_done(leaf.id, {"ok": True})
                results[leaf.id] = {"ok": True}
                break  # only first one
            return ExecutionResult(
                graph=g, completed=False,
                results=results, failures={}, steps_executed=1,
            )

        with patch(
            "agentic_concierge.application.affinity_executor.execute_graph_with_affinity",
            mock_execute,
        ):
            await resume_execute_task(
                RunId("test-run"), run_dir,
                chat_client=MagicMock(),
                run_repository=_run_repo(tmp_path),
                specialist_registry=MagicMock(),
                config=_config(),
            )

        # Checkpoint should still exist (incomplete)
        assert load_checkpoint(run_dir) is not None

    @pytest.mark.asyncio
    async def test_resume_emits_resume_event(self, tmp_path):
        """A 'resume' event should be logged."""
        run_dir = _setup_run_dir(tmp_path)
        graph = _simple_graph()
        ckpt = GraphCheckpoint(
            run_id="test-run",
            prompt="task",
            model_name="llama3.1:8b",
            specialist_ids=["engineering"],
            graph_data=serialize_graph(graph),
        )
        save_checkpoint(run_dir, ckpt)

        from agentic_concierge.application.graph_executor import ExecutionResult

        async def mock_execute(g, runtime, work_executor, *, max_steps=50, on_event=None, prior_results=None):
            results = dict(prior_results or {})
            for leaf in g.ready_nodes():
                g.transition(leaf.id, "executing")
                g.mark_done(leaf.id, {"ok": True})
                results[leaf.id] = {"ok": True}
            return ExecutionResult(
                graph=g, completed=g.all_done(),
                results=results, failures={}, steps_executed=2,
            )

        repo = _run_repo(tmp_path)

        with patch(
            "agentic_concierge.application.affinity_executor.execute_graph_with_affinity",
            mock_execute,
        ):
            await resume_execute_task(
                RunId("test-run"), run_dir,
                chat_client=MagicMock(),
                run_repository=repo,
                specialist_registry=MagicMock(),
                config=_config(),
            )

        # Check that a 'resume' event was emitted
        event_kinds = [call.args[1] for call in repo.append_event.call_args_list]
        assert "resume" in event_kinds

    @pytest.mark.asyncio
    async def test_resume_resets_executing_nodes(self, tmp_path):
        """In-flight nodes should be reset to pending on resume."""
        run_dir = _setup_run_dir(tmp_path)
        graph = _simple_graph()
        # Simulate 'a' was executing when interrupted
        graph.transition("a", "executing")
        ckpt = GraphCheckpoint(
            run_id="test-run",
            prompt="task",
            model_name="llama3.1:8b",
            specialist_ids=["engineering"],
            graph_data=serialize_graph(graph),
        )
        save_checkpoint(run_dir, ckpt)

        from agentic_concierge.application.graph_executor import ExecutionResult

        # Track what nodes get executed
        executed_nodes: List[str] = []

        async def mock_execute(g, runtime, work_executor, *, max_steps=50, on_event=None, prior_results=None):
            results = dict(prior_results or {})
            # All nodes should be pending now (a was reset from executing)
            for leaf in g.ready_nodes():
                executed_nodes.append(leaf.id)
                g.transition(leaf.id, "executing")
                g.mark_done(leaf.id, {"ok": True})
                results[leaf.id] = {"ok": True}
            # After first pass, check for more ready
            for leaf in g.ready_nodes():
                executed_nodes.append(leaf.id)
                g.transition(leaf.id, "executing")
                g.mark_done(leaf.id, {"ok": True})
                results[leaf.id] = {"ok": True}
            return ExecutionResult(
                graph=g, completed=g.all_done(),
                results=results, failures={}, steps_executed=len(executed_nodes),
            )

        with patch(
            "agentic_concierge.application.affinity_executor.execute_graph_with_affinity",
            mock_execute,
        ):
            await resume_execute_task(
                RunId("test-run"), run_dir,
                chat_client=MagicMock(),
                run_repository=_run_repo(tmp_path),
                specialist_registry=MagicMock(),
                config=_config(),
            )

        # 'a' should have been re-executed (it was reset from executing to pending)
        assert "a" in executed_nodes
