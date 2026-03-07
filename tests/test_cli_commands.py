"""Tests for CLI commands using CliRunner (no live LLM required).

Covers the happy path and error paths for:
  - concierge run
  - concierge serve
  - concierge plan
  - concierge resume

All tests use mocks so no Ollama / venv / filesystem I/O is needed.
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from agentic_concierge.interfaces.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_resolved(warnings=None):
    """Minimal stand-in for ResolvedModel."""
    m = MagicMock()
    m.model = "llama3.1:8b"
    m.base_url = "http://localhost:11434"
    m.model_config = MagicMock()
    m.warnings = warnings or []
    m.fallback_used = bool(warnings)
    m.resolved_backend = "ollama"
    return m


def _mock_run_result(specialist_id="engineering", payload=None):
    """Minimal stand-in for RunResult."""
    m = MagicMock()
    m.specialist_id = specialist_id
    m.run_dir = "/tmp/test-run"
    m.workspace_path = "/tmp/test-run/workspace"
    m.model_name = "llama3.1:8b"
    m.payload = payload or {"result": "done"}
    return m


def _make_checkpoint(tmp_path, run_id, completed=None):
    """Build a RunCheckpoint with sane defaults."""
    from agentic_concierge.infrastructure.workspace.run_checkpoint import RunCheckpoint
    now = time.time()
    return RunCheckpoint(
        run_id=run_id,
        run_dir=str(tmp_path / "runs" / run_id),
        workspace_path=str(tmp_path / "runs" / run_id / "workspace"),
        task_prompt="test task",
        specialist_ids=["engineering"],
        completed_specialists=completed if completed is not None else [],
        payloads={},
        task_force_mode="sequential",
        model_key="quality",
        routing_method="orchestrator",
        required_capabilities=[],
        orchestration_plan=None,
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# concierge run
# ---------------------------------------------------------------------------

def test_run_help():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "prompt" in result.output.lower()


def test_run_exits_on_no_llm():
    """When resolve_llm raises RuntimeError the CLI must exit non-zero."""
    with patch(
        "agentic_concierge.interfaces.cli.resolve_llm",
        side_effect=RuntimeError("No LLM available"),
    ):
        result = runner.invoke(app, ["run", "test task"])
    assert result.exit_code != 0


def test_run_success_path():
    """Fully-mocked happy path: run returns a RunResult and prints the panel."""
    mock_result = _mock_run_result()
    with (
        patch("agentic_concierge.interfaces.cli.resolve_llm", return_value=_mock_resolved()),
        patch("agentic_concierge.interfaces.cli.build_chat_client", return_value=MagicMock()),
        patch(
            "agentic_concierge.interfaces.cli.execute_task",
            new_callable=AsyncMock,
            return_value=mock_result,
        ),
    ):
        result = runner.invoke(app, ["run", "--no-stream", "write hello world"])
    assert result.exit_code == 0
    assert "engineering" in result.output


def test_run_shows_model_info():
    """The 'Using model:' dim line must appear on a successful invocation."""
    mock_result = _mock_run_result()
    with (
        patch("agentic_concierge.interfaces.cli.resolve_llm", return_value=_mock_resolved()),
        patch("agentic_concierge.interfaces.cli.build_chat_client", return_value=MagicMock()),
        patch(
            "agentic_concierge.interfaces.cli.execute_task",
            new_callable=AsyncMock,
            return_value=mock_result,
        ),
    ):
        result = runner.invoke(app, ["run", "--no-stream", "write hello world"])
    assert result.exit_code == 0
    assert "llama3.1:8b" in result.output


def test_run_shows_fallback_warnings():
    """When resolve_llm returns warnings, they must appear in CLI output."""
    mock_result = _mock_run_result()
    resolved = _mock_resolved(warnings=["Fallback: primary vLLM unreachable, using Ollama"])
    with (
        patch("agentic_concierge.interfaces.cli.resolve_llm", return_value=resolved),
        patch("agentic_concierge.interfaces.cli.build_chat_client", return_value=MagicMock()),
        patch(
            "agentic_concierge.interfaces.cli.execute_task",
            new_callable=AsyncMock,
            return_value=mock_result,
        ),
    ):
        result = runner.invoke(app, ["run", "--no-stream", "write hello"])
    assert result.exit_code == 0
    assert "Warning" in result.output
    assert "Fallback" in result.output


# ---------------------------------------------------------------------------
# concierge serve
# ---------------------------------------------------------------------------

def test_serve_help():
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0


def test_serve_calls_uvicorn():
    """serve() must delegate to uvicorn.run with the requested host/port."""
    with patch("uvicorn.run") as mock_uvicorn:
        runner.invoke(app, ["serve", "--host", "127.0.0.1", "--port", "9876"])
    mock_uvicorn.assert_called_once()
    call_repr = str(mock_uvicorn.call_args)
    assert "127.0.0.1" in call_repr
    assert "9876" in call_repr


def test_serve_default_port():
    """Default port is 8787."""
    with patch("uvicorn.run") as mock_uvicorn:
        runner.invoke(app, ["serve"])
    mock_uvicorn.assert_called_once()
    assert "8787" in str(mock_uvicorn.call_args)


# ---------------------------------------------------------------------------
# concierge plan
# ---------------------------------------------------------------------------

def test_plan_help():
    result = runner.invoke(app, ["plan", "--help"])
    assert result.exit_code == 0


def test_plan_exits_on_no_llm():
    with patch(
        "agentic_concierge.interfaces.cli.resolve_llm",
        side_effect=RuntimeError("No LLM"),
    ):
        result = runner.invoke(app, ["plan", "design a REST API"])
    assert result.exit_code != 0


def test_plan_success_path():
    """plan_cmd must print mode and specialist assignments from the plan."""
    from agentic_concierge.application.orchestrator import OrchestrationPlan, SpecialistBrief

    mock_plan = OrchestrationPlan(
        mode="sequential",
        specialist_assignments=[
            SpecialistBrief(specialist_id="engineering", brief="Implement the API"),
        ],
        synthesis_required=False,
        reasoning="Simple engineering task",
        routing_method="orchestrator",
    )
    with (
        patch("agentic_concierge.interfaces.cli.resolve_llm", return_value=_mock_resolved()),
        patch("agentic_concierge.interfaces.cli.build_chat_client", return_value=MagicMock()),
        patch(
            "agentic_concierge.application.orchestrator.orchestrate_task",
            new_callable=AsyncMock,
            return_value=mock_plan,
        ),
    ):
        result = runner.invoke(app, ["plan", "design a REST API"])
    assert result.exit_code == 0
    assert "sequential" in result.output
    assert "engineering" in result.output


def test_plan_shows_synthesis_flag():
    """synthesis_required=True must be reflected in the output."""
    from agentic_concierge.application.orchestrator import OrchestrationPlan, SpecialistBrief

    mock_plan = OrchestrationPlan(
        mode="parallel",
        specialist_assignments=[
            SpecialistBrief(specialist_id="research", brief="Research the topic"),
            SpecialistBrief(specialist_id="engineering", brief="Implement the finding"),
        ],
        synthesis_required=True,
        reasoning="Multi-specialist parallel task",
        routing_method="orchestrator",
    )
    with (
        patch("agentic_concierge.interfaces.cli.resolve_llm", return_value=_mock_resolved()),
        patch("agentic_concierge.interfaces.cli.build_chat_client", return_value=MagicMock()),
        patch(
            "agentic_concierge.application.orchestrator.orchestrate_task",
            new_callable=AsyncMock,
            return_value=mock_plan,
        ),
    ):
        result = runner.invoke(app, ["plan", "research and implement a feature"])
    assert result.exit_code == 0
    assert "yes" in result.output   # synthesis_required → "yes"
    assert "parallel" in result.output


# ---------------------------------------------------------------------------
# concierge resume
# ---------------------------------------------------------------------------

def test_resume_help():
    result = runner.invoke(app, ["resume", "--help"])
    assert result.exit_code == 0


def test_resume_no_checkpoint(tmp_path):
    """A missing checkpoint must exit with a non-zero code and an error message."""
    with patch(
        "agentic_concierge.infrastructure.workspace.run_checkpoint.load_checkpoint",
        return_value=None,
    ):
        result = runner.invoke(app, ["resume", "no-such-run", "--workspace", str(tmp_path)])
    assert result.exit_code != 0


def test_resume_all_complete(tmp_path):
    """When all specialists are done the CLI must report success (exit 0)."""
    checkpoint = _make_checkpoint(tmp_path, "run-done", completed=["engineering"])
    with patch(
        "agentic_concierge.infrastructure.workspace.run_checkpoint.load_checkpoint",
        return_value=checkpoint,
    ):
        result = runner.invoke(app, ["resume", "run-done", "--workspace", str(tmp_path)])
    assert result.exit_code == 0
    assert "already complete" in result.output.lower()


def test_resume_prints_remaining_specialist(tmp_path):
    """Before executing, resume must print the specialist it is continuing from."""
    checkpoint = _make_checkpoint(tmp_path, "run-partial", completed=[])
    mock_result = _mock_run_result()
    with (
        patch(
            "agentic_concierge.infrastructure.workspace.run_checkpoint.load_checkpoint",
            return_value=checkpoint,
        ),
        patch("agentic_concierge.interfaces.cli.resolve_llm", return_value=_mock_resolved()),
        patch("agentic_concierge.interfaces.cli.build_chat_client", return_value=MagicMock()),
        patch(
            "agentic_concierge.application.execute_task.resume_execute_task",
            new_callable=AsyncMock,
            return_value=mock_result,
        ),
    ):
        result = runner.invoke(app, ["resume", "run-partial", "--workspace", str(tmp_path)])
    assert result.exit_code == 0
    assert "engineering" in result.output


def test_resume_exits_on_no_llm(tmp_path):
    """If resolve_llm fails during resume, the CLI must exit non-zero."""
    checkpoint = _make_checkpoint(tmp_path, "run-partial", completed=[])
    with (
        patch(
            "agentic_concierge.infrastructure.workspace.run_checkpoint.load_checkpoint",
            return_value=checkpoint,
        ),
        patch(
            "agentic_concierge.interfaces.cli.resolve_llm",
            side_effect=RuntimeError("No LLM"),
        ),
    ):
        result = runner.invoke(app, ["resume", "run-partial", "--workspace", str(tmp_path)])
    assert result.exit_code != 0
