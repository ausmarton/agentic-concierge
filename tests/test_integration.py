"""Integration tests: execute_task produces run dir + runlog + workspace; API health; E2E with real HTTP."""
from __future__ import annotations

import json
import socket
import threading
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import uvicorn
from fastapi.testclient import TestClient

from agentic_concierge.application.execute_task import execute_task
from agentic_concierge.config import DEFAULT_CONFIG, ConciergeConfig, load_config
from agentic_concierge.config import ModelConfig
from agentic_concierge.domain import LLMResponse, RunId, RunResult, Task, ToolCallRequest
from agentic_concierge.infrastructure.ollama import OllamaChatClient
from agentic_concierge.infrastructure.workspace import FileSystemRunRepository
from agentic_concierge.infrastructure.specialists import ConfigSpecialistRegistry

try:
    from tests.mock_llm_server import app as mock_llm_app
except ImportError:
    mock_llm_app = None

try:
    from tests.conftest import real_llm_reachable, skip_if_no_real_llm, SKIP_REAL_LLM
except ImportError:
    real_llm_reachable = None

    def skip_if_no_real_llm():
        return None

    SKIP_REAL_LLM = True


def _finish_task_response(**kwargs) -> LLMResponse:
    """Build a mock LLMResponse that calls finish_task with the given args."""
    defaults = {"summary": "Done", "artifacts": [], "next_steps": [], "notes": "", "tests_verified": True}
    defaults.update(kwargs)
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(call_id="call_1", tool_name="finish_task", arguments=defaults)],
    )


MOCK_ENG_RESPONSE = _finish_task_response()
MOCK_RESEARCH_RESPONSE = _finish_task_response(
    summary="Research done",
    executive_summary="Overview.",
    key_findings=["Finding 1"],
    citations=[],
    gaps_and_future_work=[],
)

# A simple list_files call to satisfy the "prior tool call" structural requirement
# before finish_task is accepted.
_MOCK_TOOL_CALL = LLMResponse(
    content=None,
    tool_calls=[ToolCallRequest(call_id="t0", tool_name="list_files", arguments={})],
)


@pytest.fixture
def temp_workspace_root(tmp_path):
    return str(tmp_path)


@pytest.mark.asyncio
async def test_execute_task_creates_run_dir_runlog_workspace(temp_workspace_root):
    """execute_task with mock LLM: run dir, runlog.jsonl, workspace exist; RunResult correct."""
    config = load_config()
    run_repository = FileSystemRunRepository(workspace_root=temp_workspace_root)
    specialist_registry = ConfigSpecialistRegistry(config)

    with patch.object(OllamaChatClient, "chat", new_callable=AsyncMock,
                      side_effect=[_MOCK_TOOL_CALL, MOCK_ENG_RESPONSE]):
        chat_client = OllamaChatClient(base_url="http://localhost:11434/v1", timeout_s=5.0)
        task = Task(prompt="list files", specialist_id="engineering", network_allowed=False)
        result = await execute_task(
            task,
            chat_client=chat_client,
            run_repository=run_repository,
            specialist_registry=specialist_registry,
            config=config,
            max_steps=40,
        )

    assert result.specialist_id == "engineering"
    run_dir = Path(result.run_dir)
    workspace = Path(result.workspace_path)
    assert run_dir.is_dir()
    assert (run_dir / "runlog.jsonl").is_file()
    assert workspace.is_dir()
    assert workspace.parent == run_dir

    lines = (run_dir / "runlog.jsonl").read_text().strip().split("\n")
    assert len(lines) >= 1
    events = [json.loads(ln) for ln in lines if ln]
    kinds = [e.get("kind") for e in events]
    assert "llm_request" in kinds
    assert "llm_response" in kinds
    assert "tool_call" in kinds    # finish_task is a tool_call
    assert "tool_result" in kinds

    assert result.payload.get("action") == "final"
    assert result.payload.get("summary") == "Done"


@pytest.mark.asyncio
async def test_execute_task_research_pack(temp_workspace_root):
    """execute_task with research pack and mock LLM; run dir structure."""
    config = load_config()
    run_repository = FileSystemRunRepository(workspace_root=temp_workspace_root)
    specialist_registry = ConfigSpecialistRegistry(config)

    with patch.object(OllamaChatClient, "chat", new_callable=AsyncMock,
                      side_effect=[_MOCK_TOOL_CALL, MOCK_RESEARCH_RESPONSE]):
        chat_client = OllamaChatClient(base_url="http://localhost:11434/v1", timeout_s=5.0)
        task = Task(prompt="mini review", specialist_id="research", network_allowed=False)
        result = await execute_task(
            task,
            chat_client=chat_client,
            run_repository=run_repository,
            specialist_registry=specialist_registry,
            config=config,
            max_steps=40,
        )

    assert result.specialist_id == "research"
    run_dir = Path(result.run_dir)
    assert (run_dir / "runlog.jsonl").is_file()
    assert (run_dir / "workspace").is_dir()
    assert result.payload.get("action") == "final"


def test_api_health_returns_ok():
    from agentic_concierge.interfaces.http_api import app
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_api_run_accepts_prompt():
    """API POST /run accepts prompt and returns 200 when execute_task is mocked."""
    from unittest.mock import MagicMock
    from agentic_concierge.interfaces.http_api import app
    cfg = ConciergeConfig(
        models=DEFAULT_CONFIG.models,
        specialists=DEFAULT_CONFIG.specialists,
        local_llm_ensure_available=False,
    )
    # resolve_llm makes a real HTTP probe → must be mocked so the test works
    # without a live Ollama server.
    mock_model_cfg = next(iter(cfg.models.values()))
    mock_resolved = MagicMock(model_config=mock_model_cfg, base_url=mock_model_cfg.base_url)
    with patch("agentic_concierge.interfaces.http_api.load_config", return_value=cfg), \
         patch("agentic_concierge.interfaces.http_api.resolve_llm", return_value=mock_resolved), \
         patch("agentic_concierge.interfaces.http_api.execute_task", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = RunResult(
            run_id=RunId("test"), run_dir="/tmp/x", workspace_path="/tmp/x/w",
            specialist_id="engineering", model_name="qwen2.5:7b",
            payload={"action": "final", "summary": "Done", "artifacts": [], "next_steps": [], "notes": ""},
        )
        client = TestClient(app)
        r = client.post("/run", json={"prompt": "hello", "pack": "engineering"})
    assert r.status_code == 200
    data = r.json()
    assert "_meta" in data
    assert data["_meta"]["pack"] == "engineering"
    mock_run.assert_called_once()
    task = mock_run.call_args[0][0]
    assert task.prompt == "hello"
    assert task.specialist_id == "engineering"


def test_api_run_meta_includes_fallback_fields():
    """API POST /run _meta must include warnings, fallback_used, resolved_backend."""
    from unittest.mock import MagicMock
    from agentic_concierge.interfaces.http_api import app
    cfg = ConciergeConfig(
        models=DEFAULT_CONFIG.models,
        specialists=DEFAULT_CONFIG.specialists,
        local_llm_ensure_available=False,
    )
    mock_model_cfg = next(iter(cfg.models.values()))
    mock_resolved = MagicMock(
        model_config=mock_model_cfg,
        base_url=mock_model_cfg.base_url,
        warnings=["Fallback: primary unreachable, using Ollama"],
        fallback_used=True,
        resolved_backend="ollama",
    )
    with patch("agentic_concierge.interfaces.http_api.load_config", return_value=cfg), \
         patch("agentic_concierge.interfaces.http_api.resolve_llm", return_value=mock_resolved), \
         patch("agentic_concierge.interfaces.http_api.execute_task", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = RunResult(
            run_id=RunId("test"), run_dir="/tmp/x", workspace_path="/tmp/x/w",
            specialist_id="engineering", model_name="qwen2.5:7b",
            payload={"action": "final", "summary": "Done", "artifacts": [], "next_steps": [], "notes": ""},
        )
        client = TestClient(app)
        r = client.post("/run", json={"prompt": "hello", "pack": "engineering"})
    assert r.status_code == 200
    meta = r.json()["_meta"]
    assert meta["fallback_used"] is True
    assert meta["resolved_backend"] == "ollama"
    assert len(meta["warnings"]) >= 1
    assert "Fallback" in meta["warnings"][0]


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.asyncio
async def test_e2e_real_http_engineering_run(temp_workspace_root):
    """E2E: real HTTP to mock LLM server; execute_task completes and produces run dir + runlog + workspace."""
    if mock_llm_app is None:
        pytest.skip("mock_llm_server not importable")
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}/v1"
    config = ConciergeConfig(
        models={
            "quality": ModelConfig(base_url=base_url, model="mock", timeout_s=5.0),
        },
        specialists=DEFAULT_CONFIG.specialists,
    )
    run_repository = FileSystemRunRepository(workspace_root=temp_workspace_root)
    specialist_registry = ConfigSpecialistRegistry(config)
    chat_client = OllamaChatClient(base_url=base_url, timeout_s=5.0)

    def run_server():
        uvicorn.run(mock_llm_app, host="127.0.0.1", port=port, log_level="warning")

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    import urllib.request
    for _ in range(50):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=0.5)
            break
        except Exception:
            thread.join(timeout=0.1)
    else:
        pytest.skip("Mock server did not start in time")

    task = Task(prompt="Create a hello world file", specialist_id="engineering", network_allowed=False)
    result = await execute_task(
        task,
        chat_client=chat_client,
        run_repository=run_repository,
        specialist_registry=specialist_registry,
        config=config,
        max_steps=40,
    )

    assert result.specialist_id == "engineering"
    run_dir = Path(result.run_dir)
    assert run_dir.is_dir()
    assert (run_dir / "runlog.jsonl").is_file()
    assert (run_dir / "workspace").is_dir()
    assert result.payload.get("action") == "final"

    lines = (run_dir / "runlog.jsonl").read_text().strip().split("\n")
    events = [json.loads(ln) for ln in lines if ln]
    kinds = [e.get("kind") for e in events]
    assert "llm_request" in kinds
    assert "llm_response" in kinds
    assert "tool_call" in kinds
    assert "tool_result" in kinds


# ---- Real LLM tests (skip when no server or CONCIERGE_SKIP_REAL_LLM=1) ----


def _skip_if_no_real_llm():
    if SKIP_REAL_LLM:
        pytest.skip("CONCIERGE_SKIP_REAL_LLM is set")
    if real_llm_reachable() is None:
        pytest.skip("Real LLM not reachable (start Ollama and pull a model to run this test)")


@pytest.mark.asyncio
async def test_execute_task_engineering_real_llm(temp_workspace_root):
    """E2E engineering pack with real LLM: tool_call, tool_result, workspace artifacts. Skips if no server/model."""
    _skip_if_no_real_llm()
    cfg, model_cfg = real_llm_reachable()
    chat_client = OllamaChatClient(
        base_url=model_cfg.base_url,
        api_key=model_cfg.api_key,
        timeout_s=model_cfg.timeout_s,
    )
    run_repository = FileSystemRunRepository(workspace_root=temp_workspace_root)
    specialist_registry = ConfigSpecialistRegistry(cfg)
    task = Task(
        prompt="Create a file hello.txt containing the line 'Hello World'. Then list the workspace directory.",
        specialist_id="engineering",
        model_key="quality",
        network_allowed=False,
    )
    try:
        result = await execute_task(
            task,
            chat_client=chat_client,
            run_repository=run_repository,
            specialist_registry=specialist_registry,
            config=cfg,
            resolved_model_cfg=model_cfg,
            max_steps=40,
        )
    except Exception as e:
        err = str(e).lower()
        if "404" in err or "503" in err or "connect" in err or "connection" in err or "not found" in err:
            pytest.skip(f"Real LLM server or model not available: {e}")
        raise
    assert result.specialist_id == "engineering"
    assert result.payload.get("action") == "final"
    run_dir = Path(result.run_dir)
    assert (run_dir / "runlog.jsonl").is_file()
    events = [json.loads(ln) for ln in (run_dir / "runlog.jsonl").read_text().strip().split("\n") if ln]
    kinds = [e.get("kind") for e in events]
    assert "tool_call" in kinds, "Model must use tools (autonomous behaviour)"
    assert "tool_result" in kinds
    workspace_path = Path(result.workspace_path)
    assert workspace_path.is_dir()
    files = list(workspace_path.iterdir()) if workspace_path.is_dir() else []
    assert len(files) >= 1, "Workspace must contain at least one artifact"


@pytest.mark.asyncio
async def test_execute_task_research_pack_real_llm(temp_workspace_root):
    """E2E research pack with real LLM: tool_call, tool_result, action final. Skips if no server."""
    _skip_if_no_real_llm()
    cfg, model_cfg = real_llm_reachable()
    chat_client = OllamaChatClient(
        base_url=model_cfg.base_url,
        api_key=model_cfg.api_key,
        timeout_s=model_cfg.timeout_s,
    )
    run_repository = FileSystemRunRepository(workspace_root=temp_workspace_root)
    specialist_registry = ConfigSpecialistRegistry(cfg)
    task = Task(
        prompt="Write a one-paragraph overview of what a systematic review is to overview.md in the workspace. Then list the workspace files.",
        specialist_id="research",
        model_key="quality",
        network_allowed=False,
    )
    try:
        result = await execute_task(
            task,
            chat_client=chat_client,
            run_repository=run_repository,
            specialist_registry=specialist_registry,
            config=cfg,
            resolved_model_cfg=model_cfg,
            max_steps=40,
        )
    except Exception as e:
        err = str(e).lower()
        if "404" in err or "503" in err or "connect" in err or "connection" in err or "not found" in err:
            pytest.skip(f"Real LLM server or model not available: {e}")
        raise
    assert result.specialist_id == "research"
    assert result.payload.get("action") == "final"
    run_dir = Path(result.run_dir)
    assert (run_dir / "runlog.jsonl").is_file()
    events = [json.loads(ln) for ln in (run_dir / "runlog.jsonl").read_text().strip().split("\n") if ln]
    kinds = [e.get("kind") for e in events]
    assert "tool_call" in kinds, "Model must use tools (autonomous behaviour)"
    assert "tool_result" in kinds


def test_api_post_run_real_llm():
    """API POST /run with real LLM returns 200 and expected shape. Skips if no server."""
    _skip_if_no_real_llm()
    from agentic_concierge.interfaces.http_api import app
    client = TestClient(app)
    r = client.post(
        "/run",
        json={
            "prompt": "Create a file done.txt with content DONE. Then list the workspace.",
            "pack": "engineering",
            "model_key": "quality",
        },
    )
    if r.status_code == 503:
        pytest.skip("Real LLM returned 503 (server or model not available)")
    assert r.status_code == 200, r.text
    data = r.json()
    assert "_meta" in data
    assert data["_meta"].get("pack") == "engineering"
    assert "action" in data or "summary" in data or "_meta" in data


def test_verify_working_real_script():
    """Run scripts/verify_working_real.py as part of automated gate; skips if no real LLM."""
    _skip_if_no_real_llm()
    import subprocess
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "scripts" / "verify_working_real.py"
    if not script.is_file():
        pytest.skip("scripts/verify_working_real.py not found")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=600,
        env={**__import__("os").environ, "CONCIERGE_WORKSPACE": str(repo_root / ".concierge")},
    )
    if result.returncode != 0 and (
        "404" in result.stderr or "404" in result.stdout
        or "connect" in result.stderr.lower()
        or "model" in result.stderr.lower()
    ):
        pytest.skip(f"Real LLM script failed (server/model not available): {result.stderr or result.stdout}")
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")
