"""Tests for independent reviewer model selection (ADR-027)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agentic_concierge.application.execute_task import (
    _review_specialist_work,
)
from agentic_concierge.config.schema import ModelConfig
from agentic_concierge.domain import LLMResponse, RunId, ToolCallRequest
from agentic_concierge.infrastructure.ollama import OllamaChatClient
from agentic_concierge.infrastructure.workspace import FileSystemRunRepository


def _approve_response(call_id="r1", comment="OK"):
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(call_id=call_id, tool_name="approve_work", arguments={"comment": comment})],
    )


class _StubPack:
    """Minimal pack stub for reviewer tests."""
    @property
    def tool_definitions(self):
        return [
            {"function": {"name": "read_file", "parameters": {"type": "object", "properties": {}}}},
            {"function": {"name": "list_files", "parameters": {"type": "object", "properties": {}}}},
            {"function": {"name": "finish_task", "parameters": {"type": "object", "properties": {}}}},
        ]

    async def execute_tool(self, name, args):
        return {"result": "ok"}


@pytest.mark.asyncio
async def test_reviewer_selects_different_model_when_available(tmp_path):
    """When available_models has a reasoning-strong model, reviewer should pick it."""
    run_repo = FileSystemRunRepository(workspace_root=str(tmp_path))
    run_id, run_dir, workspace = run_repo.create_run()

    # Track which model was used in chat calls
    models_used = []

    async def _mock_chat(messages, model, **kwargs):
        models_used.append(model)
        return _approve_response()

    mock_client = AsyncMock()
    mock_client.chat = _mock_chat

    # Create a mock that returns itself (avoids real HTTP connections)
    rebuilt_client = AsyncMock()
    rebuilt_client.chat = _mock_chat

    model_cfg = ModelConfig(
        base_url="http://localhost:11434/v1", model="qwen2.5:7b",
    )

    with patch(
        "agentic_concierge.application.execute_task._rebuild_chat_client",
        return_value=rebuilt_client,
    ):
        approved, comment = await _review_specialist_work(
            finish_payload={"summary": "Done", "action": "final"},
            pack=_StubPack(),
            model_cfg=model_cfg,
            chat_client=mock_client,
            run_repository=run_repo,
            run_id=RunId(value=run_id.value),
            specialist_id="engineering",
            event_queue=None,
            review_iteration=0,
            available_models=["qwen2.5:7b", "phi-4-mini:3.8b"],
        )
    assert approved is True
    # phi-4-mini has stronger reasoning (0.85) and should be picked over qwen2.5 (0.7)
    assert models_used[0] == "phi-4-mini:3.8b"


@pytest.mark.asyncio
async def test_reviewer_falls_back_to_doer_model_when_no_alternative(tmp_path):
    """When only one model is available, reviewer uses the same model."""
    run_repo = FileSystemRunRepository(workspace_root=str(tmp_path))
    run_id, run_dir, workspace = run_repo.create_run()

    models_used = []

    async def _mock_chat(messages, model, **kwargs):
        models_used.append(model)
        return _approve_response()

    mock_client = AsyncMock()
    mock_client.chat = _mock_chat

    model_cfg = ModelConfig(
        base_url="http://localhost:11434/v1", model="qwen2.5:7b",
    )

    approved, comment = await _review_specialist_work(
        finish_payload={"summary": "Done", "action": "final"},
        pack=_StubPack(),
        model_cfg=model_cfg,
        chat_client=mock_client,
        run_repository=run_repo,
        run_id=RunId(value=run_id.value),
        specialist_id="engineering",
        event_queue=None,
        review_iteration=0,
        available_models=["qwen2.5:7b"],
    )
    assert approved is True
    assert models_used[0] == "qwen2.5:7b"


@pytest.mark.asyncio
async def test_reviewer_falls_back_when_no_available_models(tmp_path):
    """When available_models is empty, reviewer uses the doer model."""
    run_repo = FileSystemRunRepository(workspace_root=str(tmp_path))
    run_id, run_dir, workspace = run_repo.create_run()

    models_used = []

    async def _mock_chat(messages, model, **kwargs):
        models_used.append(model)
        return _approve_response()

    mock_client = AsyncMock()
    mock_client.chat = _mock_chat

    model_cfg = ModelConfig(
        base_url="http://localhost:11434/v1", model="qwen2.5:7b",
    )

    approved, comment = await _review_specialist_work(
        finish_payload={"summary": "Done", "action": "final"},
        pack=_StubPack(),
        model_cfg=model_cfg,
        chat_client=mock_client,
        run_repository=run_repo,
        run_id=RunId(value=run_id.value),
        specialist_id="engineering",
        event_queue=None,
        review_iteration=0,
        available_models=[],
    )
    assert approved is True
    assert models_used[0] == "qwen2.5:7b"


@pytest.mark.asyncio
async def test_review_start_event_includes_reviewer_model(tmp_path):
    """review_start event should include the reviewer_model field."""
    run_repo = FileSystemRunRepository(workspace_root=str(tmp_path))
    run_id, run_dir, workspace = run_repo.create_run()

    async def _mock_chat(messages, model, **kwargs):
        return _approve_response()

    mock_client = AsyncMock()
    mock_client.chat = _mock_chat

    rebuilt_client = AsyncMock()
    rebuilt_client.chat = _mock_chat

    model_cfg = ModelConfig(
        base_url="http://localhost:11434/v1", model="qwen2.5:7b",
    )

    with patch(
        "agentic_concierge.application.execute_task._rebuild_chat_client",
        return_value=rebuilt_client,
    ):
        await _review_specialist_work(
            finish_payload={"summary": "Done", "action": "final"},
            pack=_StubPack(),
            model_cfg=model_cfg,
            chat_client=mock_client,
            run_repository=run_repo,
            run_id=RunId(value=run_id.value),
            specialist_id="engineering",
            event_queue=None,
            review_iteration=0,
            available_models=["qwen2.5:7b", "phi-4-mini:3.8b"],
        )

    # Check runlog events
    runlog = Path(run_dir, "runlog.jsonl").read_text().strip().splitlines()
    events = [json.loads(ln) for ln in runlog]
    review_starts = [e for e in events if e["kind"] == "review_start"]
    assert len(review_starts) == 1
    assert "reviewer_model" in review_starts[0]["payload"]
    assert review_starts[0]["payload"]["reviewer_model"] == "phi-4-mini:3.8b"
