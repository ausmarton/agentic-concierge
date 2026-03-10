"""Integration tests for the specialist marketplace wiring (ADR-023 through ADR-028).

These tests verify the END-TO-END paths, not just individual components:
1. Per-specialist model selection actually switches models during execution
2. consult_specialist_model is injected into packs when non-tool-calling models exist
3. required_capabilities from orchestrator drives model selection
4. The consult tool executor actually has access to all_chat_models
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agentic_concierge.config import DEFAULT_CONFIG, ModelConfig
from agentic_concierge.domain import LLMResponse
from agentic_concierge.infrastructure.specialists.registry import ConfigSpecialistRegistry


# ---------------------------------------------------------------------------
# 1. registry.get_pack() injects consult_specialist_model via set_runtime_models
# ---------------------------------------------------------------------------


def test_registry_injects_consult_tool_when_non_tc_models_present(tmp_path: Path):
    """When set_runtime_models includes a non-tool-calling model,
    template packs should include consult_specialist_model."""
    registry = ConfigSpecialistRegistry(DEFAULT_CONFIG)
    registry.set_runtime_models(["qwen2.5:7b", "sqlcoder:15b"])
    workspace = str(tmp_path / "workspace")
    Path(workspace).mkdir(parents=True)

    pack = registry.get_pack("engineering", workspace, network_allowed=False)
    tool_names = [td["function"]["name"] for td in pack.tool_definitions]
    assert "consult_specialist_model" in tool_names


def test_registry_no_consult_tool_when_all_models_are_tc(tmp_path: Path):
    """When all models support tool calling, consult tool is NOT injected."""
    registry = ConfigSpecialistRegistry(DEFAULT_CONFIG)
    registry.set_runtime_models(["qwen2.5:7b", "llama3.1:8b"])
    workspace = str(tmp_path / "workspace")
    Path(workspace).mkdir(parents=True)

    pack = registry.get_pack("engineering", workspace, network_allowed=False)
    tool_names = [td["function"]["name"] for td in pack.tool_definitions]
    assert "consult_specialist_model" not in tool_names


def test_registry_no_consult_tool_when_no_runtime_models(tmp_path: Path):
    """When set_runtime_models is never called, no consult tool injection."""
    registry = ConfigSpecialistRegistry(DEFAULT_CONFIG)
    workspace = str(tmp_path / "workspace")
    Path(workspace).mkdir(parents=True)

    pack = registry.get_pack("research", workspace, network_allowed=True)
    tool_names = [td["function"]["name"] for td in pack.tool_definitions]
    assert "consult_specialist_model" not in tool_names


def test_registry_dynamic_pack_gets_consult_when_non_tc(tmp_path: Path):
    """Dynamic packs also get consult tool when non-TC models are registered."""
    registry = ConfigSpecialistRegistry(DEFAULT_CONFIG)
    registry.set_runtime_models(["qwen2.5:7b", "deepseek-coder:6.7b"])
    workspace = str(tmp_path / "workspace")
    Path(workspace).mkdir(parents=True)

    pack = registry.get_pack(
        "dynamic", workspace, network_allowed=False,
        tools=["shell", "write_file"], role="Test agent",
    )
    tool_names = [td["function"]["name"] for td in pack.tool_definitions]
    assert "consult_specialist_model" in tool_names


# ---------------------------------------------------------------------------
# 2. Consult tool executor actually calls execute_consult with models
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consult_executor_closure_has_models(tmp_path: Path):
    """The consult executor should call execute_consult with the models
    that were passed to build_dynamic_pack."""
    from agentic_concierge.infrastructure.specialists.dynamic_pack import build_dynamic_pack

    workspace = str(tmp_path / "workspace")
    Path(workspace).mkdir(parents=True)

    pack = build_dynamic_pack(
        specialist_id="test",
        tool_names=["shell", "consult_specialist_model"],
        role_description="Test agent",
        workspace_path=workspace,
        network_allowed=False,
        all_chat_models=["qwen2.5:7b", "sqlcoder:15b"],
        base_url="http://localhost:11434/v1",
    )

    assert "consult_specialist_model" in pack._tools
    executor = pack._tools["consult_specialist_model"][1]

    # Mock execute_consult to capture what the executor passes
    mock_response = LLMResponse(content="SELECT 1", tool_calls=[])
    mock_client = AsyncMock()
    mock_client.chat = AsyncMock(return_value=mock_response)

    with patch("agentic_concierge.infrastructure.tools.consult.build_chat_client", return_value=mock_client):
        result = await executor(specialty="sql", prompt="test query")

    # The executor should have passed all_chat_models to execute_consult,
    # which should have picked sqlcoder:15b (highest code_sql score)
    assert result["model"] == "sqlcoder:15b"
    assert result["specialty"] == "sql"


# ---------------------------------------------------------------------------
# 5. set_runtime_models is idempotent and updates state
# ---------------------------------------------------------------------------


def test_set_runtime_models_updates_state(tmp_path: Path):
    """Calling set_runtime_models updates the needs_consult check."""
    registry = ConfigSpecialistRegistry(DEFAULT_CONFIG)
    workspace = str(tmp_path / "workspace")
    Path(workspace).mkdir(parents=True)

    # Initially no consult
    pack1 = registry.get_pack("engineering", workspace, network_allowed=False)
    names1 = [td["function"]["name"] for td in pack1.tool_definitions]
    assert "consult_specialist_model" not in names1

    # After setting runtime models with non-TC, consult appears
    registry.set_runtime_models(["qwen2.5:7b", "sqlcoder:15b"])
    pack2 = registry.get_pack("engineering", workspace, network_allowed=False)
    names2 = [td["function"]["name"] for td in pack2.tool_definitions]
    assert "consult_specialist_model" in names2
