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
# 2. _select_specialist_model picks the right model
# ---------------------------------------------------------------------------


def test_select_specialist_model_picks_coder_for_engineering():
    """Engineering assignment should select qwen2.5-coder over plain qwen2.5."""
    from agentic_concierge.application.execute_task import _select_specialist_model
    from agentic_concierge.application.orchestrator import SpecialistBrief

    assignment = SpecialistBrief(specialist_id="engineering", brief="build it")
    available = ["qwen2.5:7b", "qwen2.5-coder:7b"]
    base_cfg = ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5:7b")

    with patch("agentic_concierge.application.execute_task._rebuild_chat_client") as mock_rebuild:
        mock_rebuild.return_value = AsyncMock()
        new_cfg, _ = _select_specialist_model(
            assignment, available, base_cfg, AsyncMock(),
        )

    assert new_cfg.model == "qwen2.5-coder:7b"
    mock_rebuild.assert_called_once()


def test_select_specialist_model_picks_generalist_for_research():
    """Research assignment should prefer qwen2.5 (web_comprehension) over coder."""
    from agentic_concierge.application.execute_task import _select_specialist_model
    from agentic_concierge.application.orchestrator import SpecialistBrief

    assignment = SpecialistBrief(specialist_id="research", brief="look it up")
    available = ["qwen2.5-coder:7b", "qwen2.5:7b"]
    base_cfg = ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5-coder:7b")

    with patch("agentic_concierge.application.execute_task._rebuild_chat_client") as mock_rebuild:
        mock_rebuild.return_value = AsyncMock()
        new_cfg, _ = _select_specialist_model(
            assignment, available, base_cfg, AsyncMock(),
        )

    assert new_cfg.model == "qwen2.5:7b"


def test_select_specialist_model_noop_single_model():
    """With only one model, returns originals unchanged — no rebuild."""
    from agentic_concierge.application.execute_task import _select_specialist_model
    from agentic_concierge.application.orchestrator import SpecialistBrief

    assignment = SpecialistBrief(specialist_id="engineering", brief="build it")
    base_cfg = ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5:7b")
    mock_client = AsyncMock()

    new_cfg, new_client = _select_specialist_model(
        assignment, ["qwen2.5:7b"], base_cfg, mock_client,
    )
    assert new_cfg is base_cfg
    assert new_client is mock_client


def test_select_specialist_model_noop_when_already_best():
    """When base model is already the best match, returns originals unchanged."""
    from agentic_concierge.application.execute_task import _select_specialist_model
    from agentic_concierge.application.orchestrator import SpecialistBrief

    assignment = SpecialistBrief(specialist_id="engineering", brief="build it")
    base_cfg = ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5-coder:7b")
    mock_client = AsyncMock()

    new_cfg, new_client = _select_specialist_model(
        assignment, ["qwen2.5-coder:7b", "qwen2.5:7b"], base_cfg, mock_client,
    )
    assert new_cfg is base_cfg
    assert new_client is mock_client


# ---------------------------------------------------------------------------
# 3. required_capabilities from orchestrator drives model selection
# ---------------------------------------------------------------------------


def test_select_specialist_model_uses_explicit_capabilities():
    """When required_capabilities is set, it overrides template inference."""
    from agentic_concierge.application.execute_task import _select_specialist_model
    from agentic_concierge.application.orchestrator import SpecialistBrief

    # Template is research, but explicit caps say code_python
    assignment = SpecialistBrief(
        specialist_id="research", brief="task",
        required_capabilities=["code_python"],
    )
    available = ["qwen2.5:7b", "qwen2.5-coder:7b"]
    base_cfg = ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5:7b")

    with patch("agentic_concierge.application.execute_task._rebuild_chat_client") as mock_rebuild:
        mock_rebuild.return_value = AsyncMock()
        new_cfg, _ = _select_specialist_model(
            assignment, available, base_cfg, AsyncMock(),
        )

    # Should pick coder because explicit caps say code_python,
    # even though template is "research"
    assert new_cfg.model == "qwen2.5-coder:7b"


# ---------------------------------------------------------------------------
# 4. Consult tool executor actually calls execute_consult with models
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
