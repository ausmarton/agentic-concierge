"""Tests for per-specialist model selection (ADR-024)."""
from __future__ import annotations

import pytest

from agentic_concierge.infrastructure.model_profiles import (
    SPECIALTY_CAPABILITIES,
    infer_task_capabilities,
    match_models,
)


# ---------------------------------------------------------------------------
# infer_task_capabilities
# ---------------------------------------------------------------------------

def test_infer_from_engineering_template():
    caps = infer_task_capabilities(template_id="engineering")
    assert caps["code_python"] >= 0.5
    assert caps["structured_output"] >= 0.5


def test_infer_from_research_template():
    caps = infer_task_capabilities(template_id="research")
    assert caps["web_comprehension"] >= 0.5
    assert caps["summarisation"] >= 0.5


def test_infer_from_tool_names():
    caps = infer_task_capabilities(tool_names=["shell", "run_tests", "web_search"])
    assert "code_python" in caps
    assert "web_comprehension" in caps


def test_infer_unknown_template():
    caps = infer_task_capabilities(template_id="nonexistent")
    assert caps == {}


def test_infer_empty():
    caps = infer_task_capabilities()
    assert caps == {}


# ---------------------------------------------------------------------------
# Per-specialist model selection via match_models
# ---------------------------------------------------------------------------

def test_engineering_prefers_coder():
    """Engineering tasks should prefer qwen2.5-coder over plain qwen2.5."""
    caps = infer_task_capabilities(template_id="engineering")
    result = match_models(
        ["qwen2.5:7b", "qwen2.5-coder:7b"],
        required_capabilities=caps,
    )
    assert result == "qwen2.5-coder:7b"


def test_research_prefers_generalist():
    """Research tasks should prefer models with web_comprehension."""
    caps = infer_task_capabilities(template_id="research")
    # qwen2.5 has higher web_comprehension (0.6) than qwen2.5-coder (0.3)
    result = match_models(
        ["qwen2.5-coder:7b", "qwen2.5:7b"],
        required_capabilities=caps,
    )
    assert result == "qwen2.5:7b"


def test_single_model_always_selected():
    """When only one model is available, it's always selected regardless of caps."""
    caps = infer_task_capabilities(template_id="engineering")
    result = match_models(
        ["llama3.1:8b"],
        required_capabilities=caps,
    )
    assert result == "llama3.1:8b"


# ---------------------------------------------------------------------------
# SPECIALTY_CAPABILITIES
# ---------------------------------------------------------------------------

def test_specialty_capabilities_defined():
    assert "code" in SPECIALTY_CAPABILITIES
    assert "sql" in SPECIALTY_CAPABILITIES
    assert "reasoning" in SPECIALTY_CAPABILITIES


def test_sql_specialty_prefers_sqlcoder():
    result = match_models(
        ["qwen2.5:7b", "sqlcoder:15b"],
        required_capabilities=SPECIALTY_CAPABILITIES["sql"],
        require_tool_calling=False,
    )
    assert result == "sqlcoder:15b"


# ---------------------------------------------------------------------------
# Issue #2: Defense-in-depth — unknown caps in _select_specialist_model
# ---------------------------------------------------------------------------


def test_select_specialist_unknown_caps_returns_base():
    """Hallucinated caps → model selection returns base model unchanged."""
    from unittest.mock import MagicMock
    from agentic_concierge.application.execute_task import _select_specialist_model
    from agentic_concierge.application.orchestrator import SpecialistBrief
    from agentic_concierge.config.schema import ModelConfig

    base_cfg = ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5:7b")
    mock_client = MagicMock()
    assignment = SpecialistBrief(
        specialist_id="research",
        brief="lookup",
        required_capabilities=["fabricated_cap", "nonsense"],
    )
    result_cfg, result_client = _select_specialist_model(
        assignment, ["qwen2.5:7b", "qwen2.5:32b"], base_cfg, mock_client,
    )
    assert result_cfg.model == "qwen2.5:7b"  # unchanged


def test_select_specialist_mixed_caps_uses_valid_only():
    """Mix of valid and invalid caps → only valid caps used for scoring."""
    from unittest.mock import MagicMock
    from agentic_concierge.application.execute_task import _select_specialist_model
    from agentic_concierge.application.orchestrator import SpecialistBrief
    from agentic_concierge.config.schema import ModelConfig

    base_cfg = ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5:7b")
    mock_client = MagicMock()
    assignment = SpecialistBrief(
        specialist_id="research",
        brief="code task",
        required_capabilities=["code_python", "fabricated"],
    )
    result_cfg, _ = _select_specialist_model(
        assignment, ["qwen2.5:7b", "qwen2.5-coder:14b"], base_cfg, mock_client,
    )
    # Should select coder variant based on valid code_python cap
    assert result_cfg.model == "qwen2.5-coder:14b"
