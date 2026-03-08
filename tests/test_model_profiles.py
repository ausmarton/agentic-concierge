"""Tests for the model capability registry (ADR-023)."""
from __future__ import annotations

import pytest

from agentic_concierge.infrastructure.model_profiles import (
    BUILTIN_PROFILES,
    ModelCapabilityProfile,
    _DEFAULT_PROFILE,
    _extract_family,
    _is_vision_model,
    capability_summary,
    get_profile,
    match_models,
)


# ---------------------------------------------------------------------------
# _extract_family
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("qwen2.5:7b", "qwen2.5"),
    ("qwen2.5-coder:14b", "qwen2.5-coder"),
    ("llama3.1:8b", "llama3.1"),
    ("phi-4-mini:latest", "phi-4-mini"),
    ("deepseek-r1-distill-qwen-14b", "deepseek-r1-distill-qwen"),
    ("sqlcoder:15b", "sqlcoder"),
    ("codellama:7b", "codellama"),
    ("gemma2:9b", "gemma2"),
])
def test_extract_family(name, expected):
    assert _extract_family(name) == expected


# ---------------------------------------------------------------------------
# get_profile
# ---------------------------------------------------------------------------

def test_get_profile_known_family():
    p = get_profile("qwen2.5:7b")
    assert p.family == "qwen2.5"
    assert p.supports_tool_calling is True
    assert p.capabilities["code_python"] == 0.7


def test_get_profile_coder_variant():
    """qwen2.5-coder should match the coder profile, not the base qwen2.5."""
    p = get_profile("qwen2.5-coder:14b")
    assert p.family == "qwen2.5-coder"
    assert p.capabilities["code_python"] == 0.95


def test_get_profile_unknown_returns_default():
    p = get_profile("some-unknown-model:7b")
    assert p.family == "unknown"
    assert p.supports_tool_calling is True
    assert p.capabilities["reasoning"] == 0.5


def test_get_profile_user_override():
    custom = ModelCapabilityProfile(
        family="qwen2.5",
        supports_tool_calling=True,
        capabilities={"code_python": 0.99},
    )
    p = get_profile("qwen2.5:7b", overrides={"qwen2.5": custom})
    assert p.capabilities["code_python"] == 0.99


# ---------------------------------------------------------------------------
# _is_tool_capable backward compatibility
# ---------------------------------------------------------------------------

def test_tool_incapable_models_have_profiles():
    """Every model in the old _TOOL_INCAPABLE_NAMES must have a profile with supports_tool_calling=False."""
    from agentic_concierge.infrastructure.llm_discovery import _TOOL_INCAPABLE_NAMES
    for name in _TOOL_INCAPABLE_NAMES:
        profile = get_profile(name)
        assert profile.supports_tool_calling is False, (
            f"Expected {name} to be tool-incapable but profile says tool_calling=True"
        )


def test_is_tool_capable_delegates_to_profiles():
    """_is_tool_capable should produce the same result as the profile system."""
    from agentic_concierge.infrastructure.llm_discovery import _is_tool_capable
    test_models = [
        ({"name": "qwen2.5:7b", "details": {"family": "qwen2.5"}}, True),
        ({"name": "sqlcoder:15b", "details": {"family": "sqlcoder"}}, False),
        ({"name": "codellama:7b", "details": {"family": "codellama"}}, False),
        ({"name": "deepseek-coder:6.7b", "details": {"family": "deepseek-coder"}}, False),
        ({"name": "llama3.1:8b", "details": {"family": "llama"}}, True),
    ]
    for model_dict, expected in test_models:
        assert _is_tool_capable(model_dict) is expected, f"Failed for {model_dict['name']}"


# ---------------------------------------------------------------------------
# match_models
# ---------------------------------------------------------------------------

def test_match_models_empty_list():
    assert match_models([]) is None


def test_match_models_filters_non_tool_calling():
    result = match_models(
        ["qwen2.5:7b", "sqlcoder:15b"],
        require_tool_calling=True,
    )
    assert result == "qwen2.5:7b"


def test_match_models_allows_non_tool_calling():
    result = match_models(
        ["sqlcoder:15b", "qwen2.5:7b"],
        required_capabilities={"code_sql": 0.8},
        require_tool_calling=False,
    )
    assert result == "sqlcoder:15b"


def test_match_models_excludes_specified():
    result = match_models(
        ["qwen2.5:7b", "llama3.1:8b"],
        exclude_models=["qwen2.5:7b"],
    )
    assert result == "llama3.1:8b"


def test_match_models_prefers_capability_match():
    """When requiring code_python >= 0.8, qwen2.5-coder should beat llama3.1."""
    result = match_models(
        ["llama3.1:8b", "qwen2.5-coder:7b"],
        required_capabilities={"code_python": 0.8},
    )
    assert result == "qwen2.5-coder:7b"


def test_match_models_prefer_smaller():
    result = match_models(
        ["qwen2.5:14b", "qwen2.5:7b"],
        prefer_smaller=True,
    )
    assert result == "qwen2.5:7b"


def test_match_models_reasoning_capability():
    """phi-4-mini should rank high for reasoning tasks."""
    result = match_models(
        ["qwen2.5:7b", "phi-4-mini:3.8b", "llama3.1:8b"],
        required_capabilities={"reasoning": 0.8},
    )
    assert result == "phi-4-mini:3.8b"


# ---------------------------------------------------------------------------
# capability_summary
# ---------------------------------------------------------------------------

def test_capability_summary_not_empty():
    s = capability_summary(["qwen2.5:7b", "sqlcoder:15b"])
    assert "qwen2.5" in s
    assert "sqlcoder" in s
    assert "tool-calling" in s
    assert "no-tool-calling" in s


def test_capability_summary_deduplicates_families():
    s = capability_summary(["qwen2.5:7b", "qwen2.5:14b"])
    assert s.count("qwen2.5") == 1


def test_capability_summary_empty():
    assert capability_summary([]) == "No model profiles available."


# ---------------------------------------------------------------------------
# ResolvedLLM.all_chat_models
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Vision model exclusion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("qwen3-vl:32b", True),
    ("llava-vision:7b", True),
    ("minicpm-v:8b", True),
    ("qwen2.5:7b", False),
    ("qwen3:8b", False),
    ("llama3.1:8b", False),
    ("qwen2.5-coder:14b", False),
])
def test_is_vision_model(name, expected):
    assert _is_vision_model(name) is expected


def test_match_models_excludes_vision_from_tool_calling():
    """Vision models should be excluded when require_tool_calling=True."""
    result = match_models(
        ["qwen2.5:7b", "qwen3-vl:32b"],
        required_capabilities={"reasoning": 0.7},
        require_tool_calling=True,
    )
    assert result == "qwen2.5:7b"


def test_match_models_vision_model_not_selected_as_reviewer():
    """Reviewer selection should not pick a vision model."""
    result = match_models(
        ["qwen2.5:7b", "qwen2.5:32b", "qwen3-vl:32b"],
        required_capabilities={"reasoning": 0.7, "instruction_following": 0.6},
        require_tool_calling=True,
        exclude_models=["qwen2.5:7b"],
        prefer_smaller=True,
    )
    assert result == "qwen2.5:32b"


def test_match_models_vision_allowed_when_no_tool_calling_required():
    """Vision models should be allowed when require_tool_calling=False."""
    result = match_models(
        ["qwen3-vl:32b"],
        require_tool_calling=False,
    )
    assert result == "qwen3-vl:32b"


def test_resolved_llm_has_all_chat_models():
    from agentic_concierge.infrastructure.llm_discovery import ResolvedLLM
    from agentic_concierge.config.schema import ModelConfig
    r = ResolvedLLM(
        base_url="http://localhost:11434/v1",
        model="qwen2.5:7b",
        model_config=ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5:7b"),
        available_models=["qwen2.5:7b"],
        all_chat_models=["qwen2.5:7b", "sqlcoder:15b"],
    )
    assert "sqlcoder:15b" in r.all_chat_models
    assert "sqlcoder:15b" not in r.available_models
