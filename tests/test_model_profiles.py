"""Tests for the model capability registry (ADR-023) and tool composition (ADR-028)."""
from __future__ import annotations

import pytest

from agentic_concierge.infrastructure.model_profiles import (
    BUILTIN_PROFILES,
    ModelCapabilityProfile,
    _BASE_TOOLS,
    _CAPABILITY_TOOLS,
    _DEFAULT_PROFILE,
    _extract_family,
    _is_vision_model,
    capability_summary,
    compose_tools_from_capabilities,
    generate_role_from_capabilities,
    get_profile,
    infer_finish_schema_from_capabilities,
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


# ---------------------------------------------------------------------------
# Capability → tool composition (ADR-028)
# ---------------------------------------------------------------------------


class TestComposeToolsFromCapabilities:
    """Tests for compose_tools_from_capabilities()."""

    def test_web_comprehension_includes_web_tools(self):
        tools = compose_tools_from_capabilities(["web_comprehension"])
        assert "web_search" in tools
        assert "fetch_url" in tools

    def test_code_python_includes_code_tools(self):
        tools = compose_tools_from_capabilities(["code_python"])
        assert "shell" in tools
        assert "write_file" in tools
        assert "run_tests" in tools

    def test_always_includes_base_tools(self):
        tools = compose_tools_from_capabilities(["web_comprehension"])
        for base in _BASE_TOOLS:
            assert base in tools

    def test_empty_capabilities_returns_base_only(self):
        tools = compose_tools_from_capabilities([])
        assert set(tools) == set(_BASE_TOOLS)

    def test_model_only_capability_returns_base_only(self):
        """reasoning has no tool requirements — only base tools."""
        tools = compose_tools_from_capabilities(["reasoning"])
        assert set(tools) == set(_BASE_TOOLS)

    def test_mixed_capabilities_compose_union(self):
        """code_python + web_comprehension = union of both tool sets + base."""
        tools = compose_tools_from_capabilities(["code_python", "web_comprehension"])
        assert "shell" in tools
        assert "write_file" in tools
        assert "run_tests" in tools
        assert "web_search" in tools
        assert "fetch_url" in tools
        assert "read_file" in tools
        assert "list_files" in tools

    def test_result_is_sorted(self):
        tools = compose_tools_from_capabilities(["code_python", "web_comprehension"])
        assert tools == sorted(tools)

    def test_result_is_deduplicated(self):
        """Multiple capabilities may share tools (e.g. shell from code_python and code_rust)."""
        tools = compose_tools_from_capabilities(["code_python", "code_rust"])
        assert len(tools) == len(set(tools))

    def test_unknown_capability_ignored(self):
        """Unknown capabilities are silently ignored (base tools only)."""
        tools = compose_tools_from_capabilities(["telekinesis"])
        assert set(tools) == set(_BASE_TOOLS)

    def test_web_comprehension_matches_research_template(self):
        """web_comprehension should produce exactly the research template's tool set."""
        from agentic_concierge.infrastructure.specialists.dynamic_pack import PACK_TEMPLATES
        tools = compose_tools_from_capabilities(["web_comprehension"])
        assert set(tools) == set(PACK_TEMPLATES["research"].tool_names)

    def test_code_python_matches_engineering_template(self):
        """code_python should produce exactly the engineering template's tool set."""
        from agentic_concierge.infrastructure.specialists.dynamic_pack import PACK_TEMPLATES
        tools = compose_tools_from_capabilities(["code_python"])
        assert set(tools) == set(PACK_TEMPLATES["engineering"].tool_names)

    def test_code_python_web_matches_no_template(self):
        """code_python + web_comprehension should NOT match any template."""
        from agentic_concierge.infrastructure.specialists.dynamic_pack import PACK_TEMPLATES
        tools = compose_tools_from_capabilities(["code_python", "web_comprehension"])
        tool_set = set(tools)
        for tpl in PACK_TEMPLATES.values():
            assert tool_set != set(tpl.tool_names), f"Unexpected match with {tpl.template_id}"


class TestGenerateRoleFromCapabilities:
    """Tests for generate_role_from_capabilities()."""

    def test_web_comprehension_mentions_web(self):
        role = generate_role_from_capabilities(["web_comprehension"])
        assert "web search" in role.lower()

    def test_code_python_mentions_python(self):
        role = generate_role_from_capabilities(["code_python"])
        assert "python" in role.lower()

    def test_empty_capabilities_returns_generic(self):
        role = generate_role_from_capabilities([])
        assert "specialist agent" in role.lower()

    def test_mixed_capabilities_mentioned(self):
        role = generate_role_from_capabilities(["code_python", "web_comprehension"])
        assert "python" in role.lower()
        assert "web" in role.lower()

    def test_unknown_capability_returns_generic(self):
        role = generate_role_from_capabilities(["telekinesis"])
        assert "specialist agent" in role.lower()


class TestInferFinishSchemaFromCapabilities:
    """Tests for infer_finish_schema_from_capabilities()."""

    def test_code_only_returns_code(self):
        assert infer_finish_schema_from_capabilities(["code_python"]) == "code"

    def test_web_only_returns_quick_answer(self):
        assert infer_finish_schema_from_capabilities(["web_comprehension"]) == "quick_answer"

    def test_mixed_code_and_web_returns_none(self):
        assert infer_finish_schema_from_capabilities(["code_python", "web_comprehension"]) is None

    def test_reasoning_only_returns_none(self):
        assert infer_finish_schema_from_capabilities(["reasoning"]) is None

    def test_code_rust_returns_code(self):
        assert infer_finish_schema_from_capabilities(["code_rust"]) == "code"

    def test_code_sql_returns_code(self):
        assert infer_finish_schema_from_capabilities(["code_sql"]) == "code"

    def test_web_plus_summarisation_returns_quick_answer(self):
        assert infer_finish_schema_from_capabilities(
            ["web_comprehension", "summarisation"]
        ) == "quick_answer"


# ---------------------------------------------------------------------------
# Issue #1: KNOWN_CAPABILITIES export
# ---------------------------------------------------------------------------


def test_known_capabilities_is_frozenset():
    from agentic_concierge.infrastructure.model_profiles import KNOWN_CAPABILITIES
    assert isinstance(KNOWN_CAPABILITIES, frozenset)
    assert "web_comprehension" in KNOWN_CAPABILITIES
    assert "code_python" in KNOWN_CAPABILITIES
    assert "reasoning" in KNOWN_CAPABILITIES
    assert "fabricated_cap" not in KNOWN_CAPABILITIES


# ---------------------------------------------------------------------------
# Issue #5: qwen3-coder model profile
# ---------------------------------------------------------------------------


def test_extract_family_qwen3_coder():
    assert _extract_family("qwen3-coder:30b") == "qwen3-coder"


def test_get_profile_qwen3_coder():
    p = get_profile("qwen3-coder:30b")
    assert p.family == "qwen3-coder"
    assert p.capabilities["code_python"] >= 0.9
    assert p.capabilities["web_comprehension"] <= 0.4


def test_qwen3_coder_not_selected_for_web_task():
    result = match_models(
        ["qwen2.5:7b", "qwen3-coder:30b"],
        required_capabilities={"web_comprehension": 0.6},
    )
    assert result == "qwen2.5:7b"


def test_qwen3_coder_selected_for_code_task():
    result = match_models(
        ["qwen2.5:7b", "qwen3-coder:30b"],
        required_capabilities={"code_python": 0.8},
    )
    assert result == "qwen3-coder:30b"
