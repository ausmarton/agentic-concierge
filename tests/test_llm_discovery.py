"""Tests for LLM discovery and model selection (chat vs embedding)."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from agentic_concierge.infrastructure.llm_discovery import (
    _is_ollama_chat_capable,
    _nano_model_path,
    _ollama_model_name,
    _try_backend,
    discover_ollama_models,
    resolve_llm,
    select_model,
)
from agentic_concierge.config.schema import DEFAULT_CONFIG, ConciergeConfig, ModelConfig


@pytest.mark.parametrize("model,expected", [
    ({"name": "bge-m3:latest", "model": "bge-m3:latest", "details": {"family": "bge-m3"}}, False),
    ({"name": "nomic-embed-text", "details": {"family": "nomic-embed-text"}}, False),
    ({"name": "qwen2.5:7b", "details": {"family": "qwen2.5"}}, True),
    ({"name": "llama3.1:8b", "details": {"family": "llama"}}, True),
])
def test_is_ollama_chat_capable(model, expected):
    assert _is_ollama_chat_capable(model) is expected


def test_select_model_prefers_chat_models():
    """When both chat and embedding models exist, selection uses only chat-capable list."""
    all_models = [
        {"name": "bge-m3:latest", "model": "bge-m3:latest", "details": {"family": "bge-m3"}},
        {"name": "qwen2.5:7b", "model": "qwen2.5:7b", "details": {"family": "qwen2.5", "parameter_size": "7B"}},
    ]
    chat_only = [m for m in all_models if _is_ollama_chat_capable(m)]
    names = [_ollama_model_name(m) for m in chat_only]
    assert "bge-m3:latest" not in names
    assert "qwen2.5:7b" in names
    # 14b not in list -> fallback to 7b
    selected = select_model("qwen2.5:14b", chat_only, is_ollama=True)
    assert selected == "qwen2.5:7b"


def test_select_model_closest_distance_no_close_match():
    """When preferred 14b is unavailable and all options are far, pick the closest by log-ratio."""
    models = [
        {"name": "qwen2.5:0.5b", "model": "qwen2.5:0.5b", "details": {"family": "qwen2.5", "parameter_size": "0.5B"}},
        {"name": "qwen2.5:32b", "model": "qwen2.5:32b", "details": {"family": "qwen2.5", "parameter_size": "32B"}},
        {"name": "qwen2.5:72b", "model": "qwen2.5:72b", "details": {"family": "qwen2.5", "parameter_size": "72B"}},
    ]
    selected = select_model("qwen2.5:14b", models, is_ollama=True)
    # log-scale: 0.5b is 28x away, 32b is 2.3x away, 72b is 5.1x away -> picks 32b
    assert selected == "qwen2.5:32b"


def test_select_model_excludes_tool_incapable():
    """sqlcoder:15b is chat-capable but not tool-capable; must not be selected."""
    models = [
        {"name": "qwen2.5:32b", "model": "qwen2.5:32b", "details": {"family": "qwen2.5", "parameter_size": "32B"}},
        {"name": "sqlcoder:15b", "model": "sqlcoder:15b", "details": {"family": "sqlcoder", "parameter_size": "15B"}},
    ]
    selected = select_model("qwen2.5:14b", models, is_ollama=True)
    assert selected == "qwen2.5:32b"


def test_select_model_prefers_same_family():
    """Same-family models preferred over closer-sized models from other families."""
    models = [
        {"name": "qwen2.5:32b", "model": "qwen2.5:32b", "details": {"family": "qwen2.5", "parameter_size": "32B"}},
        {"name": "llama3.1:8b", "model": "llama3.1:8b", "details": {"family": "llama", "parameter_size": "8B"}},
    ]
    # llama3.1:8b is closer in size to 14b, but qwen2.5:32b is same family
    selected = select_model("qwen2.5:14b", models, is_ollama=True)
    assert selected == "qwen2.5:32b"


def test_select_model_falls_back_to_largest_smaller():
    """When no model >= preferred exists, select the largest smaller one."""
    models = [
        {"name": "qwen2.5:0.5b", "model": "qwen2.5:0.5b", "details": {"family": "qwen2.5", "parameter_size": "0.5B"}},
        {"name": "qwen2.5:7b", "model": "qwen2.5:7b", "details": {"family": "qwen2.5", "parameter_size": "7B"}},
    ]
    selected = select_model("qwen2.5:14b", models, is_ollama=True)
    assert selected == "qwen2.5:7b"


def test_resolve_llm_filters_embedding_models():
    """resolve_llm with mocked discover returns a chat model even when an embedding model is first."""
    models = [
        {"name": "bge-m3:latest", "model": "bge-m3:latest", "details": {"family": "bge-m3"}},
        {"name": "qwen2.5:7b", "model": "qwen2.5:7b", "details": {"family": "qwen2.5", "parameter_size": "7B"}},
    ]
    with patch("agentic_concierge.infrastructure.llm_discovery.discover_ollama_models", return_value=models):
        with patch("agentic_concierge.infrastructure.llm_bootstrap.ensure_llm_available", return_value=True):
            resolved = resolve_llm(DEFAULT_CONFIG, "quality")
    assert resolved.model != "bge-m3:latest"
    assert resolved.model == "qwen2.5:7b"


# ---------------------------------------------------------------------------
# Adaptive backend resolution tests
# ---------------------------------------------------------------------------

_CHAT_MODELS = [
    {"name": "llama3.1:8b", "model": "llama3.1:8b", "details": {"family": "llama", "parameter_size": "8B"}},
]


def _make_config(**overrides) -> ConciergeConfig:
    """Build a config from DEFAULT_CONFIG with overrides."""
    data = DEFAULT_CONFIG.model_dump()
    data.update(overrides)
    return ConciergeConfig(**data)


def test_resolve_llm_fallback_vllm_to_ollama():
    """vLLM (primary) unreachable, Ollama healthy -> resolves via Ollama with fallback_used=True."""
    cfg = _make_config(
        models={
            "quality": ModelConfig(
                base_url="http://localhost:8000/v1",
                model="some-vllm-model",
                backend="vllm",
            ),
            "fast": ModelConfig(
                base_url="http://localhost:8000/v1",
                model="some-vllm-model",
                backend="vllm",
            ),
        },
        local_llm_ensure_available=False,
        profile="medium",
    )
    with patch("agentic_concierge.infrastructure.llm_discovery.discover_ollama_models") as mock_ollama, \
         patch("agentic_concierge.infrastructure.llm_discovery.discover_openai_models") as mock_openai:
        # Primary (vLLM at :8000) fails
        mock_openai.return_value = None
        mock_ollama.return_value = None

        # Fallback: _try_backend("ollama") succeeds
        def side_effect_ollama(url, timeout_s=15.0):
            if "11434" in url:
                return _CHAT_MODELS
            return None

        mock_ollama.side_effect = side_effect_ollama

        resolved = resolve_llm(cfg, "quality")

    assert resolved.fallback_used is True
    assert resolved.resolved_backend == "ollama"
    assert resolved.model == "llama3.1:8b"
    assert len(resolved.warnings) >= 1


def test_resolve_llm_fallback_to_inprocess():
    """All remote backends down, GGUF exists -> resolves inprocess."""
    cfg = _make_config(
        local_llm_ensure_available=False,
        profile="nano",
    )
    with patch("agentic_concierge.infrastructure.llm_discovery.discover_ollama_models", return_value=None), \
         patch("agentic_concierge.infrastructure.llm_discovery.discover_openai_models", return_value=None), \
         patch("agentic_concierge.infrastructure.llm_discovery._nano_model_path", return_value="/fake/model.gguf"), \
         patch("agentic_concierge.infrastructure.chat.inprocess.is_available", return_value=True), \
         patch("agentic_concierge.infrastructure.llm_discovery.shutil") as mock_shutil:
        mock_shutil.which.return_value = None  # no ollama binary
        resolved = resolve_llm(cfg, "quality")

    assert resolved.fallback_used is True
    assert resolved.resolved_backend == "inprocess"
    assert resolved.model == "/fake/model.gguf"


def test_resolve_llm_configured_backend_tried_first():
    """Configured Ollama tried first before fallback chain kicks in."""
    models = [
        {"name": "qwen2.5:7b", "model": "qwen2.5:7b", "details": {"family": "qwen2.5", "parameter_size": "7B"}},
    ]
    with patch("agentic_concierge.infrastructure.llm_discovery.discover_ollama_models", return_value=models), \
         patch("agentic_concierge.infrastructure.llm_bootstrap.ensure_llm_available", return_value=True):
        resolved = resolve_llm(DEFAULT_CONFIG, "quality")

    # Should resolve directly, no fallback
    assert resolved.fallback_used is False
    assert resolved.model == "qwen2.5:7b"


def test_resolve_llm_respects_disabled_feature():
    """features.ollama=False -> Ollama skipped in fallback chain."""
    from agentic_concierge.config.schema import FeaturesConfig
    cfg = _make_config(
        local_llm_ensure_available=False,
        profile="medium",
        features=FeaturesConfig(ollama=False),
    )
    with patch("agentic_concierge.infrastructure.llm_discovery.discover_ollama_models", return_value=None), \
         patch("agentic_concierge.infrastructure.llm_discovery.discover_openai_models", return_value=None), \
         patch("agentic_concierge.infrastructure.llm_discovery._nano_model_path", return_value=None), \
         patch("agentic_concierge.infrastructure.chat.inprocess.is_available", return_value=False), \
         patch("agentic_concierge.infrastructure.llm_discovery.shutil") as mock_shutil:
        mock_shutil.which.return_value = None

        with pytest.raises(RuntimeError, match="Could not resolve any LLM backend"):
            resolve_llm(cfg, "quality")


def test_resolve_llm_warnings_on_fallback():
    """Verify warnings are populated on fallback."""
    # Use a non-default primary URL so fallback chain tries the default Ollama URL
    cfg = _make_config(
        models={
            "quality": ModelConfig(
                base_url="http://localhost:9999/v1",
                model="nonexistent-model",
            ),
            "fast": ModelConfig(
                base_url="http://localhost:9999/v1",
                model="nonexistent-model",
            ),
        },
        local_llm_ensure_available=False,
        profile="small",
    )
    with patch("agentic_concierge.infrastructure.llm_discovery.discover_ollama_models") as mock_ollama, \
         patch("agentic_concierge.infrastructure.llm_discovery.discover_openai_models", return_value=None), \
         patch("agentic_concierge.infrastructure.llm_discovery.shutil") as mock_shutil:
        mock_shutil.which.return_value = None

        # Primary fails (port 9999), fallback ollama at default URL succeeds
        def side_effect(url, timeout_s=15.0):
            if "11434" in url:
                return _CHAT_MODELS
            return None

        mock_ollama.side_effect = side_effect

        resolved = resolve_llm(cfg, "quality")

    assert resolved.fallback_used is True
    assert len(resolved.warnings) >= 1
    assert "fallback" in resolved.warnings[0].lower()


def test_resolve_llm_auto_starts_ollama_on_fallback():
    """Ollama installed but not running -> auto-started during fallback."""
    # Use a non-default primary URL so fallback chain tries the default Ollama URL
    cfg = _make_config(
        models={
            "quality": ModelConfig(
                base_url="http://localhost:9999/v1",
                model="nonexistent-model",
            ),
            "fast": ModelConfig(
                base_url="http://localhost:9999/v1",
                model="nonexistent-model",
            ),
        },
        local_llm_ensure_available=False,
        profile="small",
    )
    call_count = {"n": 0}

    def ollama_side_effect(url, timeout_s=15.0):
        call_count["n"] += 1
        # Primary at :9999 fails (call 1), first fallback probe fails (call 2),
        # after auto-start succeeds (call 3)
        if call_count["n"] <= 2:
            return None
        return _CHAT_MODELS

    with patch("agentic_concierge.infrastructure.llm_discovery.discover_ollama_models", side_effect=ollama_side_effect), \
         patch("agentic_concierge.infrastructure.llm_discovery.discover_openai_models", return_value=None), \
         patch("agentic_concierge.infrastructure.llm_discovery.shutil") as mock_shutil, \
         patch("agentic_concierge.infrastructure.llm_bootstrap.ensure_llm_available", return_value=True) as mock_ensure:
        mock_shutil.which.return_value = "/usr/bin/ollama"

        resolved = resolve_llm(cfg, "quality")

    assert resolved.fallback_used is True
    assert resolved.resolved_backend == "ollama"
    # ensure_llm_available should have been called for auto-start
    assert mock_ensure.call_count >= 1


def test_resolve_llm_comprehensive_error_all_fail():
    """All backends fail -> RuntimeError lists each attempt."""
    cfg = _make_config(
        local_llm_ensure_available=False,
        profile="small",
        auto_pull_if_missing=False,
    )
    with patch("agentic_concierge.infrastructure.llm_discovery.discover_ollama_models", return_value=None), \
         patch("agentic_concierge.infrastructure.llm_discovery.discover_openai_models", return_value=None), \
         patch("agentic_concierge.infrastructure.llm_discovery._nano_model_path", return_value=None), \
         patch("agentic_concierge.infrastructure.chat.inprocess.is_available", return_value=False), \
         patch("agentic_concierge.infrastructure.llm_discovery.shutil") as mock_shutil:
        mock_shutil.which.return_value = None

        with pytest.raises(RuntimeError, match="Could not resolve any LLM backend"):
            resolve_llm(cfg, "quality")


def test_try_backend_healthy_ollama():
    """Unit test for _try_backend with healthy Ollama."""
    model_cfg = ModelConfig(base_url="http://localhost:11434/v1", model="llama3.1:8b")
    with patch("agentic_concierge.infrastructure.llm_discovery.discover_ollama_models", return_value=_CHAT_MODELS), \
         patch("agentic_concierge.infrastructure.llm_discovery.shutil") as mock_shutil:
        mock_shutil.which.return_value = None
        result = _try_backend("ollama", model_cfg, DEFAULT_CONFIG)

    assert result is not None
    assert result.resolved_backend == "ollama"
    assert result.model == "llama3.1:8b"


def test_try_backend_returns_none_on_unreachable():
    """Unit test: _try_backend returns None when the backend is unreachable."""
    model_cfg = ModelConfig(base_url="http://localhost:11434/v1", model="llama3.1:8b")
    with patch("agentic_concierge.infrastructure.llm_discovery.discover_ollama_models", return_value=None), \
         patch("agentic_concierge.infrastructure.llm_discovery.shutil") as mock_shutil:
        mock_shutil.which.return_value = None  # Ollama not installed
        result = _try_backend("ollama", model_cfg, DEFAULT_CONFIG)

    assert result is None


def test_try_backend_cloud_happy_path():
    """_try_backend('cloud') returns ResolvedLLM when a generic model with api_key exists."""
    from agentic_concierge.infrastructure.llm_discovery import _try_cloud
    cfg = _make_config(
        models={
            "quality": ModelConfig(
                base_url="http://localhost:11434/v1",
                model="qwen2.5:7b",
            ),
            "fast": ModelConfig(
                base_url="http://localhost:11434/v1",
                model="qwen2.5:7b",
            ),
            "cloud_quality": ModelConfig(
                base_url="https://api.openai.com/v1",
                model="gpt-4o",
                backend="generic",
                api_key="sk-test-key-123",
            ),
        },
    )
    model_cfg = cfg.models["quality"]
    result = _try_cloud(cfg, model_cfg)
    assert result is not None
    assert result.resolved_backend == "cloud"
    assert result.model == "gpt-4o"
    assert result.model_config.api_key == "sk-test-key-123"
    assert result.model_config.backend == "generic"


def test_try_backend_cloud_returns_none_without_api_key():
    """_try_cloud returns None when no generic model has an api_key."""
    from agentic_concierge.infrastructure.llm_discovery import _try_cloud
    model_cfg = DEFAULT_CONFIG.models["quality"]
    result = _try_cloud(DEFAULT_CONFIG, model_cfg)
    assert result is None


def test_resolve_llm_fallback_to_cloud():
    """All local backends down, cloud config with api_key exists -> resolves via cloud."""
    cfg = _make_config(
        models={
            "quality": ModelConfig(
                base_url="http://localhost:9999/v1",
                model="nonexistent",
            ),
            "fast": ModelConfig(
                base_url="http://localhost:9999/v1",
                model="nonexistent",
            ),
            "cloud": ModelConfig(
                base_url="https://api.openai.com/v1",
                model="gpt-4o",
                backend="generic",
                api_key="sk-test",
            ),
        },
        local_llm_ensure_available=False,
        profile="small",
        auto_pull_if_missing=False,
    )
    with patch("agentic_concierge.infrastructure.llm_discovery.discover_ollama_models", return_value=None), \
         patch("agentic_concierge.infrastructure.llm_discovery.discover_openai_models", return_value=None), \
         patch("agentic_concierge.infrastructure.llm_discovery._nano_model_path", return_value=None), \
         patch("agentic_concierge.infrastructure.chat.inprocess.is_available", return_value=False), \
         patch("agentic_concierge.infrastructure.llm_discovery.shutil") as mock_shutil:
        mock_shutil.which.return_value = None
        resolved = resolve_llm(cfg, "quality")

    assert resolved.fallback_used is True
    assert resolved.resolved_backend == "cloud"
    assert resolved.model == "gpt-4o"


def test_resolve_llm_primary_success_sets_resolved_backend():
    """When the primary backend succeeds, resolved_backend is populated."""
    models = [
        {"name": "qwen2.5:7b", "model": "qwen2.5:7b", "details": {"family": "qwen2.5", "parameter_size": "7B"}},
    ]
    with patch("agentic_concierge.infrastructure.llm_discovery.discover_ollama_models", return_value=models), \
         patch("agentic_concierge.infrastructure.llm_bootstrap.ensure_llm_available", return_value=True):
        resolved = resolve_llm(DEFAULT_CONFIG, "quality")

    assert resolved.fallback_used is False
    assert resolved.resolved_backend == "ollama"
    assert resolved.warnings == []


# ---------------------------------------------------------------------------
# Closest distance selection tests
# ---------------------------------------------------------------------------

def test_select_model_closest_distance():
    """preferred=14b, available=[0.5b, 8b, 32b] -> picks 32b (upgrade preferred over downgrade)."""
    models = [
        {"name": "qwen2.5:0.5b", "model": "qwen2.5:0.5b", "details": {"family": "qwen2.5", "parameter_size": "0.5B"}},
        {"name": "qwen2.5:8b", "model": "qwen2.5:8b", "details": {"family": "qwen2.5", "parameter_size": "8B"}},
        {"name": "qwen2.5:32b", "model": "qwen2.5:32b", "details": {"family": "qwen2.5", "parameter_size": "32B"}},
    ]
    selected = select_model("qwen2.5:14b", models, is_ollama=True)
    # 8b log-dist=0.56+0.4=0.96 (downgrade), 32b log-dist=0.83 (upgrade) → 32b wins
    assert selected == "qwen2.5:32b"


def test_select_model_log_scale_prefers_closer_ratio():
    """Log-scale with downgrade penalty: 5b has penalty, 40b doesn't."""
    models = [
        {"name": "qwen2.5:5b", "model": "qwen2.5:5b", "details": {"family": "qwen2.5", "parameter_size": "5B"}},
        {"name": "qwen2.5:40b", "model": "qwen2.5:40b", "details": {"family": "qwen2.5", "parameter_size": "40B"}},
    ]
    selected = select_model("qwen2.5:10b", models, is_ollama=True)
    # 5b: log(5/10)=0.69 + 0.4 penalty = 1.09; 40b: log(40/10)=1.39 → 5b still closer
    assert selected == "qwen2.5:5b"


def test_select_model_closest_same_family_first():
    """Same-family models preferred; within family, upgrade preferred over downgrade."""
    models = [
        {"name": "qwen2.5:8b", "model": "qwen2.5:8b", "details": {"family": "qwen2.5", "parameter_size": "8B"}},
        {"name": "qwen2.5:32b", "model": "qwen2.5:32b", "details": {"family": "qwen2.5", "parameter_size": "32B"}},
        {"name": "llama3.1:14b", "model": "llama3.1:14b", "details": {"family": "llama", "parameter_size": "14B"}},
    ]
    # llama 14b is exact distance=0, but qwen family is preferred;
    # within qwen, 32b (upgrade, d=0.83) beats 8b (downgrade, d=0.56+0.4=0.96)
    selected = select_model("qwen2.5:14b", models, is_ollama=True)
    assert selected == "qwen2.5:32b"


# ---------------------------------------------------------------------------
# Downgrade penalty tests — verify that upgrade is preferred over downgrade
# ---------------------------------------------------------------------------


def test_select_model_prefers_upgrade_over_downgrade():
    """14b preferred, 7b and 32b available -> 32b wins (upgrade preferred)."""
    models = [
        {"name": "qwen2.5:7b", "model": "qwen2.5:7b", "details": {"family": "qwen2.5", "parameter_size": "7B"}},
        {"name": "qwen2.5:32b", "model": "qwen2.5:32b", "details": {"family": "qwen2.5", "parameter_size": "32B"}},
    ]
    selected = select_model("qwen2.5:14b", models, is_ollama=True)
    assert selected == "qwen2.5:32b"


def test_select_model_large_downgrade_loses_to_small_upgrade():
    """14b preferred: 72b (5x upgrade) still beats 3b (4.7x downgrade)."""
    models = [
        {"name": "qwen2.5:3b", "model": "qwen2.5:3b", "details": {"family": "qwen2.5", "parameter_size": "3B"}},
        {"name": "qwen2.5:72b", "model": "qwen2.5:72b", "details": {"family": "qwen2.5", "parameter_size": "72B"}},
    ]
    selected = select_model("qwen2.5:14b", models, is_ollama=True)
    # 3b: log(3/14)=1.54+0.4=1.94; 72b: log(72/14)=1.64 → 72b wins
    assert selected == "qwen2.5:72b"


def test_select_model_very_close_downgrade_still_wins():
    """10b preferred: 8b (small downgrade) beats 100b (large upgrade)."""
    models = [
        {"name": "qwen2.5:8b", "model": "qwen2.5:8b", "details": {"family": "qwen2.5", "parameter_size": "8B"}},
        {"name": "qwen2.5:100b", "model": "qwen2.5:100b", "details": {"family": "qwen2.5", "parameter_size": "100B"}},
    ]
    selected = select_model("qwen2.5:10b", models, is_ollama=True)
    # 8b: log(8/10)=0.22+0.4=0.62; 100b: log(100/10)=2.30 → 8b wins (it's very close)
    assert selected == "qwen2.5:8b"


# ---------------------------------------------------------------------------
# resolve_routing_model tests
# ---------------------------------------------------------------------------

def test_resolve_routing_model_exact_match():
    """Configured routing model is available -> returns it."""
    from agentic_concierge.infrastructure.llm_discovery import resolve_routing_model, ResolvedLLM
    resolved = ResolvedLLM(
        base_url="http://localhost:11434/v1",
        model="qwen2.5:14b",
        model_config=ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5:14b"),
        available_models=["qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b"],
    )
    cfg = _make_config()
    routing = resolve_routing_model(resolved, cfg)
    assert routing == "qwen2.5:7b"  # DEFAULT_CONFIG routing_model_key='fast' -> model='qwen2.5:7b'


def test_resolve_routing_model_fallback_task_model_when_not_available():
    """Configured routing model not available -> uses task model (not smallest)."""
    from agentic_concierge.infrastructure.llm_discovery import resolve_routing_model, ResolvedLLM
    resolved = ResolvedLLM(
        base_url="http://localhost:11434/v1",
        model="qwen2.5:14b",
        model_config=ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5:14b"),
        available_models=["qwen2.5:3b", "qwen2.5:32b"],
    )
    cfg = _make_config()
    routing = resolve_routing_model(resolved, cfg)
    assert routing == "qwen2.5:14b"


def test_resolve_routing_model_fallback_task_model():
    """No models available -> returns task model."""
    from agentic_concierge.infrastructure.llm_discovery import resolve_routing_model, ResolvedLLM
    resolved = ResolvedLLM(
        base_url="http://localhost:11434/v1",
        model="qwen2.5:14b",
        model_config=ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5:14b"),
        available_models=[],
    )
    cfg = _make_config()
    routing = resolve_routing_model(resolved, cfg)
    assert routing == "qwen2.5:14b"


# ---------------------------------------------------------------------------
# pick_smaller_model tests
# ---------------------------------------------------------------------------

def test_pick_smaller_model():
    """Returns the largest model smaller than current."""
    from agentic_concierge.infrastructure.llm_discovery import pick_smaller_model
    result = pick_smaller_model(["qwen2.5:3b", "qwen2.5:8b", "qwen2.5:32b"], "qwen2.5:32b")
    assert result == "qwen2.5:8b"


def test_pick_smaller_model_none_when_smallest():
    """Already on smallest -> returns None."""
    from agentic_concierge.infrastructure.llm_discovery import pick_smaller_model
    result = pick_smaller_model(["qwen2.5:3b", "qwen2.5:8b"], "qwen2.5:3b")
    assert result is None


# ---------------------------------------------------------------------------
# Timeout scaling tests
# ---------------------------------------------------------------------------

def test_timeout_scaling_for_larger_model():
    """32b selected for 14b config -> timeout scaled ~2.3x."""
    from agentic_concierge.infrastructure.llm_discovery import _scale_timeout
    cfg = ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5:14b", timeout_s=360.0)
    warnings: list[str] = []
    timeout = _scale_timeout("qwen2.5:32b", cfg, warnings)
    assert timeout > 360.0
    expected = 360.0 * (32 / 14)
    assert abs(timeout - expected) < 1.0
    assert len(warnings) == 1
    assert "timeout scaled" in warnings[0].lower()


def test_timeout_scaling_no_scale_for_smaller():
    """Smaller model selected -> no scaling, original timeout returned."""
    from agentic_concierge.infrastructure.llm_discovery import _scale_timeout
    cfg = ModelConfig(base_url="http://localhost:11434/v1", model="qwen2.5:14b", timeout_s=360.0)
    warnings: list[str] = []
    timeout = _scale_timeout("qwen2.5:8b", cfg, warnings)
    assert timeout == 360.0
    assert len(warnings) == 0


# ---------------------------------------------------------------------------
# available_models populated tests
# ---------------------------------------------------------------------------

def test_available_models_populated():
    """ResolvedLLM.available_models populated after resolve."""
    models = [
        {"name": "qwen2.5:7b", "model": "qwen2.5:7b", "details": {"family": "qwen2.5", "parameter_size": "7B"}},
        {"name": "qwen2.5:32b", "model": "qwen2.5:32b", "details": {"family": "qwen2.5", "parameter_size": "32B"}},
        {"name": "nomic-embed-text", "details": {"family": "nomic-embed-text"}},
    ]
    with patch("agentic_concierge.infrastructure.llm_discovery.discover_ollama_models", return_value=models), \
         patch("agentic_concierge.infrastructure.llm_bootstrap.ensure_llm_available", return_value=True):
        resolved = resolve_llm(DEFAULT_CONFIG, "quality")

    assert "qwen2.5:7b" in resolved.available_models
    assert "qwen2.5:32b" in resolved.available_models
    assert "nomic-embed-text" not in resolved.available_models
