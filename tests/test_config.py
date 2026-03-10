"""Tests for config loading."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_concierge.config import DEFAULT_CONFIG, ConciergeConfig, get_config, load_config
from agentic_concierge.config import loader as config_loader
from agentic_concierge.config.schema import MCPServerConfig, ModelConfig, SpecialistConfig


def test_get_config_default(monkeypatch):
    monkeypatch.delenv("CONCIERGE_CONFIG_PATH", raising=False)
    monkeypatch.setattr(config_loader, "_env", None)
    cfg = get_config()
    assert cfg is DEFAULT_CONFIG
    assert "engineering" in cfg.specialists
    assert "quality" in cfg.models


def test_get_config_from_file(monkeypatch):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({
            "models": {
                "custom": {
                    "base_url": "http://127.0.0.1:9000/v1",
                    "model": "my-model",
                    "temperature": 0.2,
                    "max_tokens": 1000,
                }
            },
            "specialists": DEFAULT_CONFIG.model_dump()["specialists"],
        }, f)
        path = f.name
    try:
        monkeypatch.setenv("CONCIERGE_CONFIG_PATH", path)
        monkeypatch.setattr(config_loader, "_env", None)
        cfg = get_config()
        assert cfg.models["custom"].model == "my-model"
        assert cfg.models["custom"].temperature == 0.2
    finally:
        Path(path).unlink(missing_ok=True)


def test_config_local_llm_default():
    """Local LLM is default and primary: ensure_available is True by default."""
    assert DEFAULT_CONFIG.local_llm_ensure_available is True
    assert DEFAULT_CONFIG.local_llm_start_cmd == ["ollama", "serve"]
    assert DEFAULT_CONFIG.local_llm_start_timeout_s == 90


def test_config_local_llm_from_file(monkeypatch):
    """New key names load correctly."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({
            "models": DEFAULT_CONFIG.model_dump()["models"],
            "specialists": DEFAULT_CONFIG.model_dump()["specialists"],
            "local_llm_ensure_available": False,
            "local_llm_start_cmd": ["/usr/bin/ollama", "serve"],
            "local_llm_start_timeout_s": 120,
        }, f)
        path = f.name
    try:
        monkeypatch.setenv("CONCIERGE_CONFIG_PATH", path)
        monkeypatch.setattr(config_loader, "_env", None)
        cfg = get_config()
        assert cfg.local_llm_ensure_available is False
        assert cfg.local_llm_start_cmd == ["/usr/bin/ollama", "serve"]
        assert cfg.local_llm_start_timeout_s == 120
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_config_is_cached(monkeypatch):
    """Repeated calls to load_config() return the same object (cache hit)."""
    monkeypatch.delenv("CONCIERGE_CONFIG_PATH", raising=False)
    first = load_config()
    second = load_config()
    assert first is second, "load_config() must return the cached object on repeat calls"


def test_load_config_cache_clear_forces_reload(monkeypatch, tmp_path):
    """cache_clear() forces the next call to re-read from disk."""
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(json.dumps({
        "models": {"q": {"base_url": "http://localhost:11434/v1", "model": "m1"}},
        "specialists": DEFAULT_CONFIG.model_dump()["specialists"],
    }))

    monkeypatch.setenv("CONCIERGE_CONFIG_PATH", str(cfg_file))
    monkeypatch.setattr(config_loader, "_env", None)
    first = load_config()
    assert first.models["q"].model == "m1"

    # Overwrite config file with a different model name.
    cfg_file.write_text(json.dumps({
        "models": {"q": {"base_url": "http://localhost:11434/v1", "model": "m2"}},
        "specialists": DEFAULT_CONFIG.model_dump()["specialists"],
    }))

    # Without cache_clear, still the old cached result.
    assert load_config().models["q"].model == "m1"

    # After cache_clear, fresh read picks up the new file.
    load_config.cache_clear()
    monkeypatch.setattr(config_loader, "_env", None)
    second = load_config()
    assert second.models["q"].model == "m2"
    assert first is not second


def _minimal_model() -> ModelConfig:
    return ModelConfig(base_url="http://localhost:11434/v1", model="test-model")


def test_default_config_is_valid():
    """DEFAULT_CONFIG must pass all ConciergeConfig validators."""
    # Constructing DEFAULT_CONFIG at import time already validates it; this
    # test makes the expectation explicit and will catch regressions.
    assert "engineering" in DEFAULT_CONFIG.specialists
    assert "research" in DEFAULT_CONFIG.specialists


def test_empty_specialists_raises_validation_error():
    """A config with no specialists is rejected at construction time."""
    with pytest.raises(ValidationError, match="specialists must not be empty"):
        ConciergeConfig(models={"q": _minimal_model()}, specialists={})


def test_single_specialist_is_valid():
    """A config with exactly one specialist passes validation."""
    cfg = ConciergeConfig(
        models={"q": _minimal_model()},
        specialists={
            "engineering": SpecialistConfig(
                description="builds things",
                keywords=["build"],
                workflow="engineering",
            )
        },
    )
    assert list(cfg.specialists) == ["engineering"]


def test_config_legacy_auto_start_llm_keys(monkeypatch):
    """Legacy auto_start_llm / llm_start_cmd / llm_start_timeout_s map to new names."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({
            "models": DEFAULT_CONFIG.model_dump()["models"],
            "specialists": DEFAULT_CONFIG.model_dump()["specialists"],
            "auto_start_llm": True,
            "llm_start_cmd": ["/usr/bin/ollama", "serve"],
            "llm_start_timeout_s": 120,
        }, f)
        path = f.name
    try:
        monkeypatch.setenv("CONCIERGE_CONFIG_PATH", path)
        monkeypatch.setattr(config_loader, "_env", None)
        cfg = get_config()
        assert cfg.local_llm_ensure_available is True
        assert cfg.local_llm_start_cmd == ["/usr/bin/ollama", "serve"]
        assert cfg.local_llm_start_timeout_s == 120
    finally:
        Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# MCP config tests (P5-1)
# ---------------------------------------------------------------------------

def test_mcp_server_config_valid_stdio():
    """MCPServerConfig with transport='stdio' is valid when command is set."""
    cfg = MCPServerConfig(name="github", transport="stdio", command="npx", args=["-y", "@modelcontextprotocol/server-github"])
    assert cfg.name == "github"
    assert cfg.transport == "stdio"
    assert cfg.command == "npx"
    assert cfg.args == ["-y", "@modelcontextprotocol/server-github"]


def test_mcp_server_config_valid_sse():
    """MCPServerConfig with transport='sse' is valid when url is set."""
    cfg = MCPServerConfig(name="jira", transport="sse", url="http://localhost:3000/sse")
    assert cfg.name == "jira"
    assert cfg.transport == "sse"
    assert cfg.url == "http://localhost:3000/sse"


def test_mcp_server_config_stdio_missing_command_raises():
    """MCPServerConfig with transport='stdio' and no command raises ValidationError."""
    with pytest.raises(ValidationError, match="requires 'command'"):
        MCPServerConfig(name="bad", transport="stdio")


def test_mcp_server_config_sse_missing_url_raises():
    """MCPServerConfig with transport='sse' and no url raises ValidationError."""
    with pytest.raises(ValidationError, match="requires 'url'"):
        MCPServerConfig(name="bad", transport="sse")


def test_specialist_config_mcp_servers_default_empty():
    """SpecialistConfig.mcp_servers defaults to an empty list."""
    spec = SpecialistConfig(description="test", keywords=[], workflow="test")
    assert spec.mcp_servers == []


def test_specialist_config_duplicate_mcp_server_names_raises():
    """SpecialistConfig rejects duplicate MCP server names."""
    with pytest.raises(ValidationError, match="Duplicate MCP server names"):
        SpecialistConfig(
            description="test",
            keywords=[],
            workflow="test",
            mcp_servers=[
                MCPServerConfig(name="dup", transport="stdio", command="npx"),
                MCPServerConfig(name="dup", transport="stdio", command="npx"),
            ],
        )


# ---------------------------------------------------------------------------
# P10-4: profile, features, resource_limits fields on ConciergeConfig
# ---------------------------------------------------------------------------

def test_config_default_profile_is_auto():
    assert DEFAULT_CONFIG.profile == "auto"


def test_config_features_default_all_none():
    """All feature overrides default to None (use profile default)."""
    from agentic_concierge.config.schema import FeaturesConfig
    fc = FeaturesConfig()
    for field_name in ["ollama", "llama_cpp", "vllm", "cloud", "mcp",
                        "browser", "embedding", "telemetry", "container"]:
        assert getattr(fc, field_name) is None


def test_config_resource_limits_defaults():
    from agentic_concierge.config.schema import ResourceLimitsConfig
    rl = ResourceLimitsConfig()
    assert rl.max_concurrent_agents == 4
    assert rl.max_ram_mb is None
    assert rl.max_gpu_vram_mb is None
    assert rl.model_cache_path == ""


def test_config_features_override_round_trip():
    """Features can be set to True/False and round-trip through model_dump."""
    from agentic_concierge.config.schema import FeaturesConfig
    fc = FeaturesConfig(ollama=True, vllm=False)
    d = fc.model_dump()
    assert d["ollama"] is True
    assert d["vllm"] is False
    assert d["llama_cpp"] is None


def test_config_profile_field_from_file(monkeypatch):
    """profile field loads from JSON config."""
    import tempfile
    from agentic_concierge.config import loader as config_loader
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        import json
        json.dump({
            "models": DEFAULT_CONFIG.model_dump()["models"],
            "specialists": DEFAULT_CONFIG.model_dump()["specialists"],
            "profile": "medium",
        }, f)
        path = f.name
    try:
        monkeypatch.setenv("CONCIERGE_CONFIG_PATH", path)
        monkeypatch.setattr(config_loader, "_env", None)
        from agentic_concierge.config import get_config
        cfg = get_config()
        assert cfg.profile == "medium"
    finally:
        from pathlib import Path
        Path(path).unlink(missing_ok=True)


def test_config_resource_limits_from_file(monkeypatch):
    """resource_limits loads from JSON config."""
    import tempfile
    from agentic_concierge.config import loader as config_loader
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        import json
        json.dump({
            "models": DEFAULT_CONFIG.model_dump()["models"],
            "specialists": DEFAULT_CONFIG.model_dump()["specialists"],
            "resource_limits": {"max_concurrent_agents": 8, "max_ram_mb": 16384},
        }, f)
        path = f.name
    try:
        monkeypatch.setenv("CONCIERGE_CONFIG_PATH", path)
        monkeypatch.setattr(config_loader, "_env", None)
        from agentic_concierge.config import get_config
        cfg = get_config()
        assert cfg.resource_limits.max_concurrent_agents == 8
        assert cfg.resource_limits.max_ram_mb == 16384
    finally:
        from pathlib import Path
        Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# P11-5: RunIndexConfig new fields (provider, chromadb_path, chromadb_collection)
# ---------------------------------------------------------------------------

def test_run_index_config_provider_defaults_to_jsonl():
    from agentic_concierge.config.schema import RunIndexConfig
    cfg = RunIndexConfig()
    assert cfg.provider == "jsonl"


def test_run_index_config_chromadb_provider_roundtrip():
    from agentic_concierge.config.schema import RunIndexConfig
    cfg = RunIndexConfig(provider="chromadb")
    d = cfg.model_dump()
    assert d["provider"] == "chromadb"
    cfg2 = RunIndexConfig(**d)
    assert cfg2.provider == "chromadb"


def test_run_index_config_chromadb_path_defaults_empty():
    from agentic_concierge.config.schema import RunIndexConfig
    cfg = RunIndexConfig()
    assert cfg.chromadb_path == ""


def test_run_index_config_chromadb_collection_default():
    from agentic_concierge.config.schema import RunIndexConfig
    cfg = RunIndexConfig()
    assert cfg.chromadb_collection == "agentic_concierge_runs"
