"""Tests for ConfigSpecialistRegistry extensibility (T1-4).

Covers:
- Built-in packs (engineering, research) resolve without builder config.
- A specialist with ``builder=`` loads a custom factory without editing registry.py.
- ``builder=`` on a built-in id overrides the built-in implementation.
- Error paths: unknown specialist, no builder + no built-in, bad dotted path.
- ``list_ids`` returns all configured specialists.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from agentic_concierge.application.ports import SpecialistPack
from agentic_concierge.config import load_config
from agentic_concierge.config.schema import ConciergeConfig, SpecialistConfig
from agentic_concierge.infrastructure.specialists import ConfigSpecialistRegistry
from agentic_concierge.infrastructure.specialists.registry import _load_builder


# ---------------------------------------------------------------------------
# Stub pack (importable as tests.test_specialist_registry:build_stub_pack)
# ---------------------------------------------------------------------------

class _StubPack:
    """Minimal SpecialistPack implementation for testing."""
    system_prompt: str = "stub"
    tool_definitions: List[Dict[str, Any]] = []
    finish_tool_name: str = "finish_task"
    finish_required_fields: List[str] = []

    async def execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        return {"stub": True}

    @property
    def tool_names(self) -> List[str]:
        return []


def build_stub_pack(workspace_path: str, network_allowed: bool) -> SpecialistPack:
    """Custom pack factory — registered via SpecialistConfig.builder in tests."""
    return _StubPack()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config_with_specialist(specialist_id: str, builder_path: str | None) -> ConciergeConfig:
    """Return a ConciergeConfig that adds a custom specialist to the defaults."""
    base = load_config()
    return ConciergeConfig(
        models=base.models,
        specialists={
            **base.specialists,
            specialist_id: SpecialistConfig(
                description="Test specialist.",
                keywords=["test"],
                workflow="test",
                builder=builder_path,
            ),
        },
    )


# ---------------------------------------------------------------------------
# Built-in packs
# ---------------------------------------------------------------------------

def test_builtin_engineering_pack_resolves():
    """Built-in engineering pack resolves without any builder config."""
    registry = ConfigSpecialistRegistry(load_config())
    pack = registry.get_pack("engineering", "/tmp", network_allowed=False)
    assert pack.finish_tool_name == "finish_task"
    assert "summary" in pack.finish_required_fields


def test_builtin_research_pack_resolves():
    """Built-in research pack resolves without any builder config."""
    registry = ConfigSpecialistRegistry(load_config())
    pack = registry.get_pack("research", "/tmp", network_allowed=False)
    assert pack.finish_tool_name == "finish_task"
    assert "answer" in pack.finish_required_fields


# ---------------------------------------------------------------------------
# Custom builder via config (the key T1-4 feature)
# ---------------------------------------------------------------------------

def test_custom_pack_loaded_from_builder_field():
    """A specialist with builder= loads a pack without modifying registry.py."""
    config = _make_config_with_specialist(
        "custom",
        "tests.test_specialist_registry:build_stub_pack",
    )
    registry = ConfigSpecialistRegistry(config)
    pack = registry.get_pack("custom", "/tmp", network_allowed=False)
    assert isinstance(pack, _StubPack)


def test_builder_field_overrides_builtin_for_same_id():
    """builder= on a built-in specialist id uses the custom factory, not the built-in."""
    base = load_config()
    config = ConciergeConfig(
        models=base.models,
        specialists={
            "engineering": SpecialistConfig(
                description="Overridden engineering.",
                keywords=["build"],
                workflow="engineering",
                builder="tests.test_specialist_registry:build_stub_pack",
            ),
        },
    )
    registry = ConfigSpecialistRegistry(config)
    pack = registry.get_pack("engineering", "/tmp", network_allowed=False)
    assert isinstance(pack, _StubPack)


def test_workspace_path_and_network_forwarded_to_factory(tmp_path):
    """workspace_path and network_allowed are passed through to the factory."""
    received: dict = {}

    def capturing_factory(workspace_path: str, network_allowed: bool) -> SpecialistPack:
        received["workspace_path"] = workspace_path
        received["network_allowed"] = network_allowed
        return _StubPack()

    # Inject into sys.modules so _load_builder can find it.
    import sys
    import types
    mod = types.ModuleType("_test_capturing_factory_mod")
    mod.factory = capturing_factory  # type: ignore[attr-defined]
    sys.modules["_test_capturing_factory_mod"] = mod

    try:
        config = _make_config_with_specialist(
            "capture_test",
            "_test_capturing_factory_mod:factory",
        )
        registry = ConfigSpecialistRegistry(config)
        registry.get_pack("capture_test", str(tmp_path), network_allowed=True)
    finally:
        del sys.modules["_test_capturing_factory_mod"]

    assert received["workspace_path"] == str(tmp_path)
    assert received["network_allowed"] is True


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_unknown_specialist_raises_value_error():
    """get_pack with an id not in config raises ValueError."""
    registry = ConfigSpecialistRegistry(load_config())
    with pytest.raises(ValueError, match="Unknown specialist"):
        registry.get_pack("nonexistent", "/tmp", network_allowed=False)


def test_no_builder_and_no_builtin_raises_value_error():
    """Specialist in config with no builder and no built-in raises ValueError."""
    config = _make_config_with_specialist("mystery", builder_path=None)
    registry = ConfigSpecialistRegistry(config)
    with pytest.raises(ValueError, match="No pack implementation"):
        registry.get_pack("mystery", "/tmp", network_allowed=False)


def test_load_builder_rejects_path_without_colon():
    """_load_builder raises ValueError for a path without ':'."""
    with pytest.raises(ValueError, match="expected 'module.path:function_name'"):
        _load_builder("no_colon_here")


def test_load_builder_raises_on_nonexistent_module():
    """_load_builder raises ImportError for a module that cannot be imported."""
    with pytest.raises(ImportError, match="Cannot import builder module"):
        _load_builder("nonexistent.module.xyz:factory")


def test_load_builder_raises_on_missing_function():
    """_load_builder raises ImportError when the function doesn't exist in the module."""
    with pytest.raises(ImportError):
        _load_builder("agentic_concierge.config:_no_such_function_xyz")


# ---------------------------------------------------------------------------
# list_ids
# ---------------------------------------------------------------------------

def test_list_ids_includes_custom_specialist():
    """list_ids returns all configured specialists, including custom ones."""
    config = _make_config_with_specialist(
        "custom",
        "tests.test_specialist_registry:build_stub_pack",
    )
    registry = ConfigSpecialistRegistry(config)
    ids = registry.list_ids()
    assert "engineering" in ids
    assert "research" in ids
    assert "custom" in ids


def test_list_ids_default_config():
    """Default config includes the built-in specialist packs."""
    registry = ConfigSpecialistRegistry(load_config())
    ids = set(registry.list_ids())
    assert "engineering" in ids
    assert "research" in ids
    assert "enterprise_research" in ids
