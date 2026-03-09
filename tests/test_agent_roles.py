"""Tests for agent roles and model assignment.

Verifies:
- Role lookup and listing
- Role inference from capabilities
- Model assignment via ModelRuntime
- must_differ_from constraint
- Fail-open on must_differ_from when no alternative available
- Task capability merging
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from agentic_concierge.application.agent_roles import (
    AgentRole,
    ModelAssignment,
    assign_model,
    determine_role,
    get_role,
    list_roles,
)


# ---------------------------------------------------------------------------
# Mock ModelHandle and ModelRuntime
# ---------------------------------------------------------------------------


@dataclass
class MockHandle:
    model_id: str
    released: bool = False


class MockRuntime:
    """Minimal ModelRuntime mock for testing assign_model."""

    def __init__(
        self,
        model_id: str = "qwen2.5:7b",
        *,
        fail_on_exclude: Optional[list] = None,
    ) -> None:
        self._model_id = model_id
        self._fail_on_exclude = fail_on_exclude or []
        self.acquire_calls: List[Dict[str, Any]] = []

    async def acquire(
        self,
        requirements: Dict[str, float],
        *,
        prefer_model: Optional[str] = None,
        require_tool_calling: bool = True,
        exclude_models: Optional[List[str]] = None,
        timeout_s: float = 30.0,
    ) -> MockHandle:
        self.acquire_calls.append({
            "requirements": requirements,
            "require_tool_calling": require_tool_calling,
            "exclude_models": exclude_models,
        })
        # Simulate failure when exclusion removes all options
        if exclude_models and set(exclude_models) == set(self._fail_on_exclude):
            raise RuntimeError("No model available")
        return MockHandle(model_id=self._model_id)


class AlwaysFailRuntime:
    """Runtime that always fails."""

    async def acquire(self, *args, **kwargs) -> MockHandle:
        raise RuntimeError("No models available")


# ---------------------------------------------------------------------------
# Role lookup
# ---------------------------------------------------------------------------


class TestGetRole:

    def test_known_roles(self):
        for name in ("router", "planner", "critic", "coder", "researcher", "reviewer"):
            role = get_role(name)
            assert role.name == name
            assert len(role.requirements) > 0

    def test_unknown_role_raises(self):
        with pytest.raises(KeyError, match="Unknown agent role"):
            get_role("wizard")

    def test_list_roles(self):
        roles = list_roles()
        assert "router" in roles
        assert "planner" in roles
        assert "coder" in roles
        assert len(roles) == 6


# ---------------------------------------------------------------------------
# Role properties
# ---------------------------------------------------------------------------


class TestRoleProperties:

    def test_critic_must_differ_from_planner(self):
        critic = get_role("critic")
        assert "planner" in critic.must_differ_from

    def test_reviewer_must_differ_from_coder_and_researcher(self):
        reviewer = get_role("reviewer")
        assert "coder" in reviewer.must_differ_from
        assert "researcher" in reviewer.must_differ_from

    def test_router_prefers_small(self):
        router = get_role("router")
        assert router.prefer_small is True

    def test_planner_does_not_prefer_small(self):
        planner = get_role("planner")
        assert planner.prefer_small is False

    def test_all_roles_require_tool_calling(self):
        for name in list_roles():
            role = get_role(name)
            assert role.require_tool_calling is True


# ---------------------------------------------------------------------------
# determine_role
# ---------------------------------------------------------------------------


class TestDetermineRole:

    def test_code_capabilities_map_to_coder(self):
        assert determine_role(["code_python"]) == "coder"
        assert determine_role(["code_rust"]) == "coder"
        assert determine_role(["code_sql"]) == "coder"

    def test_web_capabilities_map_to_researcher(self):
        assert determine_role(["web_comprehension"]) == "researcher"
        assert determine_role(["summarisation"]) == "researcher"

    def test_reasoning_maps_to_planner(self):
        assert determine_role(["reasoning"]) == "planner"

    def test_mixed_capabilities_majority_wins(self):
        # 2 code caps vs 1 web cap → coder
        assert determine_role(["code_python", "code_rust", "web_comprehension"]) == "coder"

    def test_empty_defaults_to_researcher(self):
        assert determine_role([]) == "researcher"

    def test_unknown_capabilities_default_to_researcher(self):
        assert determine_role(["teleportation", "time_travel"]) == "researcher"


# ---------------------------------------------------------------------------
# assign_model
# ---------------------------------------------------------------------------


class TestAssignModel:

    @pytest.mark.asyncio
    async def test_basic_assignment(self):
        runtime = MockRuntime("qwen2.5:7b")
        result = await assign_model("coder", runtime)

        assert result.role == "coder"
        assert result.model_id == "qwen2.5:7b"
        assert result.handle.model_id == "qwen2.5:7b"
        assert "code_python" in result.requirements

    @pytest.mark.asyncio
    async def test_task_capabilities_merged(self):
        runtime = MockRuntime()
        result = await assign_model(
            "coder", runtime,
            task_capabilities=["code_sql", "structured_output"],
        )

        # code_python from role + code_sql and structured_output from task
        assert "code_python" in result.requirements
        assert "code_sql" in result.requirements
        assert "structured_output" in result.requirements

    @pytest.mark.asyncio
    async def test_task_caps_dont_override_role_requirements(self):
        """Task caps don't lower the role's own requirement thresholds."""
        runtime = MockRuntime()
        result = await assign_model(
            "coder", runtime,
            task_capabilities=["code_python"],
        )
        # Role defines code_python: 0.8, task default is 0.6
        # Role requirement should win
        assert result.requirements["code_python"] == 0.8

    @pytest.mark.asyncio
    async def test_require_tool_calling_passed(self):
        runtime = MockRuntime()
        await assign_model("coder", runtime)

        assert len(runtime.acquire_calls) == 1
        assert runtime.acquire_calls[0]["require_tool_calling"] is True

    @pytest.mark.asyncio
    async def test_no_model_available_raises(self):
        runtime = AlwaysFailRuntime()
        with pytest.raises(RuntimeError):
            await assign_model("coder", runtime)


# ---------------------------------------------------------------------------
# must_differ_from constraint
# ---------------------------------------------------------------------------


class TestMustDifferFrom:

    @pytest.mark.asyncio
    async def test_exclude_planner_model_for_critic(self):
        runtime = MockRuntime("phi-4-mini:14b")
        await assign_model(
            "critic", runtime,
            node_model_map={"planner": "qwen2.5:7b"},
        )

        # Should have passed exclude_models=["qwen2.5:7b"]
        call = runtime.acquire_calls[0]
        assert "qwen2.5:7b" in call["exclude_models"]

    @pytest.mark.asyncio
    async def test_exclude_multiple_roles_for_reviewer(self):
        runtime = MockRuntime("phi-4-mini:14b")
        await assign_model(
            "reviewer", runtime,
            node_model_map={
                "coder": "qwen2.5-coder:7b",
                "researcher": "qwen2.5:7b",
            },
        )

        call = runtime.acquire_calls[0]
        assert "qwen2.5-coder:7b" in call["exclude_models"]
        assert "qwen2.5:7b" in call["exclude_models"]

    @pytest.mark.asyncio
    async def test_no_map_means_no_excludes(self):
        runtime = MockRuntime()
        await assign_model("critic", runtime)

        call = runtime.acquire_calls[0]
        assert call["exclude_models"] is None or call["exclude_models"] == []

    @pytest.mark.asyncio
    async def test_missing_role_in_map_ignored(self):
        """If the must_differ_from role isn't in the map, no exclusion."""
        runtime = MockRuntime()
        await assign_model(
            "critic", runtime,
            node_model_map={"coder": "qwen2.5:7b"},  # no "planner" entry
        )

        call = runtime.acquire_calls[0]
        # Should not have planner model in excludes
        excludes = call.get("exclude_models") or []
        assert "qwen2.5:7b" not in excludes

    @pytest.mark.asyncio
    async def test_failopen_relaxes_must_differ_from(self):
        """When exclusion causes failure, retry without must_differ_from."""
        runtime = MockRuntime(
            "qwen2.5:7b",
            fail_on_exclude=["qwen2.5:7b"],
        )
        result = await assign_model(
            "critic", runtime,
            node_model_map={"planner": "qwen2.5:7b"},
        )

        # Should have made 2 calls: first with exclude, second without
        assert len(runtime.acquire_calls) == 2
        assert runtime.acquire_calls[0]["exclude_models"] == ["qwen2.5:7b"]
        # Second call should have no must_differ_from excludes
        assert result.model_id == "qwen2.5:7b"

    @pytest.mark.asyncio
    async def test_explicit_exclude_preserved_on_failopen(self):
        """Explicit exclude_models are preserved even when must_differ_from relaxed."""
        runtime = MockRuntime(
            "qwen2.5:7b",
            fail_on_exclude=["qwen2.5:7b", "explicit-bad"],
        )
        await assign_model(
            "critic", runtime,
            node_model_map={"planner": "qwen2.5:7b"},
            exclude_models=["explicit-bad"],
        )

        # Second call should still have explicit excludes
        assert len(runtime.acquire_calls) == 2
        second_call = runtime.acquire_calls[1]
        assert second_call["exclude_models"] == ["explicit-bad"]
