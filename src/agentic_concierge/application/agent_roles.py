"""Agent roles and model assignment for agent-model affinity.

Each agent role (router, planner, critic, coder, researcher, reviewer)
declares its capability requirements.  ``assign_model`` selects the
best model for a role using the ``ModelRuntime`` protocol.

See DESIGN_V2.md §9.1, §9.2, §9.3 for design rationale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent role definitions
# ---------------------------------------------------------------------------

_DEFAULT_TASK_CAP_THRESHOLD = 0.6


@dataclass(frozen=True)
class AgentRole:
    """Specification for an agent role."""

    name: str
    description: str
    requirements: Dict[str, float]  # capability → minimum score
    prefer_small: bool = False
    require_tool_calling: bool = True
    must_differ_from: tuple[str, ...] = ()  # role names


# Bundled role definitions — loaded from DESIGN_V2.md §9.1.
_BUILTIN_ROLES: Dict[str, AgentRole] = {
    "router": AgentRole(
        name="router",
        description="Routes user requests to appropriate decomposition",
        requirements={"structured_output": 0.8, "instruction_following": 0.7},
        prefer_small=True,
        require_tool_calling=True,
    ),
    "planner": AgentRole(
        name="planner",
        description="Decomposes tasks into subtask graphs",
        requirements={
            "reasoning": 0.8,
            "structured_output": 0.7,
            "instruction_following": 0.8,
        },
        prefer_small=False,
        require_tool_calling=True,
    ),
    "critic": AgentRole(
        name="critic",
        description="Reviews plans and identifies issues",
        requirements={"reasoning": 0.8, "instruction_following": 0.7},
        prefer_small=True,
        require_tool_calling=True,
        must_differ_from=("planner",),
    ),
    "coder": AgentRole(
        name="coder",
        description="Writes and modifies code",
        requirements={"code_python": 0.8, "instruction_following": 0.7},
        require_tool_calling=True,
    ),
    "researcher": AgentRole(
        name="researcher",
        description="Searches web and synthesises information",
        requirements={
            "web_comprehension": 0.7,
            "summarisation": 0.7,
            "instruction_following": 0.7,
        },
        require_tool_calling=True,
    ),
    "reviewer": AgentRole(
        name="reviewer",
        description="Reviews completed work for correctness",
        requirements={"reasoning": 0.8, "instruction_following": 0.8},
        prefer_small=True,
        require_tool_calling=True,
        must_differ_from=("coder", "researcher"),
    ),
}


def get_role(name: str) -> AgentRole:
    """Return an agent role by name.

    Raises ``KeyError`` if the role is not defined.
    """
    if name not in _BUILTIN_ROLES:
        raise KeyError(f"Unknown agent role: {name!r}")
    return _BUILTIN_ROLES[name]


def list_roles() -> List[str]:
    """Return all available role names."""
    return list(_BUILTIN_ROLES.keys())


# ---------------------------------------------------------------------------
# Role inference from task capabilities
# ---------------------------------------------------------------------------

# Mapping of capability prefixes to roles.
_CAP_TO_ROLE: Dict[str, str] = {
    "code_python": "coder",
    "code_rust": "coder",
    "code_sql": "coder",
    "web_comprehension": "researcher",
    "summarisation": "researcher",
    # "reasoning" intentionally omitted — too general; researchers, coders,
    # and planners all require reasoning.  Falls back to "researcher".
}


def determine_role(required_capabilities: List[str]) -> str:
    """Determine the best agent role for a set of capabilities.

    Returns the role whose domain best matches the required capabilities.
    Falls back to ``"researcher"`` as the most general role.
    """
    if not required_capabilities:
        return "researcher"

    # Count how many capabilities match each role
    role_scores: Dict[str, int] = {}
    for cap in required_capabilities:
        role = _CAP_TO_ROLE.get(cap)
        if role:
            role_scores[role] = role_scores.get(role, 0) + 1

    if not role_scores:
        return "researcher"

    # Return role with most matching capabilities
    return max(role_scores, key=lambda r: role_scores[r])


# ---------------------------------------------------------------------------
# Model assignment
# ---------------------------------------------------------------------------

@dataclass
class ModelAssignment:
    """Result of assigning a model to an agent role."""

    role: str
    model_id: str
    handle: Any  # ModelHandle from runtime
    requirements: Dict[str, float]


async def assign_model(
    role_name: str,
    runtime: Any,  # ModelRuntime protocol
    *,
    task_capabilities: Optional[List[str]] = None,
    exclude_models: Optional[List[str]] = None,
    node_model_map: Optional[Dict[str, str]] = None,
    prefer_concurrent: bool = False,
) -> ModelAssignment:
    """Select and acquire the best model for an agent role.

    1. Look up role requirements
    2. Merge with task-specific capability requirements
    3. Resolve must_differ_from constraints
    4. Acquire from ModelRuntime

    Args:
        role_name: Agent role name (e.g. "coder", "reviewer").
        runtime: ModelRuntime instance.
        task_capabilities: Additional capability requirements from the task.
        exclude_models: Explicit model IDs to exclude.
        node_model_map: Mapping of role_name → model_id for must_differ_from.

    Returns:
        ModelAssignment with the acquired handle.

    Raises:
        RuntimeError: If no suitable model is available (even after
        relaxing must_differ_from).
    """
    role = get_role(role_name)

    # Merge role requirements with task capabilities
    requirements = dict(role.requirements)
    for cap in (task_capabilities or []):
        if cap not in requirements:
            requirements[cap] = _DEFAULT_TASK_CAP_THRESHOLD

    # Build exclude list from must_differ_from constraint
    all_excludes = list(exclude_models or [])
    if role.must_differ_from and node_model_map:
        for differ_role in role.must_differ_from:
            model_id = node_model_map.get(differ_role)
            if model_id and model_id not in all_excludes:
                all_excludes.append(model_id)

    # Try to acquire with all constraints
    try:
        handle = await runtime.acquire(
            requirements,
            require_tool_calling=role.require_tool_calling,
            exclude_models=all_excludes if all_excludes else None,
            prefer_concurrent=prefer_concurrent,
        )
        return ModelAssignment(
            role=role_name,
            model_id=handle.model_id,
            handle=handle,
            requirements=requirements,
        )
    except RuntimeError:
        if not all_excludes:
            raise  # No constraints to relax

    # Fail-open: relax must_differ_from and retry without excludes
    logger.warning(
        "No model available for role %r with excludes %s; "
        "relaxing must_differ_from constraint",
        role_name, all_excludes,
    )
    try:
        handle = await runtime.acquire(
            requirements,
            require_tool_calling=role.require_tool_calling,
            exclude_models=exclude_models if exclude_models else None,
            prefer_concurrent=prefer_concurrent,
        )
        return ModelAssignment(
            role=role_name,
            model_id=handle.model_id,
            handle=handle,
            requirements=requirements,
        )
    except RuntimeError:
        raise RuntimeError(
            f"No model available for role {role_name!r} "
            f"(requirements: {requirements})"
        )
