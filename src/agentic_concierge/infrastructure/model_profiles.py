"""Model capability profiles: scored capabilities per model family.

Replaces the static ``_TOOL_INCAPABLE_NAMES`` blocklist with a richer,
extensible registry that lets the system reason about what each model
family is good at (code generation, reasoning, web comprehension, etc.).

Profiles are loaded from a bundled dict, extensible via config overrides.
Unknown model families get a permissive default profile.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelCapabilityProfile:
    """Capability profile for a model family."""

    family: str
    supports_tool_calling: bool
    capabilities: Dict[str, float] = field(default_factory=dict)
    min_size_b: float = 0.0
    max_size_b: float = 999.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Bundled profiles
# ---------------------------------------------------------------------------

BUILTIN_PROFILES: Dict[str, ModelCapabilityProfile] = {
    "qwen2.5": ModelCapabilityProfile(
        family="qwen2.5",
        supports_tool_calling=True,
        capabilities={
            "code_python": 0.7, "code_rust": 0.5, "code_sql": 0.6,
            "reasoning": 0.7, "web_comprehension": 0.6,
            "summarisation": 0.7, "instruction_following": 0.8,
            "structured_output": 0.8,
        },
    ),
    "qwen2.5-coder": ModelCapabilityProfile(
        family="qwen2.5-coder",
        supports_tool_calling=True,
        capabilities={
            "code_python": 0.95, "code_rust": 0.8, "code_sql": 0.85,
            "reasoning": 0.5, "web_comprehension": 0.3,
            "summarisation": 0.4, "instruction_following": 0.6,
            "structured_output": 0.7,
        },
    ),
    "qwen3": ModelCapabilityProfile(
        family="qwen3",
        supports_tool_calling=True,
        capabilities={
            "code_python": 0.75, "code_rust": 0.6, "code_sql": 0.65,
            "reasoning": 0.8, "web_comprehension": 0.7,
            "summarisation": 0.75, "instruction_following": 0.85,
            "structured_output": 0.8,
        },
    ),
    "llama3.1": ModelCapabilityProfile(
        family="llama3.1",
        supports_tool_calling=True,
        capabilities={
            "code_python": 0.6, "code_rust": 0.4, "code_sql": 0.5,
            "reasoning": 0.7, "web_comprehension": 0.6,
            "summarisation": 0.7, "instruction_following": 0.7,
            "structured_output": 0.7,
        },
    ),
    "phi-4-mini": ModelCapabilityProfile(
        family="phi-4-mini",
        supports_tool_calling=True,
        capabilities={
            "code_python": 0.6, "code_rust": 0.4, "code_sql": 0.5,
            "reasoning": 0.85, "web_comprehension": 0.5,
            "summarisation": 0.6, "instruction_following": 0.7,
            "structured_output": 0.7,
        },
        notes="Strong reasoning despite small size.",
    ),
    "deepseek-r1-distill": ModelCapabilityProfile(
        family="deepseek-r1-distill",
        supports_tool_calling=True,
        capabilities={
            "code_python": 0.6, "code_rust": 0.4, "code_sql": 0.5,
            "reasoning": 0.9, "web_comprehension": 0.4,
            "summarisation": 0.5, "instruction_following": 0.5,
            "structured_output": 0.5,
        },
        notes="Reasoning-first model; may struggle with structured output.",
    ),
    "sqlcoder": ModelCapabilityProfile(
        family="sqlcoder",
        supports_tool_calling=False,
        capabilities={
            "code_python": 0.3, "code_rust": 0.1, "code_sql": 0.95,
            "reasoning": 0.3, "web_comprehension": 0.0,
            "summarisation": 0.2, "instruction_following": 0.3,
            "structured_output": 0.2,
        },
    ),
    "deepseek-coder": ModelCapabilityProfile(
        family="deepseek-coder",
        supports_tool_calling=False,
        capabilities={
            "code_python": 0.9, "code_rust": 0.7, "code_sql": 0.7,
            "reasoning": 0.4, "web_comprehension": 0.1,
            "summarisation": 0.3, "instruction_following": 0.4,
            "structured_output": 0.3,
        },
    ),
    "codellama": ModelCapabilityProfile(
        family="codellama",
        supports_tool_calling=False,
        capabilities={
            "code_python": 0.85, "code_rust": 0.6, "code_sql": 0.6,
            "reasoning": 0.3, "web_comprehension": 0.0,
            "summarisation": 0.2, "instruction_following": 0.3,
            "structured_output": 0.3,
        },
    ),
    "gemma2": ModelCapabilityProfile(
        family="gemma2",
        supports_tool_calling=True,
        capabilities={
            "code_python": 0.5, "code_rust": 0.3, "code_sql": 0.4,
            "reasoning": 0.6, "web_comprehension": 0.5,
            "summarisation": 0.6, "instruction_following": 0.6,
            "structured_output": 0.6,
        },
    ),
    "starcoder": ModelCapabilityProfile(
        family="starcoder",
        supports_tool_calling=False,
        capabilities={
            "code_python": 0.8, "code_rust": 0.5, "code_sql": 0.6,
            "reasoning": 0.2, "web_comprehension": 0.0,
            "summarisation": 0.1, "instruction_following": 0.2,
            "structured_output": 0.2,
        },
    ),
    "starcoder2": ModelCapabilityProfile(
        family="starcoder2",
        supports_tool_calling=False,
        capabilities={
            "code_python": 0.85, "code_rust": 0.6, "code_sql": 0.65,
            "reasoning": 0.3, "web_comprehension": 0.0,
            "summarisation": 0.2, "instruction_following": 0.3,
            "structured_output": 0.2,
        },
    ),
    "stable-code": ModelCapabilityProfile(
        family="stable-code",
        supports_tool_calling=False,
        capabilities={
            "code_python": 0.7, "code_rust": 0.4, "code_sql": 0.5,
            "reasoning": 0.3, "web_comprehension": 0.0,
            "summarisation": 0.2, "instruction_following": 0.3,
            "structured_output": 0.2,
        },
    ),
    "magicoder": ModelCapabilityProfile(
        family="magicoder",
        supports_tool_calling=False,
        capabilities={
            "code_python": 0.8, "code_rust": 0.5, "code_sql": 0.5,
            "reasoning": 0.3, "web_comprehension": 0.0,
            "summarisation": 0.2, "instruction_following": 0.3,
            "structured_output": 0.2,
        },
    ),
    "phind-codellama": ModelCapabilityProfile(
        family="phind-codellama",
        supports_tool_calling=False,
        capabilities={
            "code_python": 0.85, "code_rust": 0.6, "code_sql": 0.6,
            "reasoning": 0.4, "web_comprehension": 0.1,
            "summarisation": 0.3, "instruction_following": 0.4,
            "structured_output": 0.3,
        },
    ),
    "wizardcoder": ModelCapabilityProfile(
        family="wizardcoder",
        supports_tool_calling=False,
        capabilities={
            "code_python": 0.8, "code_rust": 0.5, "code_sql": 0.5,
            "reasoning": 0.3, "web_comprehension": 0.0,
            "summarisation": 0.2, "instruction_following": 0.3,
            "structured_output": 0.2,
        },
    ),
}

_DEFAULT_PROFILE = ModelCapabilityProfile(
    family="unknown",
    supports_tool_calling=True,
    capabilities={
        "code_python": 0.5, "code_rust": 0.3, "code_sql": 0.4,
        "reasoning": 0.5, "web_comprehension": 0.5,
        "summarisation": 0.5, "instruction_following": 0.5,
        "structured_output": 0.5,
    },
)


_VISION_SUFFIXES = ("-vl", "-vision", "-v")


def _is_vision_model(model_name: str) -> bool:
    """Return True if the model name suggests a vision/multimodal variant."""
    base = model_name.split(":")[0].lower().strip()
    return any(base.endswith(s) for s in _VISION_SUFFIXES)


def _extract_family(model_name: str) -> str:
    """Extract the family prefix from a model name.

    Examples::

        "qwen2.5:7b" → "qwen2.5"
        "qwen2.5-coder:14b" → "qwen2.5-coder"
        "deepseek-r1-distill-qwen-14b" → "deepseek-r1-distill"
        "phi-4-mini:latest" → "phi-4-mini"
    """
    base = model_name.split(":")[0].lower().strip()
    # Remove trailing size suffixes like "-14b", "-7b" etc.
    base = re.sub(r"-\d+(\.\d+)?[bBmMkK]$", "", base)
    return base


def get_profile(
    model_name: str,
    overrides: Optional[Dict[str, ModelCapabilityProfile]] = None,
) -> ModelCapabilityProfile:
    """Look up the capability profile for a model name.

    Resolution order:
    1. User-supplied overrides (exact family match).
    2. Builtin profiles (longest prefix match to handle families like
       ``qwen2.5-coder`` vs ``qwen2.5``).
    3. Default permissive profile.
    """
    family = _extract_family(model_name)

    if overrides:
        if family in overrides:
            return overrides[family]

    # Longest prefix match: "qwen2.5-coder" should match before "qwen2.5"
    best_match: Optional[str] = None
    for profile_family in BUILTIN_PROFILES:
        if family.startswith(profile_family):
            if best_match is None or len(profile_family) > len(best_match):
                best_match = profile_family

    if best_match is not None:
        return BUILTIN_PROFILES[best_match]

    return _DEFAULT_PROFILE


def match_models(
    available_models: List[str],
    required_capabilities: Optional[Dict[str, float]] = None,
    *,
    require_tool_calling: bool = True,
    exclude_models: Optional[List[str]] = None,
    prefer_smaller: bool = False,
    overrides: Optional[Dict[str, ModelCapabilityProfile]] = None,
) -> Optional[str]:
    """Select the best model from *available_models* based on capability requirements.

    Args:
        available_models: Model names to choose from.
        required_capabilities: Capability name → minimum score threshold.
            Models that don't meet all thresholds are deprioritised (not excluded,
            since we may have no perfect match).
        require_tool_calling: When True (default), exclude models that don't
            support tool calling.
        exclude_models: Model names to skip.
        prefer_smaller: When True, prefer smaller models (useful for reviewers).
        overrides: User-supplied profile overrides.

    Returns:
        Best matching model name, or None if available_models is empty.
    """
    if not available_models:
        return None

    excluded = set(exclude_models or [])
    candidates = [m for m in available_models if m not in excluded]
    if not candidates:
        return None

    if require_tool_calling:
        tc_candidates = [
            m for m in candidates
            if get_profile(m, overrides).supports_tool_calling
            and not _is_vision_model(m)
        ]
        if tc_candidates:
            candidates = tc_candidates

    def _capability_score(model_name: str) -> float:
        """Sum of how well the model meets the required capability thresholds."""
        if not required_capabilities:
            return 0.0
        profile = get_profile(model_name, overrides)
        score = 0.0
        for cap, threshold in required_capabilities.items():
            model_score = profile.capabilities.get(cap, 0.0)
            if model_score >= threshold:
                score += model_score
            else:
                score += model_score - threshold  # negative penalty
        return score

    def _model_size(model_name: str) -> float:
        """Extract approximate parameter size for tie-breaking."""
        m = re.search(r"[:\-]([\d.]+)\s*([bBmMkK])", model_name)
        if not m:
            return 999.0
        val = float(m.group(1))
        unit = m.group(2).upper()
        if unit == "K":
            val /= 1_000_000.0
        elif unit == "M":
            val /= 1_000.0
        return val

    # Sort by capability score (desc), then by size (asc for prefer_smaller, desc otherwise)
    size_sign = 1 if prefer_smaller else -1

    ranked = sorted(
        candidates,
        key=lambda m: (-_capability_score(m), size_sign * _model_size(m)),
    )
    return ranked[0] if ranked else None


# ---------------------------------------------------------------------------
# Capability → tool composition (ADR-028)
# ---------------------------------------------------------------------------

# Tools every specialist gets regardless of capabilities.
_BASE_TOOLS: List[str] = ["read_file", "list_files"]

# Capability → additional tools required to exercise that capability.
# Model-only capabilities (reasoning, summarisation, instruction_following)
# map to empty lists — they affect model selection, not tool selection.
_CAPABILITY_TOOLS: Dict[str, List[str]] = {
    "web_comprehension": ["web_search", "fetch_url"],
    "code_python": ["shell", "write_file", "run_tests"],
    "code_rust": ["shell", "write_file", "run_tests"],
    "code_sql": ["shell", "write_file"],
    "reasoning": [],
    "summarisation": [],
    "structured_output": [],  # model capability (JSON schema adherence), not tool
    "instruction_following": [],
}

# Capability → mission fragment for generating role descriptions.
_CAPABILITY_MISSION: Dict[str, str] = {
    "web_comprehension": "find accurate, up-to-date information using web search and URL fetching",
    "code_python": "write, debug, and test Python code",
    "code_rust": "write and test Rust code",
    "code_sql": "write and verify SQL queries",
    "reasoning": "apply careful logical reasoning to complex problems",
    "summarisation": "synthesise information into clear, concise summaries",
    "structured_output": "produce well-structured output following specified formats",
    "instruction_following": "follow instructions precisely and completely",
}


def compose_tools_from_capabilities(
    required_capabilities: List[str],
) -> List[str]:
    """Compose a tool set from requested capabilities.

    Returns a deduplicated, sorted list of tool names.  Every specialist
    gets the base tools (read_file, list_files); additional tools are
    added based on the capability → tools mapping.

    Unknown capabilities are silently ignored (they may still affect
    model selection via ``match_models()``).
    """
    tools = set(_BASE_TOOLS)
    for cap in required_capabilities:
        tools.update(_CAPABILITY_TOOLS.get(cap, []))
    return sorted(tools)


def generate_role_from_capabilities(
    required_capabilities: List[str],
) -> str:
    """Generate a role description from requested capabilities.

    Composes mission fragments for each known capability into a coherent
    role description.  Falls back to a generic description when no
    fragments match.
    """
    fragments = [
        _CAPABILITY_MISSION[c]
        for c in required_capabilities
        if c in _CAPABILITY_MISSION
    ]
    if not fragments:
        return (
            "You are an autonomous specialist agent. Complete the given task "
            "using the available tools.\n\n"
            "## Workflow\n"
            "1. Understand the task.\n"
            "2. Use available tools to complete it.\n"
            "3. Verify your work.\n"
            "4. Call finish_task when complete with a clear summary."
        )

    mission = "; ".join(fragments)
    return (
        f"You are an autonomous specialist agent. Your mission is to {mission}.\n\n"
        "## Workflow\n"
        "1. Understand the task.\n"
        "2. Use available tools to complete it.\n"
        "3. Verify your work.\n"
        "4. Call finish_task when complete with a clear summary.\n\n"
        "## Important\n"
        "- Answer the ACTUAL question asked. Do not substitute a different question.\n"
        "- Base your answer ONLY on information from tool results. Never fabricate.\n"
        "- Be concise. A simple question deserves a simple answer.\n"
        "- If you cannot find the answer, say so honestly — do not make one up."
    )


def infer_finish_schema_from_capabilities(
    required_capabilities: List[str],
) -> Optional[str]:
    """Infer the best finish schema key from capabilities.

    Returns a schema key (e.g. ``"code"``, ``"quick_answer"``) or None
    when the capabilities are ambiguous (caller should use the template
    default or ``"general"``).
    """
    cap_set = set(required_capabilities)
    has_code = any(c.startswith("code_") for c in cap_set)
    has_web = "web_comprehension" in cap_set

    if has_code and not has_web:
        return "code"
    if has_web and not has_code:
        return "quick_answer"
    # Mixed or unclear — let caller decide
    return None


# Capability requirements inferred from specialist templates / tool names.
_TEMPLATE_CAPABILITIES: Dict[str, Dict[str, float]] = {
    "engineering": {"code_python": 0.7, "structured_output": 0.6},
    "research": {"web_comprehension": 0.7, "summarisation": 0.6, "instruction_following": 0.7},
    "enterprise_research": {"web_comprehension": 0.6, "summarisation": 0.6, "instruction_following": 0.7},
}

# Tool name → capability inference for dynamic packs.
_TOOL_CAPABILITY_HINTS: Dict[str, Dict[str, float]] = {
    "shell": {"code_python": 0.5},
    "run_tests": {"code_python": 0.5, "structured_output": 0.5},
    "web_search": {"web_comprehension": 0.5},
    "fetch_url": {"web_comprehension": 0.5},
    "write_file": {"instruction_following": 0.5},
}

# Specialty → required capabilities for consult_specialist_model.
SPECIALTY_CAPABILITIES: Dict[str, Dict[str, float]] = {
    "code": {"code_python": 0.7},
    "sql": {"code_sql": 0.8},
    "reasoning": {"reasoning": 0.8},
}


def infer_task_capabilities(
    template_id: Optional[str] = None,
    tool_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Infer capability requirements from a template ID or tool names.

    Used for per-specialist model selection: the system matches these
    requirements against model profiles to pick the best model for the task.
    """
    if template_id and template_id in _TEMPLATE_CAPABILITIES:
        return dict(_TEMPLATE_CAPABILITIES[template_id])

    if tool_names:
        caps: Dict[str, float] = {}
        for tool in tool_names:
            for cap, score in _TOOL_CAPABILITY_HINTS.get(tool, {}).items():
                caps[cap] = max(caps.get(cap, 0.0), score)
        return caps

    return {}


def capability_summary(
    available_models: List[str],
    overrides: Optional[Dict[str, ModelCapabilityProfile]] = None,
) -> str:
    """Generate a human-readable capability summary for the orchestrator prompt.

    Lists each available model with its top capabilities scored above 0.6.
    """
    lines = []
    seen_families: set = set()
    for model in available_models:
        family = _extract_family(model)
        if family in seen_families:
            continue
        seen_families.add(family)
        profile = get_profile(model, overrides)
        strong_caps = [
            f"{cap}={score:.1f}"
            for cap, score in sorted(profile.capabilities.items(), key=lambda x: -x[1])
            if score >= 0.6
        ]
        tc = "tool-calling" if profile.supports_tool_calling else "no-tool-calling"
        caps_str = ", ".join(strong_caps) if strong_caps else "general"
        lines.append(f"- {family}: [{tc}] {caps_str}")
    return "\n".join(lines) if lines else "No model profiles available."
