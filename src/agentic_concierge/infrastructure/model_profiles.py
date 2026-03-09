"""Model capability profiles: scored capabilities per model family.

Profiles are loaded from externalized YAML configuration
(``config/defaults/model_profiles.yaml``) with user overrides from
``~/.config/concierge/model_profiles.yaml``.  This replaces the former
hardcoded ``BUILTIN_PROFILES`` dict and the deprecated
``_TOOL_INCAPABLE_NAMES`` blocklist.

Unknown model families get a permissive default profile.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


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
# Profile loading from YAML (with lazy initialisation)
# ---------------------------------------------------------------------------

# Module-level cache: populated on first call to _loaded_profiles().
_profiles_cache: Optional[Dict[str, ModelCapabilityProfile]] = None
_default_profile_cache: Optional[ModelCapabilityProfile] = None


def _profile_from_dict(family: str, data: Dict) -> ModelCapabilityProfile:
    """Construct a ``ModelCapabilityProfile`` from a YAML dict entry."""
    return ModelCapabilityProfile(
        family=family,
        supports_tool_calling=bool(data.get("supports_tool_calling", True)),
        capabilities={
            str(k): float(v)
            for k, v in data.get("capabilities", {}).items()
        },
        min_size_b=float(data.get("min_size_b", 0.0)),
        max_size_b=float(data.get("max_size_b", 999.0)),
        notes=str(data.get("notes", "")),
    )


def _load_profiles() -> tuple[Dict[str, ModelCapabilityProfile], ModelCapabilityProfile]:
    """Load profiles from YAML config hierarchy.

    Returns (profiles_dict, default_profile).
    """
    from agentic_concierge.config.external import load_yaml_config

    data = load_yaml_config("model_profiles.yaml")
    profiles: Dict[str, ModelCapabilityProfile] = {}

    for family, entry in data.get("profiles", {}).items():
        if isinstance(entry, dict):
            profiles[family] = _profile_from_dict(family, entry)

    # Default profile for unknown families
    default_data = data.get("default_profile", {})
    if default_data and isinstance(default_data, dict):
        default_profile = _profile_from_dict("unknown", default_data)
    else:
        # Hardcoded fallback if YAML is missing entirely
        default_profile = ModelCapabilityProfile(
            family="unknown",
            supports_tool_calling=True,
            capabilities={
                "code_python": 0.5, "code_rust": 0.3, "code_sql": 0.4,
                "reasoning": 0.5, "web_comprehension": 0.5,
                "summarisation": 0.5, "instruction_following": 0.5,
                "structured_output": 0.5,
            },
        )

    return profiles, default_profile


def _loaded_profiles() -> Dict[str, ModelCapabilityProfile]:
    """Return the lazily-loaded profiles dict."""
    global _profiles_cache, _default_profile_cache
    if _profiles_cache is None:
        _profiles_cache, _default_profile_cache = _load_profiles()
    return _profiles_cache


def _loaded_default_profile() -> ModelCapabilityProfile:
    """Return the lazily-loaded default profile."""
    global _profiles_cache, _default_profile_cache
    if _default_profile_cache is None:
        _profiles_cache, _default_profile_cache = _load_profiles()
    return _default_profile_cache


def reload_profiles() -> None:
    """Force re-read of profiles from YAML.  Useful after config changes in tests."""
    global _profiles_cache, _default_profile_cache
    _profiles_cache = None
    _default_profile_cache = None


# Public alias for backward compatibility — code that imports BUILTIN_PROFILES
# will get a dict populated from YAML.  Note: this is a snapshot at import time;
# code should prefer calling _loaded_profiles() for live data.
@property  # type: ignore[misc]
def _builtin_profiles_property() -> Dict[str, ModelCapabilityProfile]:
    return _loaded_profiles()


class _BuiltinProfilesAccessor:
    """Lazy accessor that behaves like a dict but loads from YAML on first access.

    This allows ``BUILTIN_PROFILES["qwen2.5"]`` to work while loading profiles
    lazily from YAML.  It also supports iteration, ``in``, ``len()``, etc.
    """

    def __getitem__(self, key: str) -> ModelCapabilityProfile:
        return _loaded_profiles()[key]

    def __contains__(self, key: object) -> bool:
        return key in _loaded_profiles()

    def __iter__(self):
        return iter(_loaded_profiles())

    def __len__(self) -> int:
        return len(_loaded_profiles())

    def keys(self):
        return _loaded_profiles().keys()

    def values(self):
        return _loaded_profiles().values()

    def items(self):
        return _loaded_profiles().items()

    def get(self, key: str, default=None):
        return _loaded_profiles().get(key, default)

    def __repr__(self) -> str:
        return repr(_loaded_profiles())


BUILTIN_PROFILES: _BuiltinProfilesAccessor = _BuiltinProfilesAccessor()  # type: ignore[assignment]


# Backward-compatible alias — code that imports _DEFAULT_PROFILE will get
# a lazy reference to the default profile loaded from YAML.
class _DefaultProfileAccessor:
    """Lazy accessor that returns the YAML-loaded default profile."""

    @property
    def family(self) -> str:
        return _loaded_default_profile().family

    @property
    def supports_tool_calling(self) -> bool:
        return _loaded_default_profile().supports_tool_calling

    @property
    def capabilities(self) -> Dict[str, float]:
        return _loaded_default_profile().capabilities

    @property
    def min_size_b(self) -> float:
        return _loaded_default_profile().min_size_b

    @property
    def max_size_b(self) -> float:
        return _loaded_default_profile().max_size_b

    @property
    def notes(self) -> str:
        return _loaded_default_profile().notes


_DEFAULT_PROFILE: _DefaultProfileAccessor = _DefaultProfileAccessor()  # type: ignore[assignment]


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
    1. User-supplied code overrides (exact family match).
    2. YAML-loaded profiles (longest prefix match to handle families like
       ``qwen2.5-coder`` vs ``qwen2.5``).
    3. Default permissive profile.
    """
    family = _extract_family(model_name)

    if overrides:
        if family in overrides:
            return overrides[family]

    # Longest prefix match: "qwen2.5-coder" should match before "qwen2.5"
    profiles = _loaded_profiles()
    best_match: Optional[str] = None
    for profile_family in profiles:
        if family.startswith(profile_family):
            if best_match is None or len(profile_family) > len(best_match):
                best_match = profile_family

    if best_match is not None:
        return profiles[best_match]

    return _loaded_default_profile()


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
KNOWN_CAPABILITIES: frozenset = frozenset(_CAPABILITY_TOOLS.keys())

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
