"""Feature flags: profile-based capabilities with zero-cost disabled features.

Each ``Feature`` maps to a capability that can be enabled or disabled.
Disabled features have zero resource cost — no imports, no processes, no RAM.

``ProfileTier`` is defined here (not in model_advisor) so that config
schemas can reference it without importing bootstrap code.
``bootstrap/model_advisor.py`` imports ``ProfileTier`` from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ProfileTier(str, Enum):
    """System profile tier, derived from available hardware resources."""

    NANO = "nano"      # < 8 GB RAM
    SMALL = "small"    # 8–16 GB RAM, VRAM < 4 GB
    MEDIUM = "medium"  # 16–32 GB RAM OR 4–12 GB VRAM
    LARGE = "large"    # 32–64 GB RAM OR 12–24 GB VRAM
    SERVER = "server"  # 64 GB+ RAM OR 24 GB+ VRAM OR multi-GPU (2+ devices)


class Feature(str, Enum):
    """Individual capability that can be enabled or disabled per profile.

    Disabled features are **gated at four levels**:
    1. Install-time: extras (e.g. ``[nano]``, ``[browser]``) guard optional deps.
    2. Import-time: lazy imports inside each feature's module.
    3. Config-time: ``FeatureSet.require()`` before instantiating any resource.
    4. Process-time: no subprocess/model spawning for disabled features.
    """

    INPROCESS = "inprocess"   # in-process inference via mistral.rs
    OLLAMA = "ollama"         # Ollama local LLM server
    VLLM = "vllm"             # vLLM high-throughput server
    CLOUD = "cloud"           # cloud LLM endpoints (OpenAI, Anthropic, etc.)
    MCP = "mcp"               # MCP tool servers
    BROWSER = "browser"       # headless browser tool (Playwright)
    EMBEDDING = "embedding"   # vector embeddings for run index
    TELEMETRY = "telemetry"   # OpenTelemetry tracing
    CONTAINER = "container"   # containerised tool execution (Podman)


# Default features enabled per profile tier.
# Server profile drops Ollama (vLLM handles all throughput) and adds Telemetry.
PROFILE_FEATURES: dict[ProfileTier, frozenset[Feature]] = {
    ProfileTier.NANO: frozenset({
        Feature.INPROCESS,
        Feature.CLOUD,
    }),
    ProfileTier.SMALL: frozenset({
        Feature.INPROCESS,
        Feature.OLLAMA,
        Feature.CLOUD,
        Feature.MCP,
        Feature.BROWSER,
    }),
    ProfileTier.MEDIUM: frozenset({
        Feature.INPROCESS,
        Feature.OLLAMA,
        Feature.VLLM,
        Feature.CLOUD,
        Feature.MCP,
        Feature.EMBEDDING,
        Feature.BROWSER,
    }),
    ProfileTier.LARGE: frozenset({
        Feature.INPROCESS,
        Feature.OLLAMA,
        Feature.VLLM,
        Feature.CLOUD,
        Feature.MCP,
        Feature.EMBEDDING,
        Feature.CONTAINER,
        Feature.BROWSER,
    }),
    ProfileTier.SERVER: frozenset({
        Feature.INPROCESS,
        Feature.OLLAMA,
        Feature.VLLM,
        Feature.CLOUD,
        Feature.MCP,
        Feature.EMBEDDING,
        Feature.CONTAINER,
        Feature.TELEMETRY,
        Feature.BROWSER,
    }),
}

# Backend resolution priority per profile tier.
# resolve_llm() tries backends in this order when the configured primary fails.
BACKEND_PRIORITY: dict[ProfileTier, list[str]] = {
    ProfileTier.NANO: ["inprocess", "ollama", "cloud"],
    ProfileTier.SMALL: ["ollama", "inprocess", "cloud"],
    ProfileTier.MEDIUM: ["ollama", "vllm", "inprocess", "cloud"],
    ProfileTier.LARGE: ["vllm", "ollama", "inprocess", "cloud"],
    ProfileTier.SERVER: ["vllm", "ollama", "inprocess", "cloud"],
}


class FeatureDisabledError(RuntimeError):
    """Raised when code attempts to use a feature disabled for the current profile.

    Attributes:
        feature: The ``Feature`` that was attempted.
        hint: Human-readable suggestion for how to enable the feature.
    """

    def __init__(self, feature: Feature, hint: str = "") -> None:
        self.feature = feature
        self.hint = hint
        msg = f"Feature '{feature.value}' is disabled for the current profile."
        if hint:
            msg += f" {hint}"
        super().__init__(msg)


@dataclass
class FeatureSet:
    """The set of features enabled for the current session.

    Built from a ``ProfileTier`` (which sets defaults) plus explicit user
    overrides from ``FeaturesConfig`` (``None`` = use profile default,
    ``True`` = force enable, ``False`` = force disable).

    Usage::

        feature_set = FeatureSet.from_profile(tier, config.features)
        feature_set.require(Feature.VLLM, "Set vllm: true in config.features.")
    """

    enabled: frozenset[Feature]

    def is_enabled(self, f: Feature) -> bool:
        """Return ``True`` if *f* is enabled in this feature set."""
        return f in self.enabled

    def require(self, f: Feature, hint: str = "") -> None:
        """Raise ``FeatureDisabledError`` if *f* is disabled.

        Call this at the top of any code path that depends on a feature, to
        fail fast with a clear error rather than an ``ImportError`` or
        ``AttributeError`` deep in the stack.
        """
        if f not in self.enabled:
            raise FeatureDisabledError(f, hint)

    @classmethod
    def from_profile(cls, tier: ProfileTier, overrides: Any) -> "FeatureSet":
        """Build a ``FeatureSet`` from the profile tier with user overrides applied.

        *overrides* is a ``FeaturesConfig``-compatible object (any object
        with ``Optional[bool]`` attributes named after ``Feature`` values).
        ``None`` means "use profile default"; ``True`` forces enable;
        ``False`` forces disable.
        """
        base = set(PROFILE_FEATURES.get(tier, frozenset()))
        for feature in Feature:
            override_val = getattr(overrides, feature.value, None)
            if override_val is True:
                base.add(feature)
            elif override_val is False:
                base.discard(feature)
        return cls(enabled=frozenset(base))

    @classmethod
    def all_enabled(cls) -> "FeatureSet":
        """Return a ``FeatureSet`` with every feature enabled (useful for tests)."""
        return cls(enabled=frozenset(Feature))
