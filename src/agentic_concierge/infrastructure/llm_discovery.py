"""
Discover available LLM backends and models, select the best match, and optionally
auto-pull a default model so one command works without manual setup.

Strategy: try the configured backend first, then fall back through the profile's
``BACKEND_PRIORITY`` list so the system always finds *something* that works.
"""

from __future__ import annotations

import logging
import math
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

from agentic_concierge.config import ConciergeConfig, ModelConfig
from agentic_concierge.config.constants import (
    DEFAULT_BACKEND_URLS,
    LLM_DISCOVERY_TIMEOUT_S,
    LLM_PULL_TIMEOUT_S,
    NANO_GGUF_FILENAME,
)

logger = logging.getLogger(__name__)


@dataclass
class ResolvedLLM:
    """Result of resolving which backend and model to use."""
    base_url: str
    model: str
    model_config: ModelConfig  # full config (temperature, etc.) to use for chat
    warnings: list[str] = field(default_factory=list)
    fallback_used: bool = False
    resolved_backend: str = ""  # "ollama", "vllm", "inprocess", "cloud"
    available_models: list[str] = field(default_factory=list)
    all_chat_models: list[str] = field(default_factory=list)  # includes non-tool-calling models


def _ollama_root(base_url: str) -> str:
    """Ollama API root: strip /v1 from base_url (e.g. http://localhost:11434/v1 -> http://localhost:11434)."""
    u = urlparse(base_url)
    path = u.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3] or "/"
    return f"{u.scheme}://{u.netloc}{path}".rstrip("/") or base_url


def discover_ollama_models(base_url: str, timeout_s: float = LLM_DISCOVERY_TIMEOUT_S) -> list[dict] | None:
    """
    Query Ollama for available models (GET /api/tags). Returns list of model dicts
    (name, details.parameter_size, etc.) or None if not Ollama / unreachable.
    """
    root = _ollama_root(base_url)
    try:
        with httpx.Client(timeout=timeout_s) as client:
            r = client.get(f"{root}/api/tags")
            if r.status_code != 200:
                return None
            data = r.json()
            return data.get("models") or []
    except Exception:
        return None


def discover_openai_models(base_url: str, timeout_s: float = LLM_DISCOVERY_TIMEOUT_S) -> list[str] | None:
    """
    Query an OpenAI-compatible /v1/models endpoint (e.g. vLLM). Returns list of model ids or None.
    """
    url = f"{base_url.rstrip('/')}/models"
    try:
        with httpx.Client(timeout=timeout_s) as client:
            r = client.get(url)
            if r.status_code != 200:
                return None
            data = r.json()
            # OpenAI format: { "data": [ { "id": "..." }, ... ] }
            items = data.get("data") or []
            return [m.get("id") for m in items if m.get("id")]
    except Exception:
        return None


def _ollama_model_name(m: dict) -> str:
    """Preferred name for an Ollama model entry."""
    return m.get("name") or m.get("model") or ""


# Embedding-only models (Ollama returns them in /api/tags but they "do not support chat")
_EMBEDDING_ONLY_FAMILIES_OR_NAMES = frozenset(
    s.lower()
    for s in (
        "bge",
        "bge-m3",
        "nomic-embed",
        "mxbai-embed",
        "snowflake-arctic-embed",
        "all-minilm",
        "nomic-embed-text",
        "multilingual-e5",
    )
)


def _is_ollama_chat_capable(m: dict) -> bool:
    """True if this Ollama model is suitable for chat/completion (exclude embedding-only)."""
    name = (_ollama_model_name(m) or "").lower()
    details = m.get("details") or {}
    family = (details.get("family") or "").lower()
    families = [f.lower() for f in details.get("families") or []]
    for prefix in _EMBEDDING_ONLY_FAMILIES_OR_NAMES:
        if name.startswith(prefix) or family.startswith(prefix) or any(f.startswith(prefix) for f in families):
            return False
    if "embed" in name and "chat" not in name and "instruct" not in name:
        return False
    return True


def _is_tool_capable(m: dict) -> bool:
    """True if this Ollama model supports OpenAI-style tool calling.

    Delegates to ``model_profiles.get_profile()`` so that new model families
    are handled via the capability registry rather than a static blocklist.
    """
    if not _is_ollama_chat_capable(m):
        return False
    name = (_ollama_model_name(m) or "").lower()
    from agentic_concierge.infrastructure.model_profiles import get_profile
    profile = get_profile(name)
    return profile.supports_tool_calling


def _param_size_sort_key(name: str, details: dict | None) -> tuple:
    """Sort key: param size as float billions (smaller = faster / preferred fallback).

    Parses Ollama ``parameter_size`` strings such as ``"8.0B"``, ``"15B"``,
    ``"33.4B"``, ``"334M"`` into comparable floats.  Falls back to parsing
    the model name (e.g. ``qwen2.5:14b`` → 14.0).  Unknown size sorts last.
    """
    param = (details or {}).get("parameter_size") or ""
    m = re.match(r"([\d.]+)\s*([BbMmKk])", param.strip())
    if not m:
        # Try extracting size from model name (e.g. "qwen2.5:14b", "llama3.1:8b")
        m = re.search(r"[:\-]([\d.]+)\s*([BbMmKk])", name)
    if not m:
        return (999.0, name)  # unknown -> deprioritise
    val = float(m.group(1))
    unit = m.group(2).upper()
    if unit == "K":
        val /= 1_000_000.0
    elif unit == "M":
        val /= 1_000.0
    # "B" -> already in billions
    return (val, name)


def _model_family(name: str) -> str:
    """Extract family prefix from model name (e.g. 'qwen2.5:14b' → 'qwen2.5')."""
    # Strip tag (everything after ':')
    base = name.split(":")[0].lower()
    return base


def select_model(
    preferred_model: str,
    available: list[dict] | list[str],
    *,
    is_ollama: bool = True,
) -> str | None:
    """
    Select best model from available list.

    Priority: exact match → same-family by closest size → any family by closest size.
    Filters out models known not to support tool calling.
    Returns None if available is empty.
    """
    if not available:
        return None
    if is_ollama:
        # Filter to tool-capable models only
        tool_capable = [m for m in available if _is_tool_capable(m)]
        names = [n for n in (_ollama_model_name(m) for m in tool_capable) if n]
        if preferred_model in names:
            return preferred_model
        # Build size index
        name_to_entry = {_ollama_model_name(m): m for m in tool_capable}
        details_by_name = {n: (name_to_entry.get(n) or {}).get("details") for n in names}
        preferred_size = _param_size_sort_key(preferred_model, None)[0]
        preferred_family = _model_family(preferred_model)

        def _size_candidates(model_names: list[str]) -> list[str]:
            """Sort by log-scale distance from preferred size; break ties by preferring smaller.

            Log-scale (ratio) correctly handles the asymmetry: 0.5b is 28x
            away from 14b while 32b is only 2.3x away.  Absolute distance
            would incorrectly prefer the tiny (useless) model.
            """
            def _log_dist(size: float) -> float:
                if size <= 0 or preferred_size <= 0:
                    return 999.0
                return abs(math.log(size / preferred_size))

            sized = [(n, _param_size_sort_key(n, details_by_name.get(n))[0]) for n in model_names]
            return [n for n, _ in sorted(sized, key=lambda t: (_log_dist(t[1]), t[1]))]

        # Prefer same-family models
        same_family = [n for n in names if _model_family(n) == preferred_family]
        if same_family:
            candidates = _size_candidates(same_family)
            if candidates:
                return candidates[0]
        # Fall back to any tool-capable model
        candidates = _size_candidates(names)
        return candidates[0] if candidates else None
    else:
        # OpenAI/vLLM: list of id strings
        ids = [s for s in available if isinstance(s, str) and s]
        if preferred_model in ids:
            return preferred_model
        return ids[0] if ids else None


def _scale_timeout(selected: str, model_cfg: ModelConfig, warnings: list[str]) -> float:
    """Scale timeout proportionally when *selected* is larger than the preferred model."""
    selected_size = _param_size_sort_key(selected, None)[0]
    preferred_size = _param_size_sort_key(model_cfg.model, None)[0]
    if selected_size > preferred_size > 0:
        scale = selected_size / preferred_size
        timeout = min(model_cfg.timeout_s * scale, 900.0)
        warnings.append(
            f"Using {selected} ({selected_size}B) instead of {model_cfg.model}; "
            f"timeout scaled to {timeout:.0f}s"
        )
        return timeout
    return model_cfg.timeout_s


def resolve_routing_model(resolved: ResolvedLLM, config: ConciergeConfig) -> str:
    """Pick the best model for routing from available models.

    Prefers the configured routing model if available.  Otherwise uses the
    resolved task model (routing needs reliable tool calling — using a tiny
    model like 0.5b causes misrouting).
    """
    routing_cfg = config.models.get(config.routing_model_key)
    if routing_cfg and routing_cfg.model in resolved.available_models:
        return routing_cfg.model
    return resolved.model


def pick_smaller_model(available_models: list[str], current_model: str) -> str | None:
    """Return the largest model smaller than *current_model*, or None."""
    current_size = _param_size_sort_key(current_model, None)[0]
    smaller = [
        (n, _param_size_sort_key(n, None)[0])
        for n in available_models
        if _param_size_sort_key(n, None)[0] < current_size and n != current_model
    ]
    if not smaller:
        return None
    smaller.sort(key=lambda t: -t[1])  # largest first
    return smaller[0][0]


def pick_larger_model(available_models: list[str], current_model: str) -> str | None:
    """Return the smallest model larger than *current_model*, or None.

    This is the upward counterpart to ``pick_smaller_model()``: used for
    adaptive escalation when a small model produces low-quality output.
    Same-family models are preferred; falls back to any available model.

    ``available_models`` is expected to be pre-filtered for tool capability
    (as populated by ``resolve_llm()``).
    """
    current_size = _param_size_sort_key(current_model, None)[0]
    if current_size >= 999.0:
        return None  # cannot escalate from an unparseable model
    current_family = _model_family(current_model)
    larger = [
        (n, s)
        for n in available_models
        if n != current_model
        for s in (_param_size_sort_key(n, None)[0],)
        if current_size < s < 999.0
    ]
    if not larger:
        return None
    # Prefer same-family, then smallest first (closest upgrade)
    same_family = [(n, s) for n, s in larger if _model_family(n) == current_family]
    if same_family:
        same_family.sort(key=lambda t: t[1])  # smallest first
        return same_family[0][0]
    larger.sort(key=lambda t: t[1])  # smallest first
    return larger[0][0]


def _ollama_pull(model: str, ollama_root: str, timeout_s: int = 600) -> bool:
    """Run ollama pull <model>. Assumes ollama CLI is on PATH. Returns True if pull succeeded."""
    try:
        subprocess.run(
            ["ollama", "pull", model],
            capture_output=True,
            timeout=timeout_s,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _nano_model_path() -> str | None:
    """Return path to the nano GGUF model file, or None if not present."""
    try:
        from platformdirs import user_data_path
        path = user_data_path("agentic-concierge") / "models" / NANO_GGUF_FILENAME
        return str(path) if path.exists() else None
    except Exception:
        return None


def _get_profile_tier(config: ConciergeConfig):
    """Load the current profile tier from detected.json or config, defaulting to MEDIUM."""
    from agentic_concierge.config.features import ProfileTier
    if config.profile != "auto":
        try:
            return ProfileTier(config.profile.lower())
        except ValueError:
            return ProfileTier.MEDIUM
    try:
        from agentic_concierge.bootstrap.detected import load_detected
        profile = load_detected()
        if profile is not None:
            return profile.tier
    except Exception:
        pass
    return ProfileTier.MEDIUM


def _try_ollama(
    base_url: str,
    model_cfg: ModelConfig,
    config: ConciergeConfig,
    *,
    auto_start: bool = False,
) -> ResolvedLLM | None:
    """Try to resolve via Ollama at *base_url*. Returns ResolvedLLM or None."""
    from agentic_concierge.infrastructure.llm_bootstrap import ensure_llm_available

    models = discover_ollama_models(base_url, timeout_s=15.0)

    # If unreachable and auto_start requested, try starting Ollama
    if models is None and auto_start and shutil.which("ollama"):
        logger.info("Ollama unreachable at %s; attempting auto-start", base_url)
        try:
            ensure_llm_available(base_url, start_cmd=["ollama", "serve"], timeout_s=30)
            models = discover_ollama_models(base_url, timeout_s=15.0)
        except (TimeoutError, FileNotFoundError):
            pass

    if models is None:
        return None

    models = [m for m in models if _is_ollama_chat_capable(m)]
    selected = select_model(model_cfg.model, models, is_ollama=True)

    if not selected and config.auto_pull_if_missing:
        pull_model = config.auto_pull_model
        logger.info("No models at %s; auto-pulling %s", base_url, pull_model)
        ollama_root = _ollama_root(base_url)
        if _ollama_pull(pull_model, ollama_root, timeout_s=LLM_PULL_TIMEOUT_S):
            time.sleep(1.0)
            models2 = discover_ollama_models(base_url, timeout_s=15.0)
            if models2:
                models2 = [m for m in models2 if _is_ollama_chat_capable(m)]
                selected = select_model(pull_model, models2, is_ollama=True) or (
                    _ollama_model_name(models2[0]) if models2 else None
                )

    if not selected:
        return None

    tool_capable_names = [n for n in (_ollama_model_name(m) for m in models if _is_tool_capable(m)) if n]
    all_chat_names = [n for n in (_ollama_model_name(m) for m in models if _is_ollama_chat_capable(m)) if n]
    resolved_config = ModelConfig(
        base_url=base_url,
        model=selected,
        backend="ollama",
        api_key=model_cfg.api_key,
        temperature=model_cfg.temperature,
        top_p=model_cfg.top_p,
        max_tokens=model_cfg.max_tokens,
        timeout_s=model_cfg.timeout_s,
    )
    return ResolvedLLM(
        base_url=base_url,
        model=selected,
        model_config=resolved_config,
        resolved_backend="ollama",
        available_models=tool_capable_names,
        all_chat_models=all_chat_names,
    )


def _try_vllm(base_url: str, model_cfg: ModelConfig) -> ResolvedLLM | None:
    """Try to resolve via a vLLM/OpenAI-compatible endpoint at *base_url*."""
    openai_models = discover_openai_models(base_url, timeout_s=LLM_DISCOVERY_TIMEOUT_S)
    if not openai_models:
        return None
    selected = select_model(model_cfg.model, openai_models, is_ollama=False)
    if not selected:
        return None
    resolved_config = ModelConfig(
        base_url=base_url,
        model=selected,
        backend="vllm",
        api_key=model_cfg.api_key,
        temperature=model_cfg.temperature,
        top_p=model_cfg.top_p,
        max_tokens=model_cfg.max_tokens,
        timeout_s=model_cfg.timeout_s,
    )
    return ResolvedLLM(
        base_url=base_url,
        model=selected,
        model_config=resolved_config,
        resolved_backend="vllm",
        available_models=[s for s in openai_models if isinstance(s, str) and s],
    )


def _try_inprocess(model_cfg: ModelConfig) -> ResolvedLLM | None:
    """Try to resolve via in-process inference (mistral.rs + GGUF)."""
    try:
        from agentic_concierge.infrastructure.chat.inprocess import is_available
        if not is_available():
            return None
    except ImportError:
        return None
    gguf = _nano_model_path()
    if not gguf:
        return None
    resolved_config = ModelConfig(
        base_url="",
        model=gguf,
        backend="inprocess",
        api_key="",
        temperature=model_cfg.temperature,
        top_p=model_cfg.top_p,
        max_tokens=model_cfg.max_tokens,
        timeout_s=model_cfg.timeout_s,
    )
    return ResolvedLLM(
        base_url="",
        model=gguf,
        model_config=resolved_config,
        resolved_backend="inprocess",
    )


def _try_cloud(config: ConciergeConfig, model_cfg: ModelConfig) -> ResolvedLLM | None:
    """Try to find a cloud (generic) backend in config.models with an api_key."""
    for key, mc in config.models.items():
        if mc.backend == "generic" and mc.api_key:
            resolved_config = ModelConfig(
                base_url=mc.base_url,
                model=mc.model,
                backend="generic",
                api_key=mc.api_key,
                temperature=mc.temperature or model_cfg.temperature,
                top_p=mc.top_p or model_cfg.top_p,
                max_tokens=mc.max_tokens or model_cfg.max_tokens,
                timeout_s=mc.timeout_s or model_cfg.timeout_s,
            )
            return ResolvedLLM(
                base_url=mc.base_url,
                model=mc.model,
                model_config=resolved_config,
                resolved_backend="cloud",
            )
    return None


def _try_backend(
    backend_name: str,
    model_cfg: ModelConfig,
    config: ConciergeConfig,
) -> ResolvedLLM | None:
    """Attempt a single backend. Returns ResolvedLLM or None."""
    if backend_name == "ollama":
        url = DEFAULT_BACKEND_URLS["ollama"]
        return _try_ollama(url, model_cfg, config, auto_start=True)
    elif backend_name == "vllm":
        url = DEFAULT_BACKEND_URLS["vllm"]
        return _try_vllm(url, model_cfg)
    elif backend_name == "inprocess":
        return _try_inprocess(model_cfg)
    elif backend_name == "cloud":
        return _try_cloud(config, model_cfg)
    return None


def resolve_llm(
    config: ConciergeConfig,
    model_key: str,
    *,
    ensure_available: bool | None = None,
    start_cmd: list[str] | None = None,
    timeout_s: int | None = None,
) -> ResolvedLLM:
    """
    Determine which backend and model to use so that one command works.

    1. Try the configured primary backend (same as before).
    2. On failure, iterate through ``BACKEND_PRIORITY[tier]`` filtered by
       enabled features, skipping the already-tried primary.
    3. Each backend is probed; the first to return models is used.
    4. If all fail, raise ``RuntimeError`` listing each attempt and why it failed.

    The signature and ``ResolvedLLM`` return type are unchanged for callers.
    """
    from agentic_concierge.infrastructure.llm_bootstrap import ensure_llm_available

    model_cfg = config.models.get(model_key) or config.models["quality"]
    ensure_avail = config.local_llm_ensure_available if ensure_available is None else ensure_available
    start_cmd_ = config.local_llm_start_cmd if start_cmd is None else start_cmd
    timeout_s_ = config.local_llm_start_timeout_s if timeout_s is None else timeout_s

    # -----------------------------------------------------------------------
    # Phase 1: try the configured primary backend (preserves original logic)
    # -----------------------------------------------------------------------
    primary_error: str | None = None

    if ensure_avail and start_cmd_:
        try:
            ensure_llm_available(
                model_cfg.base_url,
                start_cmd=start_cmd_,
                timeout_s=timeout_s_,
            )
        except (TimeoutError, FileNotFoundError) as e:
            primary_error = str(e)
            logger.warning("Primary backend start failed: %s", primary_error)

    if primary_error is None:
        # Discover: try Ollama first (default base_url is Ollama)
        models = discover_ollama_models(model_cfg.base_url, timeout_s=15.0)
        if models is not None:
            models = [m for m in models if _is_ollama_chat_capable(m)]
            names = [m for m in (_ollama_model_name(m) for m in models) if m]
            tool_capable_names = [n for n in (_ollama_model_name(m) for m in models if _is_tool_capable(m)) if n]
            selected = select_model(model_cfg.model, models, is_ollama=True)
            if selected:
                warnings: list[str] = []
                if selected != model_cfg.model:
                    logger.warning(
                        "Preferred model %r not available; using fallback: %s (available: %s)",
                        model_cfg.model, selected, names[:5],
                    )
                else:
                    logger.info("Resolved model: %s at %s", selected, model_cfg.base_url)
                timeout = model_cfg.timeout_s
                timeout = _scale_timeout(selected, model_cfg, warnings)
                resolved_config = ModelConfig(
                    base_url=model_cfg.base_url,
                    model=selected,
                    api_key=model_cfg.api_key,
                    temperature=model_cfg.temperature,
                    top_p=model_cfg.top_p,
                    max_tokens=model_cfg.max_tokens,
                    timeout_s=timeout,
                )
                return ResolvedLLM(
                    base_url=model_cfg.base_url,
                    model=selected,
                    model_config=resolved_config,
                    resolved_backend="ollama",
                    available_models=tool_capable_names,
                    all_chat_models=list(names),
                    warnings=warnings,
                )
            # No models: optional auto-pull
            if config.auto_pull_if_missing:
                pull_model = config.auto_pull_model
                ollama_root = _ollama_root(model_cfg.base_url)
                logger.info("No models at %s; auto-pulling %s", model_cfg.base_url, pull_model)
                if _ollama_pull(pull_model, ollama_root, timeout_s=LLM_PULL_TIMEOUT_S):
                    time.sleep(1.0)
                    models2 = discover_ollama_models(model_cfg.base_url, timeout_s=15.0)
                    if models2:
                        models2 = [m for m in models2 if _is_ollama_chat_capable(m)]
                    if models2:
                        selected = select_model(pull_model, models2, is_ollama=True) or _ollama_model_name(models2[0])
                        tc_names2 = [n for n in (_ollama_model_name(m) for m in models2 if _is_tool_capable(m)) if n]
                        all_names2 = [n for n in (_ollama_model_name(m) for m in models2) if n]
                        resolved_config = ModelConfig(
                            base_url=model_cfg.base_url,
                            model=selected,
                            api_key=model_cfg.api_key,
                            temperature=model_cfg.temperature,
                            top_p=model_cfg.top_p,
                            max_tokens=model_cfg.max_tokens,
                            timeout_s=model_cfg.timeout_s,
                        )
                        return ResolvedLLM(
                            base_url=model_cfg.base_url,
                            model=selected,
                            model_config=resolved_config,
                            resolved_backend="ollama",
                            available_models=tc_names2,
                            all_chat_models=all_names2,
                        )
            primary_error = f"No chat-capable models at {model_cfg.base_url}"
        else:
            # Not Ollama or unreachable: try OpenAI-compat (e.g. vLLM) at same base_url
            logger.info("Ollama not reachable at %s; trying OpenAI-compat endpoint", model_cfg.base_url)
            openai_models = discover_openai_models(model_cfg.base_url, timeout_s=LLM_DISCOVERY_TIMEOUT_S)
            if openai_models:
                selected = select_model(model_cfg.model, openai_models, is_ollama=False)
                if selected:
                    logger.info("Resolved model (OpenAI-compat): %s at %s", selected, model_cfg.base_url)
                    resolved_config = ModelConfig(
                        base_url=model_cfg.base_url,
                        model=selected,
                        api_key=model_cfg.api_key,
                        temperature=model_cfg.temperature,
                        top_p=model_cfg.top_p,
                        max_tokens=model_cfg.max_tokens,
                        timeout_s=model_cfg.timeout_s,
                    )
                    return ResolvedLLM(
                        base_url=model_cfg.base_url,
                        model=selected,
                        model_config=resolved_config,
                        resolved_backend="vllm",
                        available_models=[s for s in openai_models if isinstance(s, str) and s],
                    )
            primary_error = f"No backend reachable at {model_cfg.base_url}"

    # -----------------------------------------------------------------------
    # Phase 2: fallback chain — try BACKEND_PRIORITY backends
    # -----------------------------------------------------------------------
    from agentic_concierge.config.features import (
        BACKEND_PRIORITY,
        Feature,
        FeatureSet,
    )

    tier = _get_profile_tier(config)
    feature_set = FeatureSet.from_profile(tier, config.features)
    priority = BACKEND_PRIORITY.get(tier, ["ollama", "vllm", "inprocess", "cloud"])

    # Map backend names to Feature flags for filtering
    _BACKEND_FEATURE = {
        "ollama": Feature.OLLAMA,
        "vllm": Feature.VLLM,
        "inprocess": Feature.INPROCESS,
        "cloud": Feature.CLOUD,
    }

    # Determine which backend was already tried (by comparing base_url)
    tried_urls = {model_cfg.base_url}
    attempts: list[str] = [f"primary ({model_cfg.base_url}): {primary_error}"]

    for backend_name in priority:
        feat = _BACKEND_FEATURE.get(backend_name)
        if feat and not feature_set.is_enabled(feat):
            logger.debug("Skipping fallback %s (feature disabled)", backend_name)
            continue

        # Skip if the default URL for this backend was already tried as primary
        default_url = DEFAULT_BACKEND_URLS.get(backend_name, "")
        if backend_name in ("ollama", "vllm") and default_url in tried_urls:
            continue

        logger.info("Fallback: trying %s backend", backend_name)
        result = _try_backend(backend_name, model_cfg, config)
        if result is not None:
            result.fallback_used = True
            result.warnings.append(
                f"Primary backend failed ({primary_error}); "
                f"using fallback: {backend_name}"
            )
            logger.warning(
                "Using fallback backend %s (model=%s) because primary failed: %s",
                backend_name, result.model, primary_error,
            )
            return result
        attempts.append(f"{backend_name}: not available")

    # -----------------------------------------------------------------------
    # All failed
    # -----------------------------------------------------------------------
    attempts_str = "; ".join(attempts)
    raise RuntimeError(
        f"Could not resolve any LLM backend. Tried: {attempts_str}. "
        "Install Ollama (https://ollama.com) and run 'ollama pull <model>', "
        "or configure a cloud backend in your config file."
    )
