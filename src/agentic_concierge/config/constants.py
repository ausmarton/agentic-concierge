"""Named constants for values that appear in multiple places or need explanation.

Each constant has a comment explaining *why* the value is what it is, so future
maintainers can decide whether a change is safe without grepping for side-effects.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Output / runlog size limits
# ---------------------------------------------------------------------------

# Maximum characters kept from a single tool execution's stdout+stderr.
# Prevents a runaway shell command (e.g. `find /` or `cat big_file`) from
# allocating gigabytes of memory and writing an unreadable runlog.
MAX_TOOL_OUTPUT_CHARS: int = 50_000

# Maximum characters stored from an LLM response's text content in the runlog.
# The full content is passed to the conversation; only the runlog entry is capped
# to keep runlog.jsonl files scannable and prevent disk blow-up on verbose models.
MAX_LLM_CONTENT_IN_RUNLOG_CHARS: int = 2_000

# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------

# HTTP timeout for model-list discovery queries (Ollama /api/tags, vLLM /models).
# Kept short so startup is fast even when a backend is unreachable or slow.
# Raise this if your network introduces latency between the agent host and the
# LLM server (e.g. remote Ollama over VPN).
LLM_DISCOVERY_TIMEOUT_S: float = 10.0

# Default HTTP read timeout for a single LLM chat-completions call.
# This is the OllamaChatClient fallback; the real value comes from
# ModelConfig.timeout_s in config (which defaults to 360 s for large models).
# 120 s is conservative for small local models; increase ModelConfig.timeout_s
# in your config for slow or quantised models.
LLM_CHAT_DEFAULT_TIMEOUT_S: float = 120.0

# Default wall-clock timeout for a single shell command inside the sandbox.
# Covers most test runs, linters, and build steps.  LLM callers can override
# this per-call via the shell tool's timeout_s argument.
SHELL_DEFAULT_TIMEOUT_S: int = 120

# Maximum wall-clock time allowed for `ollama pull <model>`.
# Large models (e.g. 70B at 4-bit) can be 30–40 GB and may take 10+ minutes on
# a slow link.  600 s (10 min) is a reasonable upper bound; if pulls routinely
# exceed this, increase it or pull the model manually before running fabric.
LLM_PULL_TIMEOUT_S: int = 600

# ---------------------------------------------------------------------------
# Backend defaults
# ---------------------------------------------------------------------------

# Loaded lazily from config/defaults/backends.yaml with user override support.
# Hardcoded fallback values are used only if YAML loading fails entirely.

_FALLBACK_BACKEND_URLS: dict[str, str] = {
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
}

_cached_backend_urls: dict[str, str] | None = None


def _load_backend_urls() -> dict[str, str]:
    """Load backend URLs from backends.yaml ``backends`` section."""
    global _cached_backend_urls
    if _cached_backend_urls is not None:
        return _cached_backend_urls
    try:
        from agentic_concierge.config.external import load_yaml_config
        raw = load_yaml_config("backends.yaml")
        backends = raw.get("backends", {})
        if not isinstance(backends, dict):
            _cached_backend_urls = dict(_FALLBACK_BACKEND_URLS)
            return _cached_backend_urls
        urls: dict[str, str] = {}
        for name, cfg in backends.items():
            if isinstance(cfg, dict) and "url" in cfg:
                urls[str(name)] = str(cfg["url"])
        # Fill any missing backends from fallback.
        for name, url in _FALLBACK_BACKEND_URLS.items():
            if name not in urls:
                urls[name] = url
        _cached_backend_urls = urls
    except Exception:
        _cached_backend_urls = dict(_FALLBACK_BACKEND_URLS)
    return _cached_backend_urls


def reload_backend_config() -> None:
    """Clear cached backend config — forces re-load on next access."""
    global _cached_backend_urls
    _cached_backend_urls = None


def default_backend_urls() -> dict[str, str]:
    """Return backend name → base URL mapping, loaded from backends.yaml."""
    return dict(_load_backend_urls())


# ---------------------------------------------------------------------------
# Nano / in-process model
# ---------------------------------------------------------------------------

# Loaded lazily from config/defaults/models.yaml with user override support.
# Hardcoded fallback values are used only if YAML loading fails entirely.

_FALLBACK_NANO_GGUF_FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
_FALLBACK_NANO_GGUF_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/"
    "qwen2.5-3b-instruct-q4_k_m.gguf"
)

_cached_nano_config: dict | None = None


def _load_nano_config() -> dict:
    """Load nano section from models.yaml; return dict with gguf_filename/gguf_url."""
    global _cached_nano_config
    if _cached_nano_config is not None:
        return _cached_nano_config
    try:
        from agentic_concierge.config.external import load_yaml_config
        raw = load_yaml_config("models.yaml")
        nano = raw.get("nano", {})
        if not isinstance(nano, dict):
            nano = {}
        _cached_nano_config = nano
    except Exception:
        _cached_nano_config = {}
    return _cached_nano_config


def reload_nano_config() -> None:
    """Clear cached nano config — forces re-load on next access."""
    global _cached_nano_config
    _cached_nano_config = None


def nano_gguf_filename() -> str:
    """Return the nano GGUF model filename, loaded from models.yaml."""
    return _load_nano_config().get("gguf_filename", _FALLBACK_NANO_GGUF_FILENAME)


def nano_gguf_url() -> str:
    """Return the nano GGUF download URL, loaded from models.yaml."""
    return _load_nano_config().get("gguf_url", _FALLBACK_NANO_GGUF_URL)
