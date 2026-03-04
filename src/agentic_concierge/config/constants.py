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

# Default base URLs for local LLM backends (OpenAI-compatible endpoints).
DEFAULT_BACKEND_URLS: dict[str, str] = {
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
}

# ---------------------------------------------------------------------------
# Nano / in-process model
# ---------------------------------------------------------------------------

# Default GGUF model for the inprocess backend (mistral.rs).
NANO_GGUF_FILENAME: str = "qwen2.5-3b-instruct-q4_k_m.gguf"
NANO_GGUF_URL: str = (
    "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/"
    "qwen2.5-3b-instruct-q4_k_m.gguf"
)
