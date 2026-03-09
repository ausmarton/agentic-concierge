"""Capability probing for unknown model families.

When a model has no profile (Tier 4 fallback), run micro-prompt probes
to determine its actual capabilities.  Results are cached to disk to
avoid re-probing.

Each probe is a single chat completion with a tiny prompt (< 50 tokens).
Currently probes for:

- **tool_calling**: Can the model produce valid function calls?
- **structured_output**: Can it produce valid JSON on request?
- **instruction_following**: Does it follow a specific format constraint?

See DESIGN_V2.md §11.4 for rationale.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from agentic_concierge.application.ports import ChatClient
from agentic_concierge.domain import LLMResponse

logger = logging.getLogger(__name__)

# Default cache file location (XDG data home).
_DEFAULT_CACHE_DIR = os.path.join(
    os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")),
    "concierge",
)
_CACHE_FILENAME = "model_probes.json"


# ---------------------------------------------------------------------------
# Individual probe functions
# ---------------------------------------------------------------------------


async def _probe_tool_calling(
    client: ChatClient,
    model: str,
) -> float:
    """Probe whether the model can produce a valid tool call.

    Sends a chat with a simple tool definition and checks if the response
    contains a tool_calls entry.  Returns 1.0 if successful, 0.0 otherwise.
    """
    tool_def = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    }
    messages = [{"role": "user", "content": "What is the weather in London?"}]
    try:
        resp: LLMResponse = await client.chat(
            messages, model, tools=[tool_def], max_tokens=100,
        )
        if resp.tool_calls:
            return 1.0
    except Exception:
        logger.debug("tool_calling probe failed for %s", model, exc_info=True)
    return 0.0


async def _probe_structured_output(
    client: ChatClient,
    model: str,
) -> float:
    """Probe whether the model can produce valid JSON.

    Asks for a simple JSON object and checks if the response parses.
    Returns 1.0 for valid JSON, 0.5 for partial, 0.0 for failure.
    """
    messages = [
        {
            "role": "user",
            "content": (
                'Respond with ONLY a JSON object: {"name": "Alice", "age": 30}. '
                "No other text."
            ),
        },
    ]
    try:
        resp: LLMResponse = await client.chat(
            messages, model, max_tokens=50,
        )
        text = (resp.content or "").strip()
        if not text:
            return 0.0
        # Try to parse as JSON
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "name" in parsed:
            return 1.0
        return 0.5
    except json.JSONDecodeError:
        return 0.2  # produced text but not valid JSON
    except Exception:
        logger.debug("structured_output probe failed for %s", model, exc_info=True)
    return 0.0


async def _probe_instruction_following(
    client: ChatClient,
    model: str,
) -> float:
    """Probe whether the model follows a specific format constraint.

    Asks for a single-word answer and checks compliance.
    Returns 1.0 if the response is a single word, 0.5 if short, 0.0 otherwise.
    """
    messages = [
        {
            "role": "user",
            "content": "Reply with exactly one word: the capital of France.",
        },
    ]
    try:
        resp: LLMResponse = await client.chat(
            messages, model, max_tokens=20,
        )
        text = (resp.content or "").strip().rstrip(".")
        words = text.split()
        if len(words) == 1:
            return 1.0
        if len(words) <= 3:
            return 0.7
        return 0.3
    except Exception:
        logger.debug(
            "instruction_following probe failed for %s", model, exc_info=True,
        )
    return 0.0


# ---------------------------------------------------------------------------
# Main probe function
# ---------------------------------------------------------------------------


async def probe_model_capabilities(
    client: ChatClient,
    model: str,
) -> Dict[str, float]:
    """Run micro-prompt probes to determine a model's actual capabilities.

    Returns a dict of capability → score (0.0–1.0).

    This function does NOT check or update the cache — that is handled
    by ``probe_or_cache()`` which wraps this function.
    """
    results: Dict[str, float] = {}

    # Run probes sequentially (they're tiny, and we want to avoid
    # overwhelming a single-model backend)
    results["tool_calling"] = await _probe_tool_calling(client, model)
    results["structured_output"] = await _probe_structured_output(client, model)
    results["instruction_following"] = await _probe_instruction_following(
        client, model,
    )

    # Infer supports_tool_calling from tool_calling probe score
    results["supports_tool_calling"] = results["tool_calling"]

    return results


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def _cache_path(cache_dir: Optional[str] = None) -> Path:
    """Return the path to the probe cache file."""
    d = cache_dir or _DEFAULT_CACHE_DIR
    return Path(d) / _CACHE_FILENAME


def load_probe_cache(cache_dir: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """Load the probe cache from disk.  Returns empty dict on any error."""
    path = _cache_path(cache_dir)
    if not path.is_file():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        logger.debug("Failed to load probe cache from %s", path, exc_info=True)
    return {}


def save_probe_cache(
    cache: Dict[str, Dict[str, float]],
    cache_dir: Optional[str] = None,
) -> None:
    """Write the probe cache to disk.  Silently ignores errors."""
    path = _cache_path(cache_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        logger.debug("Failed to save probe cache to %s", path, exc_info=True)


async def probe_or_cache(
    client: ChatClient,
    model: str,
    cache_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Probe a model's capabilities, using cached results if available.

    If the model has been probed before (results in cache file), returns
    the cached results.  Otherwise, runs ``probe_model_capabilities()``
    and saves the results to cache.

    Args:
        client: Chat client to use for probing.
        model: Model ID to probe.
        cache_dir: Override cache directory (for testing).

    Returns:
        Capability scores dict.
    """
    cache = load_probe_cache(cache_dir)
    if model in cache:
        logger.debug("Using cached probe results for %s", model)
        return cache[model]

    logger.info("Probing capabilities of unknown model %s", model)
    results = await probe_model_capabilities(client, model)

    cache[model] = results
    save_probe_cache(cache, cache_dir)
    logger.info(
        "Probed %s: %s",
        model,
        ", ".join(f"{k}={v:.1f}" for k, v in sorted(results.items())),
    )

    return results
