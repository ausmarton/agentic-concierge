"""Consult specialist model tool: single-shot query to a non-tool-calling model.

This lets a tool-calling agent delegate specific sub-tasks (code generation,
SQL queries, reasoning) to a specialist model that may not support OpenAI
tool calling (e.g. sqlcoder, deepseek-coder, codellama).

The tool makes a single chat completion call (no tool loop) and returns
the response as plain text.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agentic_concierge.config.schema import ModelConfig
from agentic_concierge.infrastructure.chat import build_chat_client
from agentic_concierge.infrastructure.model_profiles import (
    SPECIALTY_CAPABILITIES,
    match_models,
)

logger = logging.getLogger(__name__)


async def execute_consult(
    specialty: str,
    prompt: str,
    all_chat_models: List[str],
    base_url: str,
    backend: str = "ollama",
    api_key: str = "",
    timeout_s: float = 120.0,
) -> Dict[str, Any]:
    """Consult a specialist model for a specific sub-task.

    Args:
        specialty: What kind of specialist ("code", "sql", "reasoning").
        prompt: The question or task for the specialist.
        all_chat_models: All available chat-capable models (including non-tool-calling).
        base_url: LLM backend URL.
        backend: Backend type.
        api_key: API key for the backend.
        timeout_s: Timeout for the chat call.

    Returns:
        Dict with ``response`` (the model's answer) and ``model`` (which model was used).
    """
    caps = SPECIALTY_CAPABILITIES.get(specialty)
    if not caps:
        return {
            "error": f"Unknown specialty {specialty!r}. Available: {sorted(SPECIALTY_CAPABILITIES.keys())}",
            "response": "",
            "model": "",
        }

    model = match_models(
        all_chat_models,
        required_capabilities=caps,
        require_tool_calling=False,  # key: allows non-tool-calling models
    )
    if not model:
        return {
            "error": f"No specialist model available for {specialty!r}",
            "response": "",
            "model": "",
        }

    model_cfg = ModelConfig(
        base_url=base_url,
        model=model,
        backend=backend,
        api_key=api_key,
        timeout_s=timeout_s,
    )
    client = build_chat_client(model_cfg)

    try:
        response = await client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.1,
            max_tokens=2048,
        )
        return {
            "response": response.content or "",
            "model": model,
            "specialty": specialty,
        }
    except Exception as exc:
        logger.warning("consult_specialist_model failed (%s): %s", model, exc)
        return {
            "error": f"Specialist model call failed: {exc}",
            "response": "",
            "model": model,
        }
