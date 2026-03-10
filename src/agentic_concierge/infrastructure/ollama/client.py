"""OpenAI-compatible HTTP chat client.

Default config points at Ollama (``http://localhost:11434/v1``) but any backend
exposing the same ``POST /v1/chat/completions`` endpoint works (vLLM, LiteLLM,
OpenAI, etc.).

Supports native function calling: when ``tools`` is provided the response is
parsed for ``tool_calls`` and returned as ``LLMResponse`` with structured
``ToolCallRequest`` objects, rather than raw text.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from agentic_concierge.config.constants import LLM_CHAT_DEFAULT_TIMEOUT_S
from agentic_concierge.domain import LLMResponse
from agentic_concierge.infrastructure.chat._parser import parse_chat_response

logger = logging.getLogger(__name__)


class OllamaChatClient:
    """OpenAI-compatible HTTP client with native tool-calling support.

    Sends ``POST {base_url}/chat/completions`` using the standard OpenAI
    request shape.  Falls back to a minimal payload (model + messages + stream)
    when the server returns 400, which some older Ollama versions do for unknown
    top-level parameters.

    Uses a shared ``httpx.AsyncClient`` for connection pooling — concurrent
    calls reuse TCP connections instead of opening a new socket per request.
    """

    def __init__(self, base_url: str, api_key: str = "", timeout_s: float = LLM_CHAT_DEFAULT_TIMEOUT_S):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_s
        self._client: "httpx.AsyncClient | None" = None

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        url = f"{self._base_url}/chat/completions"
        headers: Dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools

        logger.debug(
            "POST %s model=%s messages=%d tools=%d",
            url, model, len(messages), len(tools or []),
        )
        client = self._get_client()
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code == 400:
            # Inspect the error body before retrying.
            err_msg = _extract_error_message(r)
            if "does not support tools" in err_msg.lower():
                raise RuntimeError(
                    f"Model {model!r} does not support tool calling. "
                    "Use a tool-capable model such as llama3.1:8b, "
                    "mistral-small3.2:24b, or qwen2.5-coder:32b."
                )
            # Some backends 400 on unknown top-level params (temperature,
            # top_p, …); retry with a minimal payload that still includes
            # tools (required for tool calling).
            logger.warning(
                "400 from %s (model=%s): %s — retrying with minimal payload",
                url, model, err_msg[:200],
            )
            payload_minimal: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "stream": False,
            }
            if tools:
                payload_minimal["tools"] = tools
            r2 = await client.post(url, headers=headers, json=payload_minimal)
            if r2.status_code == 400:
                err_msg2 = _extract_error_message(r2)
                if "does not support tools" in err_msg2.lower():
                    raise RuntimeError(
                        f"Model {model!r} does not support tool calling. "
                        "Use a tool-capable model such as llama3.1:8b, "
                        "mistral-small3.2:24b, or qwen2.5-coder:32b."
                    )
            r2.raise_for_status()
            data = r2.json()
        else:
            r.raise_for_status()
            data = r.json()

        return parse_chat_response(data)

    def _get_client(self) -> "httpx.AsyncClient":
        """Return the shared HTTP client, creating it lazily on first use."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._timeout))
        return self._client


def _extract_error_message(response: "httpx.Response") -> str:
    """Extract a human-readable error string from a (likely 4xx) HTTP response."""
    try:
        body = response.json()
        if isinstance(body, dict):
            err = body.get("error") or {}
            if isinstance(err, dict):
                return err.get("message") or ""
            if isinstance(err, str):
                return err
    except Exception:
        pass
    return response.text or ""


