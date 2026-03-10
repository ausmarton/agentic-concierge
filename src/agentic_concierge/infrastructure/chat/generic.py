"""Generic OpenAI-compatible chat client.

Works with any backend exposing ``POST /v1/chat/completions`` in the standard
OpenAI format: cloud providers (OpenAI, Anthropic via LiteLLM bridge, etc.),
self-hosted vLLM, LM Studio, and others.

Unlike :class:`~agentic_concierge.infrastructure.ollama.client.OllamaChatClient`
this client does **not** implement Ollama-specific workarounds:

- No 400 retry with a minimal payload (Ollama quirk for old versions).
- No detection of the "does not support tools" error message (Ollama-specific
  phrasing).

Use it by setting ``backend: "generic"`` on the relevant ``ModelConfig`` in
your ``CONCIERGE_CONFIG_PATH`` file.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from agentic_concierge.config.constants import LLM_CHAT_DEFAULT_TIMEOUT_S
from agentic_concierge.domain import LLMResponse
from agentic_concierge.infrastructure.chat._parser import parse_chat_response

logger = logging.getLogger(__name__)


class GenericChatClient:
    """Bare OpenAI-compatible chat client (no backend-specific workarounds).

    Suitable for cloud APIs and any server that faithfully implements the
    OpenAI ``/chat/completions`` spec.  Raises ``httpx.HTTPStatusError`` for
    any non-2xx response without retrying.

    Uses a shared ``httpx.AsyncClient`` for connection pooling.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout_s: float = LLM_CHAT_DEFAULT_TIMEOUT_S,
    ) -> None:
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
        r.raise_for_status()
        return parse_chat_response(r.json())

    def _get_client(self) -> "httpx.AsyncClient":
        """Return the shared HTTP client, creating it lazily on first use."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._timeout))
        return self._client
