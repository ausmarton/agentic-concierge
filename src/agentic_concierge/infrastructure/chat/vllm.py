"""vLLM ChatClient: OpenAI-compatible HTTP client for vLLM endpoints.

No ``vllm`` Python package required — uses pure httpx for all calls.
vLLM supports both CUDA (NVIDIA) and ROCm (AMD) and handles concurrent
requests efficiently via its internal continuous-batching engine.

Use by setting ``backend: "vllm"`` on a ``ModelConfig`` in your config.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from agentic_concierge.domain import LLMResponse
from agentic_concierge.infrastructure.chat._parser import parse_chat_response

logger = logging.getLogger(__name__)


class VLLMChatClient:
    """ChatClient for vLLM OpenAI-compatible endpoints.

    Implements the same ``chat()`` interface as ``OllamaChatClient`` and
    ``GenericChatClient``.  Does not require the ``vllm`` Python package —
    communicates purely over HTTP.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "",
        timeout_s: float = 360.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_s
        self._client: "httpx.AsyncClient | None" = None

    def _get_client(self) -> "httpx.AsyncClient":
        """Return the shared HTTP client, creating it lazily on first use."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._timeout))
        return self._client

    async def health_check(self) -> bool:
        """Return ``True`` if the vLLM server is healthy (``GET /health``)."""
        health_url = self._base_url.replace("/v1", "").rstrip("/") + "/health"
        try:
            resp = await self._get_client().get(health_url, timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> List[str]:
        """Return list of model IDs available on the vLLM server."""
        headers: Dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            resp = await self._get_client().get(
                f"{self._base_url}/models", headers=headers, timeout=5.0
            )
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            logger.warning("vLLM list_models failed: %s", e)
            return []

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
        """Send a chat completion request to the vLLM server."""
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
            "POST %s/chat/completions model=%s messages=%d tools=%d",
            self._base_url, model, len(messages), len(tools or []),
        )
        client = self._get_client()
        resp = await client.post(
            f"{self._base_url}/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        return parse_chat_response(resp.json())
