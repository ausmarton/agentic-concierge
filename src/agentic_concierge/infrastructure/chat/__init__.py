"""LLM client factory: build the right ChatClient for a ModelConfig."""

from __future__ import annotations

from agentic_concierge.config.schema import ModelConfig
from agentic_concierge.application.ports import ChatClient


def build_chat_client(model_config: ModelConfig) -> ChatClient:
    """Return the correct ``ChatClient`` implementation for *model_config*.

    Dispatch is based on ``model_config.backend``:

    ``"ollama"`` (default)
        :class:`~agentic_concierge.infrastructure.ollama.client.OllamaChatClient` —
        includes the Ollama 400 retry and "does not support tools" error
        detection.

    ``"generic"``
        :class:`~agentic_concierge.infrastructure.chat.generic.GenericChatClient` —
        bare OpenAI-compatible client; suitable for cloud providers (OpenAI,
        Anthropic via LiteLLM bridge, vLLM, LM Studio, etc.).

    ``"vllm"``
        :class:`~agentic_concierge.infrastructure.chat.vllm.VLLMChatClient` —
        OpenAI-compatible client optimised for vLLM; supports CUDA and ROCm,
        no ``vllm`` Python package required.

    Raises:
        ValueError: For unknown backend values.
    """
    backend = model_config.backend
    if backend == "ollama":
        from agentic_concierge.infrastructure.ollama.client import OllamaChatClient
        return OllamaChatClient(
            base_url=model_config.base_url,
            api_key=model_config.api_key,
            timeout_s=model_config.timeout_s,
        )
    if backend == "generic":
        from agentic_concierge.infrastructure.chat.generic import GenericChatClient
        return GenericChatClient(
            base_url=model_config.base_url,
            api_key=model_config.api_key,
            timeout_s=model_config.timeout_s,
        )
    if backend == "vllm":
        from agentic_concierge.infrastructure.chat.vllm import VLLMChatClient
        return VLLMChatClient(
            base_url=model_config.base_url,
            api_key=model_config.api_key,
            timeout_s=model_config.timeout_s,
        )
    raise ValueError(
        f"Unknown LLM backend {backend!r}. "
        "Supported backends: 'ollama' (default), 'generic' (OpenAI-compatible cloud/vLLM), "
        "'vllm' (vLLM server)."
    )
