"""Tests for consult_specialist_model tool (ADR-025)."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agentic_concierge.infrastructure.tools.consult import execute_consult


@pytest.mark.asyncio
async def test_consult_unknown_specialty():
    result = await execute_consult(
        specialty="unknown",
        prompt="test",
        all_chat_models=["qwen2.5:7b"],
        base_url="http://localhost:11434/v1",
    )
    assert "error" in result
    assert "Unknown specialty" in result["error"]


@pytest.mark.asyncio
async def test_consult_no_models_available():
    result = await execute_consult(
        specialty="sql",
        prompt="SELECT * FROM users",
        all_chat_models=[],
        base_url="http://localhost:11434/v1",
    )
    assert "error" in result
    assert "No specialist model" in result["error"]


@pytest.mark.asyncio
async def test_consult_selects_specialist_model():
    """consult_specialist_model with specialty='sql' should prefer sqlcoder."""
    from agentic_concierge.domain import LLMResponse

    mock_response = LLMResponse(content="SELECT * FROM customers ORDER BY revenue DESC LIMIT 10", tool_calls=[])

    mock_client = AsyncMock()
    mock_client.chat = AsyncMock(return_value=mock_response)

    with patch("agentic_concierge.infrastructure.tools.consult.build_chat_client", return_value=mock_client):
        result = await execute_consult(
            specialty="sql",
            prompt="Write a query to find top customers",
            all_chat_models=["qwen2.5:7b", "sqlcoder:15b"],
            base_url="http://localhost:11434/v1",
        )

    assert result["model"] == "sqlcoder:15b"
    assert "SELECT" in result["response"]
    assert result["specialty"] == "sql"


@pytest.mark.asyncio
async def test_consult_code_prefers_coder():
    """consult_specialist_model with specialty='code' should prefer a code-specialized model."""
    from agentic_concierge.domain import LLMResponse

    mock_response = LLMResponse(content="def hello(): print('world')", tool_calls=[])
    mock_client = AsyncMock()
    mock_client.chat = AsyncMock(return_value=mock_response)

    with patch("agentic_concierge.infrastructure.tools.consult.build_chat_client", return_value=mock_client):
        result = await execute_consult(
            specialty="code",
            prompt="Write a hello world function",
            all_chat_models=["qwen2.5:7b", "deepseek-coder:6.7b"],
            base_url="http://localhost:11434/v1",
        )

    assert result["model"] == "deepseek-coder:6.7b"


@pytest.mark.asyncio
async def test_consult_handles_llm_error():
    """consult_specialist_model should handle LLM errors gracefully."""
    mock_client = AsyncMock()
    mock_client.chat = AsyncMock(side_effect=Exception("connection refused"))

    with patch("agentic_concierge.infrastructure.tools.consult.build_chat_client", return_value=mock_client):
        result = await execute_consult(
            specialty="code",
            prompt="test",
            all_chat_models=["qwen2.5:7b"],
            base_url="http://localhost:11434/v1",
        )

    assert "error" in result
    assert "connection refused" in result["error"]
    assert result["model"] == "qwen2.5:7b"


@pytest.mark.asyncio
async def test_consult_reasoning_prefers_reasoning_model():
    """specialty='reasoning' should prefer models with high reasoning scores."""
    from agentic_concierge.domain import LLMResponse

    mock_response = LLMResponse(content="The answer is 42.", tool_calls=[])
    mock_client = AsyncMock()
    mock_client.chat = AsyncMock(return_value=mock_response)

    with patch("agentic_concierge.infrastructure.tools.consult.build_chat_client", return_value=mock_client):
        result = await execute_consult(
            specialty="reasoning",
            prompt="What is the meaning of life?",
            all_chat_models=["qwen2.5:7b", "deepseek-r1-distill-qwen:14b"],
            base_url="http://localhost:11434/v1",
        )

    # deepseek-r1-distill has reasoning=0.9, qwen2.5 has 0.7
    assert result["model"] == "deepseek-r1-distill-qwen:14b"


def test_consult_tool_in_catalog():
    """consult_specialist_model should be registered in the tool catalog."""
    from agentic_concierge.infrastructure.specialists.tool_catalog import get_tool
    entry = get_tool("consult_specialist_model")
    assert entry.name == "consult_specialist_model"
    assert entry.category == "ai"


def test_consult_tool_definition_schema():
    from agentic_concierge.infrastructure.specialists.tool_catalog import get_tool
    entry = get_tool("consult_specialist_model")
    func = entry.openai_def["function"]
    assert func["name"] == "consult_specialist_model"
    params = func["parameters"]
    assert "prompt" in params["properties"]
    assert "specialty" in params["properties"]
    assert params["properties"]["specialty"]["enum"] == ["code", "sql", "reasoning"]
