"""Ports (abstract interfaces) used by the application layer.

Each port is a ``Protocol`` so the application depends only on the *shape* of the
collaborator, not on a concrete implementation.  Infrastructure adapters must satisfy
these shapes; the application never imports from infrastructure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from agentic_concierge.domain import LLMResponse, RunId, RunResult


class ChatClient(Protocol):
    """LLM chat interface (OpenAI chat-completions API).

    ``tools`` is a list of OpenAI-format function tool definitions.  When provided
    the LLM may respond with ``tool_calls`` (native function calling) rather than
    plain text.  ``OllamaChatClient`` is the default implementation; any backend
    exposing ``POST /v1/chat/completions`` with the same response shape works.
    """

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ) -> LLMResponse: ...


class RunRepository(Protocol):
    """Create runs and append run-log events."""

    def create_run(self) -> tuple[RunId, str, str]:
        """Create a new run directory; return (RunId, run_dir path, workspace path)."""
        ...

    def append_event(
        self,
        run_id: RunId,
        kind: str,
        payload: Dict[str, Any],
        step: Optional[str] = None,
    ) -> None: ...


class SpecialistPack(Protocol):
    """A specialist pack: system prompt, OpenAI tool definitions, and tool execution.

    The pack is responsible for:
    - Providing the system prompt for this specialist's role.
    - Providing the list of OpenAI-format tool definitions (passed to the LLM).
    - Executing individual tool calls (by name) and returning structured results.
    - Identifying the *finish tool* — the tool whose call signals task completion.
    """

    @property
    def specialist_id(self) -> str: ...

    @property
    def system_prompt(self) -> str: ...

    @property
    def tool_definitions(self) -> List[Dict[str, Any]]:
        """OpenAI-format tool definitions, including the finish tool."""
        ...

    @property
    def finish_tool_name(self) -> str:
        """Name of the tool that terminates the loop (e.g. ``'finish_task'``)."""
        ...

    @property
    def finish_required_fields(self) -> List[str]:
        """Required argument field names for the finish tool.

        Derived from the finish tool's OpenAI parameter schema.  Used by
        ``execute_task`` to validate the LLM's ``finish_task`` call before
        accepting it as the final payload.  If the call is missing any of
        these fields, the error is returned to the LLM as a tool result so
        it can retry with complete arguments.
        """
        ...

    async def aopen(self) -> None: ...

    async def aclose(self) -> None: ...

    async def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]: ...


class SpecialistRegistry(Protocol):
    """Resolve a specialist pack by id."""

    def get_pack(
        self,
        specialist_id: str,
        workspace_path: str,
        network_allowed: bool,
        *,
        tools: Optional[List[str]] = None,
        role: Optional[str] = None,
    ) -> SpecialistPack: ...

    def list_ids(self) -> List[str]: ...
