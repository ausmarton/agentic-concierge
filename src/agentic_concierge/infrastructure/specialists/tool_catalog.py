"""Central tool catalog: registry of all available tools with metadata.

Each tool is registered as a ``ToolEntry`` with its OpenAI function definition,
an executor factory, and metadata (category, network requirement, quality gate).

The executor factory is a callable that takes a ``SandboxPolicy`` and optional
keyword arguments (e.g. ``workspace_root`` for cross_run_search) and returns the
actual tool executor function.  This defers binding to runtime so the same
catalog entry can be used across different workspaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agentic_concierge.config.constants import SHELL_DEFAULT_TIMEOUT_S
from agentic_concierge.infrastructure.tools.sandbox import SandboxPolicy

from .tool_defs import (
    LIST_FILES_TOOL_DEF,
    READ_FILE_TOOL_DEF,
    WRITE_FILE_TOOL_DEF,
    make_tool_def,
)


@dataclass
class ToolEntry:
    """Metadata for a single tool in the catalog."""

    name: str
    description: str
    category: str  # "file_io" | "code" | "web" | "search"
    openai_def: Dict[str, Any]
    executor_factory: Callable[..., Callable[..., Any]]
    requires_network: bool = False
    quality_gate: Optional[str] = None


# ---------------------------------------------------------------------------
# Executor factories
# ---------------------------------------------------------------------------

def _shell_executor(policy: SandboxPolicy, **_kw: Any) -> Callable[..., Any]:
    from agentic_concierge.infrastructure.tools.shell_tools import run_shell
    return lambda cmd, timeout_s=SHELL_DEFAULT_TIMEOUT_S: run_shell(policy, cmd, timeout_s=timeout_s)


def _read_file_executor(policy: SandboxPolicy, **_kw: Any) -> Callable[..., Any]:
    from agentic_concierge.infrastructure.tools.file_tools import read_text
    return lambda path: read_text(policy, path)


def _write_file_executor(policy: SandboxPolicy, **_kw: Any) -> Callable[..., Any]:
    from agentic_concierge.infrastructure.tools.file_tools import write_text
    return lambda path, content: write_text(policy, path, content)


def _list_files_executor(policy: SandboxPolicy, **_kw: Any) -> Callable[..., Any]:
    from agentic_concierge.infrastructure.tools.file_tools import list_tree
    return lambda max_files=500: list_tree(policy, max_files=max_files)


def _run_tests_executor(policy: SandboxPolicy, **_kw: Any) -> Callable[..., Any]:
    from agentic_concierge.infrastructure.tools.test_runner import run_tests
    return lambda framework="auto", path=".": run_tests(policy, framework, path)


def _web_search_executor(_policy: SandboxPolicy, **_kw: Any) -> Callable[..., Any]:
    from agentic_concierge.infrastructure.tools.web_tools import web_search
    return lambda query, max_results=8: web_search(query, max_results=max_results)


def _fetch_url_executor(_policy: SandboxPolicy, **_kw: Any) -> Callable[..., Any]:
    from agentic_concierge.infrastructure.tools.web_tools import fetch_url
    return lambda url: fetch_url(url)


def _cross_run_search_executor(_policy: SandboxPolicy, **kw: Any) -> Callable[..., Any]:
    workspace_root = kw.get("workspace_root", "")

    def _search(query: str, limit: int = 5) -> dict:
        try:
            from agentic_concierge.infrastructure.workspace.run_index import search_index
            results = search_index(workspace_root, query, limit=limit)
            return {
                "results": [
                    {
                        "run_id": entry.run_id,
                        "prompt": entry.prompt_prefix,
                        "summary": entry.summary,
                        "specialists": entry.specialist_ids,
                        "model": entry.model_name,
                    }
                    for entry in results
                ],
                "query": query,
                "count": len(results),
                "note": (
                    "These are summaries from prior runs. "
                    "Verify key claims with current tool calls."
                ),
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": f"cross_run_search failed: {exc}", "query": query, "results": []}

    return _search


# ---------------------------------------------------------------------------
# OpenAI tool definitions
# ---------------------------------------------------------------------------

_SHELL_TOOL_DEF = make_tool_def(
    "shell",
    "Run a shell command inside the sandbox workspace. Use for compiling, "
    "testing, running scripts, git operations, etc.",
    {
        "type": "object",
        "properties": {
            "cmd": {
                "type": "array",
                "items": {"type": "string"},
                "description": 'Command and arguments as a list, e.g. ["pytest", "-v"].',
            },
            "timeout_s": {
                "type": "integer",
                "description": "Timeout in seconds (default 120).",
            },
        },
        "required": ["cmd"],
    },
)

_RUN_TESTS_TOOL_DEF = make_tool_def(
    "run_tests",
    "Run the project's test suite and return pass/fail status. "
    "Call this before finish_task to verify correctness.",
    {
        "type": "object",
        "properties": {
            "framework": {
                "type": "string",
                "description": (
                    "Test framework: 'auto' (detect), 'pytest', 'cargo', 'npm'. "
                    "Default: 'auto'."
                ),
            },
            "path": {
                "type": "string",
                "description": "Relative path to run tests from (default '.').",
            },
        },
        "required": [],
    },
)

_WEB_SEARCH_TOOL_DEF = make_tool_def(
    "web_search",
    "Search the web and return a list of results (title, URL, snippet).",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query string."},
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default 8).",
            },
        },
        "required": ["query"],
    },
)

_FETCH_URL_TOOL_DEF = make_tool_def(
    "fetch_url",
    "Fetch the full text content of a URL. Only URLs fetched here may be cited.",
    {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to fetch."},
        },
        "required": ["url"],
    },
)

_CROSS_RUN_SEARCH_TOOL_DEF = make_tool_def(
    "cross_run_search",
    (
        "Search the cross-run memory index for relevant prior research results. "
        "Always call this first to avoid repeating previous work. "
        "Returns summaries of past runs whose prompts or summaries match the query."
    ),
    {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (keywords or a short phrase).",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default 5).",
            },
        },
        "required": ["query"],
    },
)


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------

TOOL_CATALOG: Dict[str, ToolEntry] = {
    "shell": ToolEntry(
        name="shell",
        description="Run shell commands in the sandbox workspace",
        category="code",
        openai_def=_SHELL_TOOL_DEF,
        executor_factory=_shell_executor,
    ),
    "read_file": ToolEntry(
        name="read_file",
        description="Read a file from the workspace",
        category="file_io",
        openai_def=READ_FILE_TOOL_DEF,
        executor_factory=_read_file_executor,
    ),
    "write_file": ToolEntry(
        name="write_file",
        description="Write or overwrite a file in the workspace",
        category="file_io",
        openai_def=WRITE_FILE_TOOL_DEF,
        executor_factory=_write_file_executor,
    ),
    "list_files": ToolEntry(
        name="list_files",
        description="List all files in the workspace",
        category="file_io",
        openai_def=LIST_FILES_TOOL_DEF,
        executor_factory=_list_files_executor,
    ),
    "run_tests": ToolEntry(
        name="run_tests",
        description="Run the project test suite",
        category="code",
        openai_def=_RUN_TESTS_TOOL_DEF,
        executor_factory=_run_tests_executor,
        quality_gate="tests_verified",
    ),
    "web_search": ToolEntry(
        name="web_search",
        description="Search the web for information",
        category="web",
        openai_def=_WEB_SEARCH_TOOL_DEF,
        executor_factory=_web_search_executor,
        requires_network=True,
    ),
    "fetch_url": ToolEntry(
        name="fetch_url",
        description="Fetch full text content from a URL",
        category="web",
        openai_def=_FETCH_URL_TOOL_DEF,
        executor_factory=_fetch_url_executor,
        requires_network=True,
    ),
    "cross_run_search": ToolEntry(
        name="cross_run_search",
        description="Search prior run summaries in the cross-run index",
        category="search",
        openai_def=_CROSS_RUN_SEARCH_TOOL_DEF,
        executor_factory=_cross_run_search_executor,
    ),
}


def get_tool(name: str) -> ToolEntry:
    """Look up a tool by name. Raises ``KeyError`` if not found."""
    if name not in TOOL_CATALOG:
        raise KeyError(
            f"Unknown tool {name!r}. Available: {sorted(TOOL_CATALOG.keys())}"
        )
    return TOOL_CATALOG[name]


def list_tools() -> List[ToolEntry]:
    """Return all registered tools."""
    return list(TOOL_CATALOG.values())


def list_tools_by_category(category: str) -> List[ToolEntry]:
    """Return tools matching a given category."""
    return [t for t in TOOL_CATALOG.values() if t.category == category]


def tool_catalog_summary() -> str:
    """Human-readable summary of available tools, suitable for LLM prompts."""
    lines = []
    for entry in TOOL_CATALOG.values():
        net = " (requires network)" if entry.requires_network else ""
        lines.append(f"- {entry.name}: {entry.description}{net}")
    return "\n".join(lines)
