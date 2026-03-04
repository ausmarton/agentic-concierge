"""Base specialist pack: system prompt, OpenAI tool definitions, execute_tool."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agentic_concierge.config.features import FeatureSet


class BaseSpecialistPack:
    """Concrete ``SpecialistPack``: holds system prompt, tool definitions, and executors.

    ``tools`` maps tool name → ``(openai_tool_def, executor_fn)``.  The finish tool
    definition is stored separately so the loop can detect termination without the
    pack needing to know about the loop.  The finish tool is **included** in
    ``tool_definitions`` (so the LLM knows about it) but is **not** in the executor
    map (the loop handles it directly by extracting the arguments as the final
    payload).

    When ``feature_set`` is provided and ``Feature.BROWSER`` is enabled, browser
    tools are registered during ``aopen()`` if Playwright is installed.
    """

    FINISH_TOOL_NAME = "finish_task"

    def __init__(
        self,
        specialist_id: str,
        system_prompt: str,
        tools: Dict[str, Tuple[Dict[str, Any], Callable[..., Any]]],
        finish_tool_def: Dict[str, Any],
        workspace_path: str = "",
        network_allowed: bool = True,
        feature_set: Optional["FeatureSet"] = None,
    ):
        """
        Args:
            specialist_id: Identifier (e.g. ``"engineering"``).
            system_prompt: System message for this specialist.
            tools: Regular (non-finish) tools: ``name → (openai_def, executor)``.
                ``openai_def`` is a full OpenAI function tool definition dict.
            finish_tool_def: OpenAI tool definition for ``finish_task`` (the
                terminal tool).  Its arguments become the run payload.
            workspace_path: Workspace directory path; forwarded to ``BrowserTool``
                for screenshot storage.  Defaults to ``""`` (browser disabled).
            network_allowed: Whether network tools are permitted.  Browser tools
                are suppressed when ``False``.  Defaults to ``True``.
            feature_set: Optional ``FeatureSet`` controlling which capabilities
                are active.  When ``None`` no browser tools are registered.
        """
        self._specialist_id = specialist_id
        self._system_prompt = system_prompt
        self._tools: Dict[str, Tuple[Dict[str, Any], Callable[..., Any]]] = dict(tools)
        self._finish_tool_def = finish_tool_def
        self._workspace_path = workspace_path
        self._network_allowed = network_allowed
        self._feature_set = feature_set
        self._browser_tool: Optional[Any] = None  # Optional[BrowserTool]

    @property
    def specialist_id(self) -> str:
        return self._specialist_id

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def tool_definitions(self) -> List[Dict[str, Any]]:
        """All tool definitions (regular + finish) in OpenAI format."""
        return [defn for defn, _ in self._tools.values()] + [self._finish_tool_def]

    @property
    def tool_names(self) -> List[str]:
        """Names of regular tools (excludes finish tool)."""
        return list(self._tools.keys())

    @property
    def finish_tool_name(self) -> str:
        return self.FINISH_TOOL_NAME

    @property
    def finish_required_fields(self) -> List[str]:
        """Required argument field names derived from the finish tool's parameter schema."""
        return list(
            self._finish_tool_def.get("function", {})
            .get("parameters", {})
            .get("required", [])
        )

    def validate_finish_payload(self, payload: dict) -> Optional[str]:
        """Return an error message string if the payload fails quality checks, else None.

        Subclasses (e.g. EngineeringSpecialistPack) override this to enforce
        pack-specific quality gates such as requiring tests to pass before
        finish_task is accepted.  The base implementation always returns None
        (no additional quality checks beyond required-field validation).
        """
        return None

    def set_feature_set(self, feature_set: "FeatureSet") -> None:
        """Replace the feature set.  Called by the registry after pack construction."""
        self._feature_set = feature_set

    async def aopen(self) -> None:
        """Lifecycle hook: register browser tools if feature is enabled and available."""
        from agentic_concierge.config.features import Feature
        from agentic_concierge.infrastructure.tools.browser_tool import (
            BrowserTool,
            is_available as browser_is_available,
        )

        if (
            self._feature_set is not None
            and self._feature_set.is_enabled(Feature.BROWSER)
            and self._network_allowed
            and self._workspace_path
            and browser_is_available()
        ):
            self._browser_tool = BrowserTool(self._workspace_path)
            try:
                await self._browser_tool.aopen()
                self._register_browser_tools()
            except Exception as e:
                logger.warning("Browser tool unavailable (browsers not installed?): %s", e)
                self._browser_tool = None

    async def aclose(self) -> None:
        """Lifecycle hook: close browser if it was opened."""
        if self._browser_tool is not None:
            await self._browser_tool.aclose()
            self._browser_tool = None

    async def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a regular (non-finish) tool by name.

        Sync tool functions are called directly.  Async tool functions (coroutines)
        are awaited — this allows browser tool executors to be async methods.
        """
        if tool_name not in self._tools:
            return {"error": f"Unknown tool: {tool_name!r}. Available: {list(self._tools)}"}
        _, fn = self._tools[tool_name]
        result = fn(**args)
        if asyncio.iscoroutine(result):
            return await result
        return result

    # ------------------------------------------------------------------
    # Browser tool registration (called from aopen when browser is ready)
    # ------------------------------------------------------------------

    def _register_browser_tools(self) -> None:
        """Add browser tool definitions and executors to ``self._tools``.

        Called only when browser is available and feature is enabled.
        The ``BrowserTool`` instance (``self._browser_tool``) must be opened
        before this method is called.
        """
        from agentic_concierge.infrastructure.specialists.tool_defs import make_tool_def

        bt = self._browser_tool
        if bt is None:
            return

        self._tools["browser_navigate"] = (
            make_tool_def(
                "browser_navigate",
                "Navigate to a URL in the headless browser and return page metadata.",
                {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Full URL including scheme, e.g. https://example.com",
                        },
                    },
                    "required": ["url"],
                },
            ),
            bt.navigate,
        )

        self._tools["browser_get_text"] = (
            make_tool_def(
                "browser_get_text",
                "Extract inner text from a CSS selector on the current page.",
                {
                    "type": "object",
                    "properties": {
                        "selector": {
                            "type": "string",
                            "description": "CSS selector (default: 'body').",
                        },
                    },
                    "required": [],
                },
            ),
            bt.get_text,
        )

        self._tools["browser_get_links"] = (
            make_tool_def(
                "browser_get_links",
                "Return all anchor links (text + href) on the current page.",
                {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            bt.get_links,
        )

        self._tools["browser_click"] = (
            make_tool_def(
                "browser_click",
                "Click an element matching a CSS selector on the current page.",
                {
                    "type": "object",
                    "properties": {
                        "selector": {
                            "type": "string",
                            "description": "CSS selector of the element to click.",
                        },
                    },
                    "required": ["selector"],
                },
            ),
            bt.click,
        )

        self._tools["browser_fill"] = (
            make_tool_def(
                "browser_fill",
                "Fill an input field matching a CSS selector with a value.",
                {
                    "type": "object",
                    "properties": {
                        "selector": {
                            "type": "string",
                            "description": "CSS selector of the input element.",
                        },
                        "value": {
                            "type": "string",
                            "description": "Text value to fill into the field.",
                        },
                    },
                    "required": ["selector", "value"],
                },
            ),
            bt.fill,
        )

        self._tools["browser_screenshot"] = (
            make_tool_def(
                "browser_screenshot",
                "Take a screenshot of the current page and save it to the workspace.",
                {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Filename for the screenshot (default: screenshot.png).",
                        },
                    },
                    "required": [],
                },
            ),
            bt.screenshot,
        )
