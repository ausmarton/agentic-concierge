"""Tests for BrowserTool — all Playwright calls mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_concierge.infrastructure.tools.browser_tool import BrowserTool, is_available
from agentic_concierge.config.features import Feature, FeatureDisabledError


# ---------------------------------------------------------------------------
# is_available()
# ---------------------------------------------------------------------------

def test_is_available_when_playwright_installed():
    with patch("agentic_concierge.infrastructure.tools.browser_tool.importlib.util.find_spec",
               return_value=MagicMock()):
        assert is_available() is True


def test_is_available_when_playwright_absent():
    with patch("agentic_concierge.infrastructure.tools.browser_tool.importlib.util.find_spec",
               return_value=None):
        assert is_available() is False


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

def test_init_raises_when_playwright_absent(tmp_path):
    with patch(
        "agentic_concierge.infrastructure.tools.browser_tool.is_available",
        return_value=False,
    ):
        with pytest.raises(FeatureDisabledError) as exc_info:
            BrowserTool(str(tmp_path))
    assert exc_info.value.feature == Feature.BROWSER
    assert "browser" in str(exc_info.value).lower()


def test_init_succeeds_when_playwright_present(tmp_path):
    with patch(
        "agentic_concierge.infrastructure.tools.browser_tool.is_available",
        return_value=True,
    ):
        bt = BrowserTool(str(tmp_path))
    assert bt._workspace_path == str(tmp_path)
    assert bt._headless is True
    assert bt._browser is None
    assert bt._page is None


# ---------------------------------------------------------------------------
# aopen / aclose
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_aopen_launches_browser_and_page(tmp_path):
    import sys

    mock_page = AsyncMock()
    mock_browser = AsyncMock()
    mock_browser.new_page = AsyncMock(return_value=mock_page)
    mock_playwright = MagicMock()
    mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)

    mock_pw_instance = MagicMock()
    mock_pw_instance.start = AsyncMock(return_value=mock_playwright)

    mock_async_playwright = MagicMock(return_value=mock_pw_instance)
    mock_playwright_module = MagicMock()
    mock_playwright_module.async_playwright = mock_async_playwright

    with patch(
        "agentic_concierge.infrastructure.tools.browser_tool.is_available",
        return_value=True,
    ):
        bt = BrowserTool(str(tmp_path))

    with patch.dict(sys.modules, {
        "playwright": MagicMock(),
        "playwright.async_api": mock_playwright_module,
    }):
        await bt.aopen()

    assert bt._browser is mock_browser
    assert bt._page is mock_page
    mock_playwright.chromium.launch.assert_awaited_once_with(headless=True)
    mock_browser.new_page.assert_awaited_once()


@pytest.mark.asyncio
async def test_aclose_releases_resources(tmp_path):
    mock_browser = AsyncMock()
    mock_playwright = AsyncMock()

    with patch(
        "agentic_concierge.infrastructure.tools.browser_tool.is_available",
        return_value=True,
    ):
        bt = BrowserTool(str(tmp_path))
    bt._browser = mock_browser
    bt._playwright = mock_playwright
    bt._page = AsyncMock()

    await bt.aclose()

    mock_browser.close.assert_awaited_once()
    mock_playwright.stop.assert_awaited_once()
    assert bt._browser is None
    assert bt._page is None
    assert bt._playwright is None


# ---------------------------------------------------------------------------
# navigate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_navigate_happy_path(tmp_path):
    mock_response = MagicMock()
    mock_response.status = 200
    mock_page = AsyncMock()
    mock_page.goto = AsyncMock(return_value=mock_response)
    mock_page.title = AsyncMock(return_value="Example Domain")
    mock_page.content = AsyncMock(return_value="<html><body>Hello</body></html>")

    with patch(
        "agentic_concierge.infrastructure.tools.browser_tool.is_available",
        return_value=True,
    ):
        bt = BrowserTool(str(tmp_path))
    bt._page = mock_page

    result = await bt.navigate("https://example.com")

    assert result["url"] == "https://example.com"
    assert result["title"] == "Example Domain"
    assert result["status_code"] == 200
    assert result["content_length"] > 0


@pytest.mark.asyncio
async def test_navigate_invalid_url_returns_error(tmp_path):
    with patch(
        "agentic_concierge.infrastructure.tools.browser_tool.is_available",
        return_value=True,
    ):
        bt = BrowserTool(str(tmp_path))
    bt._page = AsyncMock()

    result = await bt.navigate("ftp://not-http.com")

    assert result["success"] is False
    assert "invalid URL" in result["error"]


# ---------------------------------------------------------------------------
# get_text
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_text_returns_text(tmp_path):
    mock_page = AsyncMock()
    mock_page.inner_text = AsyncMock(return_value="Hello world from page body")

    with patch(
        "agentic_concierge.infrastructure.tools.browser_tool.is_available",
        return_value=True,
    ):
        bt = BrowserTool(str(tmp_path))
    bt._page = mock_page

    result = await bt.get_text(selector="body")

    assert result["selector"] == "body"
    assert result["text"] == "Hello world from page body"
    assert result["length"] == len("Hello world from page body")


# ---------------------------------------------------------------------------
# get_links
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_links_returns_list(tmp_path):
    anchor1 = AsyncMock()
    anchor1.inner_text = AsyncMock(return_value="Google")
    anchor1.get_attribute = AsyncMock(return_value="https://google.com")
    anchor2 = AsyncMock()
    anchor2.inner_text = AsyncMock(return_value="Bing")
    anchor2.get_attribute = AsyncMock(return_value="https://bing.com")

    mock_page = AsyncMock()
    mock_page.query_selector_all = AsyncMock(return_value=[anchor1, anchor2])

    with patch(
        "agentic_concierge.infrastructure.tools.browser_tool.is_available",
        return_value=True,
    ):
        bt = BrowserTool(str(tmp_path))
    bt._page = mock_page

    result = await bt.get_links()

    assert result["count"] == 2
    assert result["links"][0] == {"text": "Google", "href": "https://google.com"}
    assert result["links"][1] == {"text": "Bing", "href": "https://bing.com"}


# ---------------------------------------------------------------------------
# click
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_click_success(tmp_path):
    mock_page = AsyncMock()

    with patch(
        "agentic_concierge.infrastructure.tools.browser_tool.is_available",
        return_value=True,
    ):
        bt = BrowserTool(str(tmp_path))
    bt._page = mock_page

    result = await bt.click("#submit-btn")

    assert result["success"] is True
    assert result["selector"] == "#submit-btn"
    mock_page.click.assert_awaited_once()


@pytest.mark.asyncio
async def test_click_failure_returns_error(tmp_path):
    mock_page = AsyncMock()
    mock_page.click = AsyncMock(side_effect=Exception("Element not found"))

    with patch(
        "agentic_concierge.infrastructure.tools.browser_tool.is_available",
        return_value=True,
    ):
        bt = BrowserTool(str(tmp_path))
    bt._page = mock_page

    result = await bt.click("#missing")

    assert result["success"] is False
    assert "Element not found" in result["error"]


# ---------------------------------------------------------------------------
# fill
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fill_success(tmp_path):
    mock_page = AsyncMock()

    with patch(
        "agentic_concierge.infrastructure.tools.browser_tool.is_available",
        return_value=True,
    ):
        bt = BrowserTool(str(tmp_path))
    bt._page = mock_page

    result = await bt.fill("#email", "user@example.com")

    assert result["success"] is True
    assert result["selector"] == "#email"
    assert result["value"] == "user@example.com"
    mock_page.fill.assert_awaited_once()


# ---------------------------------------------------------------------------
# screenshot
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_screenshot_saves_to_workspace(tmp_path):
    mock_page = AsyncMock()

    with patch(
        "agentic_concierge.infrastructure.tools.browser_tool.is_available",
        return_value=True,
    ):
        bt = BrowserTool(str(tmp_path))
    bt._page = mock_page

    result = await bt.screenshot("page.png")

    assert result["path"] == "page.png"
    assert str(tmp_path) in result["workspace_path"]
    mock_page.screenshot.assert_awaited_once()
    # Check path passed to screenshot includes workspace
    call_kwargs = mock_page.screenshot.call_args
    assert "path" in call_kwargs.kwargs
    assert str(tmp_path) in call_kwargs.kwargs["path"]


@pytest.mark.asyncio
async def test_screenshot_path_traversal_rejected(tmp_path):
    """screenshot() must reject filenames that escape the workspace directory."""
    with patch(
        "agentic_concierge.infrastructure.tools.browser_tool.is_available",
        return_value=True,
    ):
        bt = BrowserTool(str(tmp_path))
    bt._page = AsyncMock()

    result = await bt.screenshot("../../etc/passwd")

    assert result["success"] is False
    assert "invalid filename" in result["error"]
    bt._page.screenshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_aopen_failure_does_not_crash_pack(tmp_path):
    """When BrowserTool.aopen() fails (e.g. browsers not installed), the pack degrades gracefully."""
    from agentic_concierge.infrastructure.specialists.base import BaseSpecialistPack
    from agentic_concierge.config.features import FeatureSet, Feature

    feature_set = FeatureSet(frozenset({Feature.BROWSER}))
    pack = BaseSpecialistPack.__new__(BaseSpecialistPack)
    pack._tools = {}
    pack._finish_tool_def = None
    pack._system_prompt = "test"
    pack._specialist_id = "test"
    pack._workspace_path = str(tmp_path)
    pack._network_allowed = True
    pack._feature_set = feature_set
    pack._browser_tool = None
    pack._mcp_sessions = []

    with patch(
        "agentic_concierge.infrastructure.tools.browser_tool.is_available",
        return_value=True,
    ):
        # Simulate browser binaries not installed
        with patch.object(BrowserTool, "aopen", side_effect=Exception("Executable doesn't exist")):
            await pack.aopen()

    # Pack should continue without browser — _browser_tool set to None
    assert pack._browser_tool is None


@pytest.mark.asyncio
async def test_navigate_without_aopen_returns_error(tmp_path):
    """All tool methods return an error dict when the browser is not opened."""
    with patch(
        "agentic_concierge.infrastructure.tools.browser_tool.is_available",
        return_value=True,
    ):
        bt = BrowserTool(str(tmp_path))
    # _page is None — aopen() was never called.
    result = await bt.navigate("https://example.com")
    assert result["success"] is False
    assert "aopen" in result["error"]
