"""Headless browser tool via Playwright.

Requires the ``[browser]`` extra: ``pip install 'agentic-concierge[browser]'``
which pulls in ``playwright``.  This module can be freely imported regardless
of whether the extra is installed — the ``is_available()`` check guards usage.

Async lifecycle::

    bt = BrowserTool(workspace_path)
    await bt.aopen()      # launches Chromium, creates a page
    result = await bt.navigate("https://example.com")
    await bt.aclose()     # closes browser, releases Playwright resources

Safety:

- 30 s timeout on all Playwright calls; ``TimeoutError`` is caught and returned
  as ``{"error": ..., "success": False}``.
- ``url`` must start with ``http://`` or ``https://`` — otherwise an immediate
  ``{"error": "invalid URL"}`` is returned.
- Screenshots are saved inside ``workspace_path`` only (sandbox-consistent).
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Timeout applied to every Playwright operation (milliseconds).
_TIMEOUT_MS = 30_000


def _browsers_installed() -> bool:
    """Check whether Playwright browser binaries have been downloaded."""
    try:
        from playwright._impl._driver import compute_driver_executable
        driver = Path(compute_driver_executable())
        # Browsers live in the same parent tree as the driver.
        registry = driver.parent / ".." / ".." / "driver" / "package" / ".local-browsers"
        if registry.exists():
            return True
        # Fallback: check the standard cache directory.
        import os
        cache = Path(os.environ.get(
            "PLAYWRIGHT_BROWSERS_PATH",
            Path.home() / ".cache" / "ms-playwright",
        ))
        return cache.exists() and any(cache.iterdir())
    except Exception:
        return False


def is_available() -> bool:
    """Return ``True`` if ``playwright`` is importable AND browsers are downloaded.

    Uses ``importlib.util.find_spec`` so the module is not actually loaded
    when checking availability.  Also verifies that browser binaries exist —
    without them, any launch attempt will fail with a noisy stderr message.
    """
    if importlib.util.find_spec("playwright") is None:
        return False
    return _browsers_installed()


class BrowserTool:
    """Headless browser tool wrapping Playwright Chromium.

    Call ``aopen()`` before any tool method, and ``aclose()`` when done.
    All tool methods return a ``Dict[str, Any]``.  Errors are returned as
    ``{"error": ..., "success": False}`` rather than raised.

    Raises:
        FeatureDisabledError: On instantiation if ``playwright`` is not installed.
    """

    def __init__(self, workspace_path: str, headless: bool = True) -> None:
        if not is_available():
            from agentic_concierge.config.features import Feature, FeatureDisabledError
            raise FeatureDisabledError(
                Feature.BROWSER,
                "Install with: pip install 'agentic-concierge[browser]'",
            )
        self._workspace_path = workspace_path
        self._headless = headless
        self._playwright: Optional[Any] = None
        self._browser: Optional[Any] = None
        self._page: Optional[Any] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def aopen(self) -> None:
        """Launch Playwright Chromium and create a single page."""
        from playwright.async_api import async_playwright  # type: ignore[import]

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self._headless)
        self._page = await self._browser.new_page()
        logger.debug("BrowserTool: browser opened (headless=%s)", self._headless)

    async def aclose(self) -> None:
        """Close the browser and release Playwright resources."""
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
            self._page = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None
        logger.debug("BrowserTool: browser closed")

    # ------------------------------------------------------------------
    # Tool methods
    # ------------------------------------------------------------------

    async def navigate(self, url: str) -> Dict[str, Any]:
        """Navigate to *url* and return page metadata.

        Returns:
            ``{"url", "title", "status_code", "content_length"}`` on success.
            ``{"error": ..., "success": False}`` on failure.
        """
        if not url.startswith(("http://", "https://")):
            return {
                "error": "invalid URL: must start with http:// or https://",
                "success": False,
            }
        if self._page is None:
            return {"error": "BrowserTool not opened — call aopen() first", "success": False}
        try:
            response = await self._page.goto(url, timeout=_TIMEOUT_MS)
            status = response.status if response is not None else None
            title = await self._page.title()
            content = await self._page.content()
            return {
                "url": url,
                "title": title,
                "status_code": status,
                "content_length": len(content),
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "success": False}

    async def get_text(self, selector: str = "body") -> Dict[str, Any]:
        """Extract inner text from *selector*.

        Returns:
            ``{"selector", "text", "length"}`` on success.
            ``{"error": ..., "success": False}`` on failure.
        """
        if self._page is None:
            return {"error": "BrowserTool not opened — call aopen() first", "success": False}
        try:
            text = await self._page.inner_text(selector, timeout=_TIMEOUT_MS)
            return {"selector": selector, "text": text, "length": len(text)}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "success": False}

    async def get_links(self) -> Dict[str, Any]:
        """Return all anchor links on the current page.

        Returns:
            ``{"links": [{"text", "href"}], "count"}`` on success.
            ``{"error": ..., "success": False}`` on failure.
        """
        if self._page is None:
            return {"error": "BrowserTool not opened — call aopen() first", "success": False}
        try:
            anchors = await self._page.query_selector_all("a")
            links: List[Dict[str, str]] = []
            for a in anchors:
                text = (await a.inner_text()) or ""
                href = (await a.get_attribute("href")) or ""
                links.append({"text": text.strip(), "href": href})
            return {"links": links, "count": len(links)}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "success": False}

    async def click(self, selector: str) -> Dict[str, Any]:
        """Click the element matching *selector*.

        Returns:
            ``{"success": True, "selector": ...}`` on success.
            ``{"error": ..., "success": False, "selector": ...}`` on failure.
        """
        if self._page is None:
            return {"error": "BrowserTool not opened — call aopen() first", "success": False}
        try:
            await self._page.click(selector, timeout=_TIMEOUT_MS)
            # Best-effort: wait for any resulting navigation to settle (5 s).
            # Ignored if no navigation was triggered.
            try:
                await self._page.wait_for_load_state("domcontentloaded", timeout=5_000)
            except Exception:  # noqa: BLE001
                pass
            return {"success": True, "selector": selector}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "success": False, "selector": selector}

    async def fill(self, selector: str, value: str) -> Dict[str, Any]:
        """Fill *selector* with *value*.

        Returns:
            ``{"success": True, "selector": ..., "value": ...}`` on success.
            ``{"error": ..., "success": False}`` on failure.
        """
        if self._page is None:
            return {"error": "BrowserTool not opened — call aopen() first", "success": False}
        try:
            await self._page.fill(selector, value, timeout=_TIMEOUT_MS)
            return {"success": True, "selector": selector, "value": value}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "success": False}

    async def screenshot(self, filename: str = "screenshot.png") -> Dict[str, Any]:
        """Take a screenshot and save it to the workspace directory.

        Args:
            filename: Filename within the workspace directory (no path components).

        Returns:
            ``{"path": filename, "workspace_path": absolute_path}`` on success.
            ``{"error": ..., "success": False}`` on failure.
        """
        if self._page is None:
            return {"error": "BrowserTool not opened — call aopen() first", "success": False}
        workspace = Path(self._workspace_path).resolve()
        save_path = (workspace / filename).resolve()
        # Enforce sandbox: reject paths outside the workspace directory.
        if not str(save_path).startswith(str(workspace) + "/") and save_path != workspace:
            return {
                "error": (
                    f"invalid filename {filename!r}: path must remain inside the workspace directory"
                ),
                "success": False,
            }
        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            await self._page.screenshot(path=str(save_path), timeout=_TIMEOUT_MS)
            return {"path": filename, "workspace_path": str(save_path)}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "success": False}
