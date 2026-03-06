"""Web tools: search and fetch (gated by network_allowed)."""

from __future__ import annotations

from datetime import timezone
from typing import Any, Dict

import httpx
import trafilatura

try:
    from ddgs import DDGS
except Exception:  # pragma: no cover
    try:
        from duckduckgo_search import DDGS
    except Exception:
        DDGS = None  # type: ignore


def web_search(query: str, max_results: int = 8) -> Dict[str, Any]:
    if DDGS is None:
        return {"query": query, "results": [], "warning": "ddgs not available"}
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({"title": r.get("title"), "href": r.get("href"), "body": r.get("body")})
    return {"query": query, "results": results, "ts": _utc_iso()}


def fetch_url(url: str, timeout_s: float = 30.0) -> Dict[str, Any]:
    headers = {"User-Agent": "agentic-concierge/0.1 (+local)"}
    with httpx.Client(timeout=float(timeout_s), headers=headers, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        html = r.text
    text = trafilatura.extract(html) or ""
    return {"url": url, "fetched_at": _utc_iso(), "text": text[:200_000]}


def _utc_iso() -> str:
    from datetime import datetime
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
