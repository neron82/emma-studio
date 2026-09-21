"""Web search fallback via SearXNG — used only when facts don't cover a question."""
from __future__ import annotations

import httpx

from .. import config


class SearchUnavailable(RuntimeError):
    pass


async def search(query: str, *, top_k: int | None = None) -> list[dict]:
    """Return [{title, snippet}] from local SearXNG, empty on any failure."""
    if not config.SEARCH_ENABLED:
        return []
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{config.SEARXNG_URL.rstrip('/')}/search",
                params={"q": query, "format": "json"},
            )
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError):
        return []
    k = top_k or config.SEARCH_TOP_K
    return [
        {"title": r.get("title", ""), "snippet": r.get("content", "")}
        for r in (data.get("results") or [])[:k]
    ]
