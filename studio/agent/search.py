"""Web search fallback via SearXNG — used only when facts don't cover a question.

One implementation, two entry points: the batch pipeline awaits :func:`search`,
the realtime engine calls :func:`search_sync` (it runs on pipeline threads and
must not block an event loop). Both return ``[{title, snippet}]`` and degrade to
an empty list on any failure.
"""
from __future__ import annotations

import asyncio

import httpx

from .. import config


def search_sync(query: str, *, top_k: int | None = None) -> list[dict]:
    """Return [{title, snippet}] from local SearXNG, empty on any failure."""
    if not config.SEARCH_ENABLED:
        return []
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(
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


async def search(query: str, *, top_k: int | None = None) -> list[dict]:
    """Async facade over :func:`search_sync` for the batch dialog pipeline."""
    return await asyncio.to_thread(search_sync, query, top_k=top_k)
