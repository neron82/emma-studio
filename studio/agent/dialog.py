"""Dialog agent: persona-driven conversation with fact-card RAG and web fallback.

The agent receives, per turn:
  - persona system prompt (personality)
  - top-k fact cards (FTS5 keyword match)
  - optional condensed web search results when the persona likely can't know
"""
from __future__ import annotations

import asyncio
import json
import re
from uuid import uuid4

import httpx

from .memory import load_memory
from .memory_tools import (
    LIMIT_FALLBACK,
    LIMIT_INSTRUCTION,
    MAX_MEMORY_CALLS,
    MEMORY_TOOL_PROMPT,
    execute_memory_call,
    parse_memory_call,
    result_block,
)
from .models import resolve_endpoint

_DIALOG_SESSION_ID = str(uuid4())


class AgentError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def build_system_prompt(persona: dict, facts: list[dict], search_results: list[dict] | None = None, *, store=None) -> str:
    parts = [persona["personality_prompt"].strip() or "You are a friendly assistant."]
    parts.append(f"You speak {persona.get('language') or 'German'}.")
    # The database keeps summaries after older entries leave the bounded prompt pool.
    memory_text = (store.memory_summaries(persona["id"]) if store is not None else "")
    if not memory_text:
        memory_text = load_memory(persona["id"]) if persona.get("id") else ""
        if store is not None and memory_text:
            # Compatibility for pre-index summary files: resolve their old short
            # internal IDs to the stable public IDs without losing the context.
            sessions = store.list_sessions(persona["id"], limit=100000)
            by_prefix = {s['id'][:8]: s['memory_id'] for s in sessions}
            memory_text = re.sub(r'Call #([A-Za-z0-9]{8})',
                                 lambda m: 'Session ' + by_prefix.get(m[1], m[1]), memory_text)
    if memory_text:
        parts.append(
            "Long-term memory from previous calls (trust it as context):\n" + memory_text
        )
    if facts:
        lines = "\n".join(f"- {f['fact']}" for f in facts)
        parts.append(f"Known facts about the user (trust these):\n{lines}")
    if search_results:
        lines = "\n".join(f"- {r['title']}: {r['snippet']}" for r in search_results)
        parts.append(
            "Fresh web results for the current question (cite that you looked it up):\n" + lines
        )
    parts.append(
        "Keep answers conversational and short — they will be spoken aloud. "
        "Never invent facts beyond what is listed here or general knowledge."
    )
    parts.append(MEMORY_TOOL_PROMPT)
    return "\n\n".join(parts)


def _user_content(text: str, image_data_url: str | None):
    """OpenAI-compatible user content: plain text, or text + one image part.

    The vision model's server accepts ``{"type": "image_url", ...}`` with a
    ``data:`` URL (see strata ``serve/frontend.py``: JPEG/PNG pass straight in).
    """
    if not image_data_url:
        return text
    parts = []
    if text:
        parts.append({"type": "text", "text": text})
    parts.append({"type": "image_url", "image_url": {"url": image_data_url}})
    return parts


async def chat(persona: dict, history: list[dict], user_message: str,
               facts: list[dict], search_results: list[dict] | None = None,
               *, session_id: str | None = None, store=None,
               image_data_url: str | None = None, timeout: float = 300) -> str:
    """One user turn with private retrieval. History excludes the system prompt.

    ``image_data_url`` (optional) attaches one image to the current user turn as
    a ``data:`` URL; the vision-capable model then sees it. History stays text-only.
    """
    endpoint = resolve_endpoint(persona)
    session_id = session_id or f"{_DIALOG_SESSION_ID}:{persona.get('id', 'default')}"
    messages: list[dict] = [{"role": "system", "content": build_system_prompt(persona, facts, search_results, store=store)}]
    messages += history[-12:]  # keep context bounded
    messages.append({"role": "user", "content": _user_content(user_message, image_data_url)})
    async with httpx.AsyncClient(timeout=timeout) as client:
        for count in range(MAX_MEMORY_CALLS + 1):
            resp = await client.post(
                f"{endpoint.base_url.rstrip('/')}/chat/completions",
                headers=endpoint.headers(session_id),
                json={"model": endpoint.model, "messages": messages},
            )
            if resp.status_code != 200:
                raise AgentError(f"LLM HTTP {resp.status_code}", status_code=resp.status_code)
            try:
                message = resp.json()["choices"][0]["message"]
                text = (message.get("content") or "").strip()
                call = parse_memory_call(text)
                for native in message.get("tool_calls") or []:
                    call = call or parse_memory_call(json.dumps(native))
            except (KeyError, IndexError, TypeError, AttributeError) as exc:
                raise AgentError("LLM response malformed") from exc
            if call is None:
                return text
            if count == MAX_MEMORY_CALLS:
                return LIMIT_FALLBACK
            result = (await asyncio.to_thread(execute_memory_call, store, persona["id"], call)
                      if store is not None else {"error": "Memory retrieval unavailable"})
            # Normalize accidental native calls into the same provider-independent protocol.
            messages.append({"role": "assistant", "content": json.dumps(
                {"tool": call.name, "arguments": call.arguments})})
            messages.append({"role": "system", "content": result_block(call, result, count + 1)})
            if count + 1 == MAX_MEMORY_CALLS:
                messages.append({"role": "system", "content": LIMIT_INSTRUCTION})
    return LIMIT_FALLBACK
