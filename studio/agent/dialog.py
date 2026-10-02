"""Dialog agent: persona-driven conversation with fact-card RAG and web fallback.

The agent receives, per turn:
  - persona system prompt (personality)
  - top-k fact cards (FTS5 keyword match)
  - optional condensed web search results when the persona likely can't know
"""
from __future__ import annotations

from uuid import uuid4

import httpx

from .models import resolve_endpoint
from .memory import load_memory

_DIALOG_SESSION_ID = str(uuid4())


class AgentError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def build_system_prompt(persona: dict, facts: list[dict], search_results: list[dict] | None = None) -> str:
    parts = [persona["personality_prompt"].strip() or "You are a friendly assistant."]
    parts.append(f"You speak {persona.get('language') or 'German'}.")
    # Langer Kontext über Calls hinweg: eine Memory-Datei pro Persona, gefüllt
    # mit dichten Call-Zusammenfassungen (eine pro abgeschlossener Call).
    memory_text = load_memory(persona["id"]) if persona.get("id") else ""
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
    return "\n\n".join(parts)


async def chat(persona: dict, history: list[dict], user_message: str,
               facts: list[dict], search_results: list[dict] | None = None,
               *, session_id: str | None = None) -> str:
    """One LLM turn. `history` is prior [{role, content}] without the system prompt."""
    endpoint = resolve_endpoint(persona)
    session_id = session_id or f"{_DIALOG_SESSION_ID}:{persona.get('id', 'default')}"
    messages = [{"role": "system", "content": build_system_prompt(persona, facts, search_results)}]
    messages += history[-12:]  # keep context bounded
    messages.append({"role": "user", "content": user_message})
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{endpoint.base_url.rstrip('/')}/chat/completions",
            headers=endpoint.headers(session_id),
            json={"model": endpoint.model, "messages": messages},
        )
    if resp.status_code != 200:
        raise AgentError(f"LLM HTTP {resp.status_code}", status_code=resp.status_code)
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as e:
        raise AgentError("LLM response malformed") from e
