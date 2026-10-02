"""Lange Erinnerungen der Persona: Call-Zusammenfassungen in ``memory.md``.

Nach jedem abgeschlossenen Gespräch kondensiert das LLM das Transkript
(User + Avatar, inkl. Silent-Turn-Antworten) in einen dichten Absatz
(Ziel 200–300 words). Der Absatz wird als Eintrag in die Persona-Memory
(``data/personas/<id>/memory.md``) angehängt und über ``build_system_prompt``
in künftige Gespräche aller Transports eingespeist.

Einträge sind durch ``## ``-Header getrennt, dadurch kann bei Überschreitung
von ``MEMORY_MAX_WORDS`` der älteste Eintrag entfernt werden.

Nach jedem Call finalisieren zwei Pfade (``POST /api/calls/end`` und der
defensive WS-Trennungspfad) denselben Call; nur der Gewinner des
``finalize_session_now``-Übergangs startet die Zusammenfassung — dadurch gibt
es genau einen Eintrag pro Call.
Facetime und Telegram teilen diesen Pool; ein Dateilock schützt gleichzeitige
Read/Trim/Write-Zyklen auch bei getrennten Bot- und Web-Prozessen.
"""
from __future__ import annotations

import fcntl
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import httpx

from studio import config
from .models import ModelEndpoint, resolve_endpoint

logger = logging.getLogger(__name__)

MEMORY_TITLE = "# Long-term memory"
_ENTRY_PREFIX = "## "


# ── Datei-Layout ─────────────────────────────────────────────────────────

def memory_path(persona_id: str, base_dir: str | Path | None = None) -> Path:
    root = Path(base_dir or config.DATA_DIR)
    return root / "personas" / persona_id / "memory.md"


def load_memory(persona_id: str, base_dir: str | Path | None = None) -> str:
    """Inhalt der Memory-Datei (ohne Titelzeile), leer wenn nicht vorhanden."""
    path = memory_path(persona_id, base_dir)
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Memory-Datei %s nicht lesbar: %s", path, exc)
        return ""
    lines = [ln for ln in text.splitlines() if not ln.startswith(MEMORY_TITLE)]
    return "\n".join(lines).strip()


def parse_entries(text: str) -> list[str]:
    """Memory-Inhalt in chronologisch geordnete Einträge zerlegen.

    Ein Eintrag besteht aus Headerzeile (``## ...``) und Body bis zum nächsten
    ``## ``-Header.
    """
    entries: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith(_ENTRY_PREFIX):
            if current:
                entries.append("\n".join(current).strip())
            current = [line]
        elif current:
            current.append(line)
    if current:
        entries.append("\n".join(current).strip())
    return [e for e in entries if e]


def count_words(text: str) -> int:
    return len(text.split())


def _entry_header(session: dict | None, fallback_ts: float | None = None) -> str:
    ts = (session or {}).get("started_at") or fallback_ts or time.time()
    when = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    sid = (session or {}).get("id", "?")
    return f"{_ENTRY_PREFIX}{when} — Call #{str(sid)[:8]}"


def append_summary(persona_id: str, summary: str, session: dict | None = None,
                   base_dir: str | Path | None = None) -> Path:
    """Call-Zusammenfassung anhängen; älteste Einträge bis zur Word-Grenze werfen."""
    path = memory_path(persona_id, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep a stable lock inode: Telegram and the web app may be separate
    # processes, and both rewrite this persona's shared summary pool.
    with path.with_name(".memory.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = load_memory(persona_id, base_dir)
        entry = f"{_entry_header(session)}\n{summary.strip()}"
        entries = parse_entries(existing) + [entry]
        body = "\n\n".join(entries)
        full = f"{MEMORY_TITLE}\n\n{body}\n"
        while count_words(full) > config.MEMORY_MAX_WORDS and len(entries) > 1:
            entries.pop(0)  # ältester Eintrag weg
            body = "\n\n".join(entries)
            full = f"{MEMORY_TITLE}\n\n{body}\n"
        # Readers see either the old or the complete new pool, never a partial write.
        temporary = path.with_name(f".memory.{uuid4().hex}.tmp")
        try:
            temporary.write_text(full, encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


# ── Zusammenfassung nach dem Call ────────────────────────────────────────

_SUMMARY_SYSTEM = (
    "You keep the long-term memory of an AI avatar named after its persona. "
    "Summarize a finished voice call between the avatar and its user into ONE "
    "dense paragraph of 200-300 words. Cover the main topics, decisions, facts "
    "learned about the user, and any unresolved threads. Write it as neutral "
    "context for future conversations — no dialogue, no greetings, no camera or "
    "visual descriptions, no meta commentary."
)


def _truncate_to_words(text: str, limit: int = 300) -> str:
    """Deterministisch auf *limit* Wörter kürzen; bevorzugt einen
    Satzendpunkt innerhalb des [200, limit]-Fensters."""
    words = text.split()
    if len(words) <= limit:
        return text
    cut = " ".join(words[:limit])
    last_dot = cut.rfind(".")
    if last_dot > 0 and len(cut[:last_dot].split()) >= 200:
        return cut[:last_dot].strip()
    return cut.strip()


def summarize_call(transcript: list[dict], persona: dict | None = None,
                   *, session_id: str | None = None) -> str:
    """LLM-Zusammenfassung eines Call-Transkripts (Liste von {role, content}).

    Garantiert 200-300 Wörter: zu kurze Ausgabe wird einmal mit ausdrücklicher
    Erweitern-Anweisung regeneriert; zu lange Ausgabe wird deterministisch auf
    300 Wörter gekürzt. Bleibt das Ergebnis außerhalb des Fensters, wird eine
    ``RuntimeError`` geworfen — es darf keine ungültige Memory-Eintrag entstehen.
    """
    endpoint = resolve_endpoint(persona)
    session_id = session_id or str(uuid4())
    lines = [f"{m['role']}: {m['content']}" for m in transcript if m.get("content")]
    prompt = (
        "Summarize the conversation below into ONE dense paragraph of 200-300 words. "
        "Cover main topics, decisions, facts learned about the user, and unresolved "
        "threads. Neutral context for future conversations — no dialogue, no greetings, "
        "no camera or visual descriptions, no meta commentary. Output only the paragraph.\n\n"
        "Conversation:\n" + "\n".join(lines)
    )
    text = _llm_complete([{"role": "system", "content": _SUMMARY_SYSTEM},
                          {"role": "user", "content": prompt}], endpoint,
                         session_id=session_id)
    if not text:
        raise RuntimeError("LLM lieferte keine Zusammenfassung")
    if len(text.split()) < 200:
        retry = (
            f"Your previous summary was only {len(text.split())} words. Rewrite it as ONE "
            "dense paragraph of 200-300 words, adding missing specifics (facts, names, "
            "dates, commitments, decisions, unresolved threads). Output only the paragraph.\n\n"
        )
        text = _llm_complete([{"role": "system", "content": _SUMMARY_SYSTEM},
                              {"role": "user", "content": retry + prompt}], endpoint,
                             session_id=session_id)
    if not text:
        raise RuntimeError("LLM lieferte keine Zusammenfassung")
    if len(text.split()) > 300:
        text = _truncate_to_words(text, 300)
    n = len(text.split())
    if not (200 <= n <= 300):
        raise RuntimeError(f"Zusammenfassung außerhalb von 200-300 Wörtern nach Retry: {n} Wörter")
    return text


def _llm_complete(messages: list[dict], endpoint: ModelEndpoint,
                  *, session_id: str) -> str:
    """Ein Chat-Completion-Call gegen die gewählte Persona-LLM."""
    payload = {"model": endpoint.model, "messages": messages, "temperature": 0.3}
    with httpx.Client(timeout=180.0) as client:
        resp = client.post(f"{endpoint.base_url.rstrip('/')}/chat/completions",
                           json=payload, headers=endpoint.headers(session_id))
        resp.raise_for_status()
        data = resp.json()
    return (data["choices"][0]["message"].get("content") or "").strip()


def finalize_call_memory(store, session: dict) -> None:
    """Call finalisieren: Transkript → Zusammenfassung → DB + Memory-Datei.

    Muss nur laufen, wenn der Aufrfer den ``finalize_session_now``-Übergang
    gewonnen hat (sonst doppelte Einträge).
    """
    transcript = store.session_transcript(session["id"])
    if not transcript:
        logger.info("Call %s ohne Transkript — keine Memory-Eintrag", session["id"])
        return
    try:
        persona = store.get_persona(session["persona_id"])
        summary = summarize_call(transcript, persona, session_id=session["id"])
    except Exception as exc:  # noqa: BLE001 — Call bleibt finalisiert, nur ohne Memory
        logger.exception("Memory-Zusammenfassung für Call %s fehlgeschlagen: %s", session["id"], exc)
        return
    store.set_session_summary(session["id"], summary)
    append_summary(session["persona_id"], summary, session=session)
    logger.info("Call %s in Memory (%d Wörter)", session["id"], count_words(summary))


def schedule_call_finalization(store, session: dict):
    """``finalize_call_memory`` in einem Daemon-Thread starten (LLM-Call dauert)."""
    thread = threading.Thread(
        target=finalize_call_memory, args=(store, session),
        name=f"memory-{str(session.get('id', ''))[:8]}", daemon=True,
    )
    thread.start()
    return thread
