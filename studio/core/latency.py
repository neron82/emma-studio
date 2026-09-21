"""Latency instrumentation for dialog turns.

Measures per-phase wall time on the server and stores it on the job row
(timings_json). Phases:
  stt        — audio transcription (call_turn via /api/stt only)
  facts      — fact-card hybrid search
  search     — web fallback (only when triggered)
  llm        — agent LLM round-trip
  tts        — sentence-chunked synthesis (until last chunk done)
  tts_first  — first chunk ready (what the user perceives first)
"""
from __future__ import annotations

import time
from contextlib import contextmanager

from ..core.store import Store


def job_timings(store: Store, job_id: str) -> dict:
    row = store.conn.execute(
        "SELECT timings_json FROM jobs WHERE id=?", (job_id,)
    ).fetchone()
    import json
    return json.loads(row[0]) if row and row[0] else {}


@contextmanager
def phase(store: Store, job_id: str, name: str, timings: dict | None = None):
    """Time a phase and record into the job's timings dict."""
    t0 = time.monotonic()
    yield
    dt = time.monotonic() - t0
    if timings is not None:
        timings[name] = round(dt, 3)
    _persist(store, job_id, timings)


def _persist(store: Store, job_id: str, timings: dict | None) -> None:
    if timings is None:
        return
    import json
    store.conn.execute(
        "UPDATE jobs SET timings_json=? WHERE id=?",
        (json.dumps(timings), job_id),
    )
    store.conn.commit()
