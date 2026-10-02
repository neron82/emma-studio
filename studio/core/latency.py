"""Latency instrumentation for batch dialog turns.

Per-phase wall time is measured by the pipeline (``run_call_turn``) and stored
on the job row (``timings_json``) so the benchmark script can read it back.
Phases:
  facts      — fact-card hybrid search
  search     — web fallback (only when triggered)
  llm        — agent LLM round-trip
  tts        — sentence-chunked synthesis (until last chunk done)
  tts_first  — first chunk ready (what the user perceives first)
  total      — whole turn
"""
from __future__ import annotations

import json

from ..core.store import Store


def _persist(store: Store, job_id: str, timings: dict | None) -> None:
    if timings is None:
        return
    store.conn.execute(
        "UPDATE jobs SET timings_json=? WHERE id=?",
        (json.dumps(timings), job_id),
    )
    store.conn.commit()
