#!/usr/bin/env python3
"""E2E latency benchmark for Emma Studio dialog turns.

Runs N turns against the live studio API (text input, skipping STT — STT is
measured separately), reports per-phase stats and the perceived latency:
  perceived = max(filler_delay + filler_duration, tts_first)
The backchannel filler masks everything below its end.

Output: console table + JSON at data/benchmarks/e2e_latency.json
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import httpx

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from studio import config  # noqa: E402

API = f"http://127.0.0.1:{config.STUDIO_PORT}" if hasattr(config, "STUDIO_PORT") else "http://127.0.0.1:8890"
PERSONA_ID = "bb8b992fc5b1"
N_TURNS = int(sys.argv[1]) if len(sys.argv) > 1 else 6

UTTERANCES = [
    "Wie funktioniert eigentlich die Hybrid-Suche in deinem Gedächtnis?",
    "Erzähl mir kurz was über deine Fähigkeiten.",
    "Großartige Neuigkeiten, das Projekt ist fertig geworden!",
    "Was hältst du von kurzen Antworten?",
    "Ich bin heute ziemlich müde gewesen.",
    "Nenn mir einen Fun Fact über Sprachmodelle.",
]

FILLER_DELAY_S = 0.45  # frontend schedules playback at 450 ms


def main() -> None:
    results = []
    client = httpx.Client(timeout=120)
    persona_id = PERSONA_ID
    # resolve first persona if hardcoded id missing
    personas = client.get(f"{API}/api/personas").json()
    if not any(p["id"] == persona_id for p in personas):
        persona_id = personas[0]["id"]

    for i in range(N_TURNS):
        text = UTTERANCES[i % len(UTTERANCES)]
        t0 = time.monotonic()

        # filler pick (as the frontend would, parallel to the turn)
        t_fill = time.monotonic()
        fr = client.post(f"{API}/api/fillers/pick", json={"text": text, "persona_id": persona_id}).json()
        filler_pick_ms = (time.monotonic() - t_fill) * 1000

        job = client.post(
            f"{API}/api/jobs/turn",
            data={"persona_id": persona_id, "message": text},
        ).json()

        # poll until done
        while True:
            j = client.get(f"{API}/api/jobs/{job['id']}").json()
            if j["state"] in ("done", "failed"):
                break
            time.sleep(0.15)

        wall_s = time.monotonic() - t0
        timings = j.get("timings_json") and json.loads(j["timings_json"]) or {}
        tts_first = timings.get("tts_first")
        filler_dur = (fr.get("duration_s") or 0) if fr.get("url") else 0
        perceived = max(FILLER_DELAY_S + filler_dur, tts_first or 99)

        results.append({
            "turn": i + 1,
            "text": text[:40],
            "state": j["state"],
            "wall_s": round(wall_s, 2),
            "facts_ms": timings.get("facts", 0) * 1000,
            "search_ms": (timings.get("search") or 0) * 1000,
            "llm_ms": timings.get("llm", 0) * 1000,
            "tts_first_s": tts_first,
            "tts_total_s": timings.get("tts"),
            "filler": fr.get("phrase"),
            "filler_category": fr.get("category"),
            "filler_end_s": FILLER_DELAY_S + filler_dur if fr.get("url") else None,
            "perceived_first_audio_s": round(perceived, 2),
        })
        print(f"[{i+1}/{N_TURNS}] {j['state']:5s} wall {wall_s:4.1f}s | "
              f"llm {timings.get('llm',0):4.1f}s | tts_first {tts_first} | "
              f"filler '{fr.get('phrase')}' ends @ {results[-1]['filler_end_s']}s → "
              f"perceived {perceived:.2f}s")

    done = [r for r in results if r["state"] == "done"]
    print("\n===== E2E LATENCY SUMMARY =====")
    if done:
        llms = [r["llm_ms"] for r in done]
        firsts = [r["tts_first_s"] for r in done if r["tts_first_s"]]
        perceived = [r["perceived_first_audio_s"] for r in done]
        walls = [r["wall_s"] for r in done]
        rows = [
            ("LLM round-trip", [v / 1000 for v in llms]),
            ("TTS first chunk", firsts),
            ("Perceived first audio (with filler)", perceived),
            ("Total turn (server)", walls),
        ]
        print(f"{'phase':38s} {'min':>6s} {'median':>7s} {'max':>6s}")
        for name, vals in rows:
            if vals:
                print(f"{name:38s} {min(vals):5.2f}s {statistics.median(vals):6.2f}s {max(vals):5.2f}s")

    out = BASE / "data" / "benchmarks" / "e2e_latency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_at": time.time(),
        "n_turns": N_TURNS,
        "filler_delay_s": FILLER_DELAY_S,
        "results": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nDetails: {out}")


if __name__ == "__main__":
    main()
