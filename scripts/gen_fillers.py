#!/usr/bin/env python3
"""Generate backchannel filler clips via OmniVoice + manifest.json.

Run once per voice profile. Output: data/fillers/clips/<voice>/<id>.wav
Manifest: category, phrase, energy, duration_s, file — used by the
backchannel selector at dialog time.
"""
import json
import sys
import time
import urllib.parse
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from studio.speech.tts import synthesize_sync  # noqa: E402

CORPUS = BASE / "data" / "fillers" / "corpus.json"
VOICE = sys.argv[1] if len(sys.argv) > 1 else "emma_alicia"
OUT_DIR = BASE / "data" / "fillers" / "clips" / VOICE


def wav_duration(path: Path) -> float:
    import wave
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def main() -> None:
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"voice": VOICE, "generated_at": time.time(), "clips": []}
    total = sum(len(c["phrases"]) for c in corpus["categories"].values())
    done = 0
    failures = []
    for cat, spec in corpus["categories"].items():
        for i, phrase in enumerate(spec["phrases"]):
            clip_id = f"{cat}_{i:02d}"
            out = OUT_DIR / f"{clip_id}.wav"
            if out.is_file():
                done += 1
                continue
            try:
                synthesize_sync(phrase, voice=VOICE, out_path=out)
                dur = wav_duration(out)
                manifest["clips"].append({
                    "id": clip_id,
                    "category": cat,
                    "phrase": phrase,
                    "energy": max(spec["energy"]),
                    "energy_range": spec["energy"],
                    "duration_s": round(dur, 2),
                    "file": f"clips/{VOICE}/{out.name}",
                })
                done += 1
                print(f"[{done}/{total}] {clip_id}: {dur:.1f}s")
            except Exception as e:  # noqa: BLE001
                failures.append((clip_id, str(e)[:120]))
                print(f"[{done}/{total}] {clip_id}: FAILED {e}")
                # keep going; one bad phrase shouldn't kill the batch
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\n{len(manifest['clips'])} clips in manifest, {len(failures)} failures")
    if failures:
        for cid, err in failures:
            print(f"  FAIL {cid}: {err}")


if __name__ == "__main__":
    main()
