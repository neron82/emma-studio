"""Backchannel selector: pick a filler clip matching the user's message energy.

Classification via bge-m3 embeddings against per-category anchor texts
(no LLM round-trip — this must stay fast and cheap). Falls back to a safe
neutral filler when confidence is low or the embed server is down.

Anti-repetition: remembers recently used clips per category.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from .. import config

FILLER_DIR = config.DATA_DIR / "fillers"

# Anchor sentences per category — the user utterance is compared against these.
ANCHORS = {
    "positive_surprise": [
        "Großartige Neuigkeiten! Es ist unglaublich toll gelaufen!",
        "Ich habe eine fantastische Nachricht, ich freue mich so sehr!",
        "Das war richtig gut, super erfolgreich, wow!",
        "Es hat alles geklappt und ich bin total begeistert!",
    ],
    "empathy_negative": [
        "Ich bin heute echt fertig und erschöpft.",
        "Das lief leider total schief und ich bin frustriert.",
        "Mir geht es nicht gut, das war anstrengend und deprimierend.",
        "Leider hat alles nicht funktioniert, ich bin sauer.",
    ],
    "amusement": [
        "Haha du glaubst nicht was passiert ist, das war so lustig!",
        "Das ist ja zum Lachen, total absurd!",
        "Eine richtig lustige Geschichte ist mir passiert.",
        "Wie komisch, das ist doch witzig!",
    ],
    "thinking": [
        "Kannst du mir das genauer erklären?",
        "Wie funktioniert das eigentlich technisch?",
        "Was denkst du darüber, ich bin unsicher?",
        "Warum ist das so? Erklär mir das mal.",
    ],
    "concern": [
        "Macht ihr dir Sorgen um die Situation?",
        "Das bereitet mir Kopfzerbrechen und ist ernst.",
        "Ist das gefährlich oder problematisch?",
        "Ich mache mir Gedanken, ob das okay ist.",
    ],
}

NEUTRAL_FALLBACK_CATEGORIES = ("acknowledge", "thinking")
MIN_CONFIDENCE = 0.45
_recent: list[str] = []  # recently used clip ids (global ring)


def _load_manifest(voice: str) -> dict | None:
    p = FILLER_DIR / "clips" / voice / "manifest.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _embed(texts: list[str]) -> list[list[float]] | None:
    import httpx
    try:
        with httpx.Client(timeout=8) as client:
            resp = client.post(
                f"{config.EMBED_URL.rstrip('/')}/v1/embeddings",
                json={"input": texts, "model": config.EMBED_MODEL},
            )
            resp.raise_for_status()
            return [d["embedding"] for d in resp.json()["data"]]
    except Exception:  # noqa: BLE001
        return None


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def classify_energy(user_text: str) -> tuple[str | None, float]:
    """Return (category, confidence) for the user's message."""
    flat_anchors = [(cat, a) for cat, lst in ANCHORS.items() for a in lst]
    vecs = _embed([user_text] + [a for _, a in flat_anchors])
    if not vecs:
        return None, 0.0
    q = vecs[0]
    best_cat, best_score = None, -1.0
    cat_scores: dict[str, list[float]] = {}
    for (cat, _anchor), v in zip(flat_anchors, vecs[1:]):
        s = _cosine(q, v)
        cat_scores.setdefault(cat, []).append(s)
        if s > best_score:
            best_score, best_cat = s, cat
    # average of the category's top anchors gives stabler confidence
    conf = sum(sorted(cat_scores.get(best_cat, []), reverse=True)[:2]) / 2
    if conf < MIN_CONFIDENCE:
        return None, conf
    return best_cat, conf


def pick_filler(user_text: str, *, voice: str = "emma_alicia",
                max_duration_s: float = 2.5) -> dict | None:
    """Choose one filler clip. Returns manifest entry or None."""
    manifest = _load_manifest(voice)
    if not manifest or not manifest.get("clips"):
        return None
    clips = [c for c in manifest["clips"] if c["duration_s"] <= max_duration_s]
    if not clips:
        return None

    cat, conf = classify_energy(user_text)
    if cat is None or conf < MIN_CONFIDENCE:
        cat = random.choice(NEUTRAL_FALLBACK_CATEGORIES)

    pool = [c for c in clips if c["category"] == cat]
    if not pool:
        pool = [c for c in clips if c["category"] in NEUTRAL_FALLBACK_CATEGORIES]
    if not pool:
        pool = clips

    # avoid recent repeats; refill the exclusion window if pool is small
    fresh = [c for c in pool if c["id"] not in _recent] or pool
    choice = random.choice(fresh)

    _recent.append(choice["id"])
    del _recent[:-12]
    return choice


def filler_url(entry: dict) -> str:
    from urllib.parse import quote
    return f"/api/fillers/audio/{quote(entry['file'].split('/', 1)[1])}"
