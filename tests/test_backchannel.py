"""Backchannel selector tests — mocked embeddings where needed."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio.speech import backchannel as bc


_HAS_CLIPS = (bc.FILLER_DIR / "clips" / "emma_alicia" / "manifest.json").is_file()
needs_clips = pytest.mark.skipif(not _HAS_CLIPS, reason="needs generated filler clips in data/fillers")


@needs_clips
@pytest.mark.needs_embed
def test_pick_returns_clip_for_good_news():
    entry = bc.pick_filler("Großartige Neuigkeiten, ich habe den Job bekommen!", voice="emma_alicia")
    assert entry is not None
    assert entry["category"] == "positive_surprise"
    assert entry["duration_s"] <= 2.5


@needs_clips
@pytest.mark.needs_embed
def test_pick_returns_empathy_for_bad_news():
    entry = bc.pick_filler("Ich bin heute echt fertig und erschöpft, alles lief schief.", voice="emma_alicia")
    assert entry is not None
    assert entry["category"] == "empathy_negative"


@needs_clips
def test_low_confidence_falls_back_to_neutral():
    # gibberish should not trigger a confident category — neutral fallback
    with patch.object(bc, "classify_energy", return_value=(None, 0.2)):
        entry = bc.pick_filler("xyzzy blorp", voice="emma_alicia")
    assert entry is not None
    assert entry["category"] in bc.NEUTRAL_FALLBACK_CATEGORIES


@needs_clips
@pytest.mark.needs_embed
def test_no_recent_repeats():
    texts = ["Erzähl mir was Lustiges", "Was hältst du von dem Witz", "Lach mal nicht so",
             "Das war ein echter Gag gestern"]
    seen = []
    for t in texts:
        e = bc.pick_filler(t, voice="emma_alicia")
        if e:
            seen.append(e["id"])
    assert len(seen) == len(set(seen)), f"repeats detected: {seen}"


def test_missing_manifest_returns_none(tmp_path):
    with patch.object(bc, "FILLER_DIR", tmp_path):
        assert bc.pick_filler("hallo", voice="nobody") is None
