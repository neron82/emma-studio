"""Sprach-Helfer ohne Netz/GPU: Sprachangaben → von Whisper akzeptierter Code."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio.speech.utils import whisper_language_code


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("de", "de"),            # ISO-Code (STT/Session-Pfad)
        ("de-DE", "de"),         # Region
        ("de_DE", "de"),
        ("de-auto", "de"),       # -auto-Suffix des Session-Configs
        ("EN", "en"),            # Groß-/Kleinschreibung
        ("German", "de"),        # Klarname (Persona-Feld, Silent-Turn-Pfad)
        ("german", "de"),
        ("english", "en"),
    ],
)
def test_language_specs_resolve_to_whisper_codes(value, expected):
    assert whisper_language_code(value) == expected


@pytest.mark.parametrize("value", ["", None, "  ", "auto", "klingon"])
def test_unknown_language_specs_return_none(value):
    """None → der Aufrufer nimmt seinen Default (config.STT_LANGUAGE)."""
    assert whisper_language_code(value) is None
