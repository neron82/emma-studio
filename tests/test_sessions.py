"""Session-Lifecycle-Tests — Store-Ebene, ohne Netzwerk."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio.core.store import Store


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


def test_session_lifecycle(store):
    p = store.create_persona("Emma")
    session = store.create_session(p["id"])
    assert session["state"] == "active"

    row = store.get_session(session["id"])
    assert row is not None and row["persona_id"] == p["id"]


def test_finalize_now_is_atomic_single_winner(store):
    """finalize_session_now liefert die Zeile nur dem Gewinner — der
    Verlierer (paralleles Hangup/Disconnect) erhält None, damit genau
    eine Zusammenfassung pro Call geplant wird."""
    p = store.create_persona("Emma")
    s1 = store.create_session(p["id"])

    winner = store.finalize_session_now(s1["id"])
    assert winner is not None and winner["state"] == "finalized"

    # paralleler Zweitaufrufer (Hangup/Disconnect-Rennen) erhält None
    loser = store.finalize_session_now(s1["id"])
    assert loser is None

    # unbekannte Session → ebenfalls None
    assert store.finalize_session_now("does-not-exist") is None


def test_finalize_idempotent_via_public_api(store):
    p = store.create_persona("Emma")
    s = store.create_session(p["id"])
    assert store.finalize_session(s["id"])["state"] == "finalized"
    # zweiter Aufruf bleibt ein no-op (Zeile existiert bereits finalisiert)
    assert store.finalize_session(s["id"])["state"] == "finalized"


def test_messages_tagged_to_session(store):
    p = store.create_persona("Emma")
    s = store.create_session(p["id"])
    m1 = store.add_message(p["id"], "user", "Hallo Emma", session_id=s["id"])
    m2 = store.add_message(p["id"], "assistant", "Hallo! Wie kann ich helfen?", session_id=s["id"])
    assert m1["session_id"] == s["id"] and m2["role"] == "assistant"
