"""Session-Lifecycle-Tests — Store-Ebene, ohne Netzwerk."""
from __future__ import annotations

import sqlite3
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


def test_legacy_database_migrates_without_losing_recent_history(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, persona_id TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'active', started_at REAL NOT NULL,
                ended_at REAL, turn_count INTEGER NOT NULL DEFAULT 0, summary TEXT
            );
            INSERT INTO sessions (id, persona_id, started_at) VALUES ('old-call', 'emma', 1);
            CREATE TABLE messages (
                id TEXT PRIMARY KEY, persona_id TEXT NOT NULL, role TEXT NOT NULL,
                content TEXT NOT NULL, job_id TEXT, created_at REAL NOT NULL
            );
            INSERT INTO messages VALUES ('old-message', 'emma', 'user', 'Legacy hello', NULL, 1);
        """)
    db = Store(path)
    try:
        db.conn.execute(
            "INSERT INTO personas (id, name, created_at, updated_at) VALUES ('emma', 'Emma', 1, 1)"
        )
        db.conn.commit()
        old = db.get_session("old-call")
        assert old["transport"] == "facetime"
        assert old["llm_provider"] == "" and old["llm_model"] == ""
        assert old["llm_override"] == 0
        telegram = db.create_session("emma", transport="telegram", transport_key="bot:1")
        db.add_message("emma", "user", "Telegram hello", session_id=telegram["id"])
        assert [m["content"] for m in db.get_history("emma")] == ["Legacy hello"]
        assert [m["content"] for m in db.get_history("emma", transport="telegram")] == ["Telegram hello"]
        assert [s["id"] for s in db.list_sessions(transport="facetime")] == ["old-call"]
    finally:
        db.close()


def test_transport_history_and_transcripts_are_isolated(store):
    persona = store.create_persona("Emma")
    pid = persona["id"]
    facetime = store.create_session(pid)
    telegram = store.create_session(pid, transport="telegram", transport_key="bot:1")
    store.add_message(pid, "user", "Legacy turn")
    store.add_message(pid, "user", "Call turn", session_id=facetime["id"])
    store.add_message(pid, "assistant", "Private Telegram turn", session_id=telegram["id"])
    assert [m["content"] for m in store.get_history(pid)] == ["Legacy turn", "Call turn"]
    assert [m["content"] for m in store.get_history(pid, transport="telegram")] == ["Private Telegram turn"]
    assert [m["content"] for m in store.session_transcript(facetime["id"])] == ["Call turn"]
    assert [m["content"] for m in store.session_transcript(telegram["id"])] == ["Private Telegram turn"]
    assert {s["id"] for s in store.list_sessions(pid)} == {facetime["id"], telegram["id"]}


def test_transport_resume_chooses_latest_active_for_its_key(store):
    persona = store.create_persona("Emma")
    old = store.create_session(persona["id"], transport="telegram", transport_key="bot:1")
    current = store.create_session(persona["id"], transport="telegram", transport_key="bot:1")
    store.create_session(persona["id"], transport="telegram", transport_key="bot:2")
    store.create_session(persona["id"], transport="facetime", transport_key="bot:1")
    assert store.get_transport_session("telegram", "bot:1")["id"] == current["id"]
    store.finalize_session_now(current["id"])
    assert store.get_transport_session("telegram", "bot:1")["id"] == old["id"]
    store.finalize_session_now(old["id"])
    assert store.get_transport_session("telegram", "bot:1") is None


def test_session_model_selection_survives_restart_without_mutating_persona(tmp_path):
    path = tmp_path / "snapshot.db"
    db = Store(path)
    persona = db.create_persona(
        "Emma", llm_provider="configured", llm_base_url="http://model-a/v1", llm_model="a"
    )
    session = db.create_session(persona["id"], transport="telegram", transport_key="bot:1")
    supplied = db.create_session(persona["id"], model_settings={
        "llm_provider": "configured", "llm_base_url": "", "llm_model": "override"
    }, model_override=True)
    assert supplied["llm_model"] == "override"
    assert session["llm_model"] == "a" and session["llm_base_url"] == "http://model-a/v1"
    assert session["llm_override"] == 0
    db.update_session_model(
        session["id"], llm_provider="opencode-go", llm_base_url="", llm_model=""
    )
    assert db.get_persona(persona["id"]) == persona
    db.close()
    reopened = Store(path)
    try:
        resumed = reopened.get_transport_session("telegram", "bot:1")
        assert resumed["id"] == session["id"]
        assert resumed["llm_provider"] == "opencode-go"
        assert resumed["llm_base_url"] == "" and resumed["llm_model"] == ""
        assert resumed["llm_override"] == 1
        assert reopened.get_session(supplied["id"])["llm_override"] == 1
        assert reopened.get_persona(persona["id"])["llm_model"] == "a"
    finally:
        reopened.close()


def test_session_override_can_return_to_persistent_inheritance(tmp_path):
    path = tmp_path / "override.db"
    db = Store(path)
    persona = db.create_persona("Emma", llm_provider="opencode-go", llm_model="persona-model")
    session = db.create_session(persona["id"], transport="telegram", transport_key="bot:1")
    message = db.add_message(persona["id"], "user", "Keep this turn", session_id=session["id"])
    db.close()

    db = Store(path)
    try:
        inherited = db.get_transport_session("telegram", "bot:1")
        assert inherited["id"] == session["id"] and inherited["llm_override"] == 0
        db.update_session_model(
            session["id"], llm_provider="configured", llm_base_url="", llm_model=""
        )
    finally:
        db.close()

    db = Store(path)
    try:
        explicit = db.get_transport_session("telegram", "bot:1")
        assert explicit["llm_override"] == 1
        assert (explicit["llm_provider"], explicit["llm_base_url"], explicit["llm_model"]) == (
            "configured", "", ""
        )
        db.update_persona(persona["id"], llm_model="updated-persona-model")
        assert db.get_session(session["id"]) == explicit
        db.update_session_model(
            session["id"], llm_provider="opencode-go", llm_base_url="",
            llm_model="updated-persona-model", model_override=False,
        )
    finally:
        db.close()

    db = Store(path)
    try:
        inherited = db.get_transport_session("telegram", "bot:1")
        assert inherited["id"] == session["id"] and inherited["llm_override"] == 0
        assert inherited["llm_model"] == "updated-persona-model"
        assert [row["id"] for row in db.session_transcript(session["id"])] == [message["id"]]
    finally:
        db.close()


@pytest.mark.parametrize("field, value", [
    ("llm_provider", "opencode-go"),
    ("llm_base_url", "http://changed.invalid/v1"),
    ("llm_model", "changed-persona-model"),
])
def test_session_model_update_rejects_stale_persona_selection(store, field, value):
    persona = store.create_persona("Emma", llm_model="persona-model")
    session = store.create_session(persona["id"], transport="telegram", transport_key="bot:1")
    store.add_message(persona["id"], "user", "Keep this turn", session_id=session["id"])
    transcript_before = store.session_transcript(session["id"])
    before = store.get_session(session["id"])
    changed_persona = store.update_persona(persona["id"], **{field: value})

    rejected = store.update_session_model(
        session["id"], llm_provider="configured", llm_base_url="", llm_model="chat-model",
        expected_persona_selection=persona,
    )
    assert rejected is None
    assert store.get_session(session["id"]) == before

    accepted = store.update_session_model(
        session["id"], llm_provider="configured", llm_base_url="", llm_model="chat-model",
        expected_persona_selection=changed_persona,
    )
    assert accepted["llm_override"] == 1
    assert accepted["llm_model"] == "chat-model"
    assert accepted["id"] == session["id"] and accepted["state"] == "active"
    assert store.session_transcript(session["id"]) == transcript_before


def test_legacy_model_overrides_are_migrated_only_once(tmp_path):
    path = tmp_path / "legacy-models.db"
    selections = [
        ("blank", "", "", "", 0),
        ("configured-default", "configured", "", "", 0),
        ("provider-only", "opencode-go", "", "", 1),
        ("custom-base", "configured", "http://custom.invalid/v1", "", 1),
        ("custom-model", "configured", "", "chosen-model", 1),
    ]
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, persona_id TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'active', started_at REAL NOT NULL,
                ended_at REAL, turn_count INTEGER NOT NULL DEFAULT 0, summary TEXT,
                transport TEXT NOT NULL DEFAULT 'facetime', transport_key TEXT,
                llm_provider TEXT NOT NULL DEFAULT '',
                llm_base_url TEXT NOT NULL DEFAULT '', llm_model TEXT NOT NULL DEFAULT ''
            );
        """)
        conn.executemany(
            "INSERT INTO sessions (id, persona_id, started_at, transport, transport_key,"
            " llm_provider, llm_base_url, llm_model) VALUES (?, 'emma', 1, 'telegram', ?,?,?,?)",
            [(sid, sid, provider, base, model) for sid, provider, base, model, _ in selections],
        )

    db = Store(path)
    try:
        for sid, provider, base, model, override in selections:
            session = db.get_session(sid)
            assert session["llm_override"] == override
            assert (session["llm_provider"], session["llm_base_url"], session["llm_model"]) == (
                provider, base, model
            )
        db.update_session_model(
            "custom-model", llm_provider="configured", llm_base_url="",
            llm_model="chosen-model", model_override=False,
        )
    finally:
        db.close()

    db = Store(path)
    try:
        assert db.get_session("custom-model")["llm_override"] == 0
        assert db.get_session("provider-only")["llm_override"] == 1
        assert db.get_session("configured-default")["llm_override"] == 0
    finally:
        db.close()


@pytest.mark.parametrize("started_at", [0, None], ids=["stale", "fresh"])
def test_web_calls_neither_lock_nor_finalize_telegram(store, monkeypatch, started_at):
    import importlib
    from fastapi.testclient import TestClient
    from studio import config

    monkeypatch.setattr(config, "DB_PATH", store.db_path)
    web = importlib.import_module("studio.deliver.app")
    monkeypatch.setattr(web, "store", store)
    monkeypatch.setattr(web, "_rt_holder", {})
    monkeypatch.setattr(web, "_rt_pool", [])
    monkeypatch.setattr(config, "STALE_CALL_GRACE_S", 5)
    persona = store.create_persona("Emma")
    telegram = store.create_session(persona["id"], transport="telegram", transport_key="bot:1")
    store.add_message(persona["id"], "user", "Telegram-only turn", session_id=telegram["id"])
    if started_at is not None:
        store.conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (started_at, telegram["id"]))
        store.conn.commit()
    with TestClient(web.app) as client:
        response = client.post("/api/calls/start", json={"persona_id": persona["id"]})
        assert response.status_code == 200
        call = response.json()
        assert call["transport"] == "facetime"
        assert store.get_session(telegram["id"])["state"] == "active"
        assert client.post("/api/calls/end", json={"session_id": telegram["id"]}).status_code == 404
        assert store.get_session(telegram["id"])["state"] == "active"
        assert client.get(f"/api/history/{persona['id']}").json() == []
        assert [s["id"] for s in client.get("/api/sessions").json()] == [call["id"]]
        assert client.get(f"/api/sessions/{telegram['id']}").status_code == 404
