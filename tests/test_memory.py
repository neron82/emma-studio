"""Memory-Module-Tests — ohne Netzwerk/LLM (Zusammenfassung wird ge-patcht)."""
from __future__ import annotations

import multiprocessing
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio import config
from studio.core.store import Store
from studio.agent import memory as mem
from studio.agent.dialog import build_system_prompt


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


def _personas_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", root)
    return root


def _age(store, session_id, seconds=100000):
    """Move a finished call into the past, behind the missing-summary grace window."""
    row = store.get_session(session_id)
    store.conn.execute("UPDATE sessions SET ended_at=? WHERE id=?",
                       ((row["ended_at"] or 0) - seconds, session_id))
    store.conn.commit()


def test_append_and_roundtrip(tmp_path, monkeypatch):
    _personas_root(tmp_path, monkeypatch)
    mem.append_summary("abc123", "User and Emma discussed the garden plan.")
    mem.append_summary("abc123", "Second entry about the roof repair.")

    text = mem.load_memory("abc123")
    entries = mem.parse_entries(text)
    assert len(entries) == 2
    assert "garden plan" in text and "roof repair" in text
    assert mem.memory_path("abc123").exists()


def _append_from_process(base_dir, barrier, summary):
    barrier.wait(timeout=10)
    mem.append_summary("shared", summary, base_dir=base_dir)


def test_summary_pool_keeps_concurrent_transport_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEMORY_MAX_WORDS", 1000)
    context = multiprocessing.get_context("fork")
    summaries = ["A garden plan.", "A roof repair.", "A travel plan.", "A book discussion."]
    barrier = context.Barrier(len(summaries))
    processes = [
        context.Process(target=_append_from_process, args=(tmp_path, barrier, summary))
        for summary in summaries
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=15)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    entries = mem.parse_entries(mem.load_memory("shared", base_dir=tmp_path))
    assert {entry.split("\n", 1)[1] for entry in entries} == set(summaries)


def test_oldest_entry_dropped_above_word_cap(tmp_path, monkeypatch):
    _personas_root(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "MEMORY_MAX_WORDS", 40)

    old = "old entry " * 20         # 40 Wörter + Header → allein schon über dem Limit
    mem.append_summary("abc123", old.strip())
    new = "new entry one two three four five six seven"
    mem.append_summary("abc123", new)

    text = mem.load_memory("abc123")
    entries = mem.parse_entries(text)
    # der älteste Eintrag wurde entfernt, der neue bleibt
    assert len(entries) == 1
    assert "new entry" in entries[0]
    assert "old entry" not in text


def test_single_entry_is_never_dropped(tmp_path, monkeypatch):
    _personas_root(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "MEMORY_MAX_WORDS", 5)
    long = "word " * 20
    mem.append_summary("abc123", long.strip())
    entries = mem.parse_entries(mem.load_memory("abc123"))
    assert len(entries) == 1 and "word" in entries[0]


def _word_blob(n):
    return " ".join(f"tok{i}" for i in range(n))


def test_summarize_in_range_returned_verbatim(monkeypatch):
    calls = []

    def fake(messages, endpoint, *, session_id):
        calls.append(messages)
        return _word_blob(250)

    monkeypatch.setattr(mem, "_llm_complete", fake)
    out = mem.summarize_call([{"role": "user", "content": "hi"},
                              {"role": "assistant", "content": "hello"}])
    assert len(out.split()) == 250 and len(calls) == 1


def test_summarize_short_output_retried_once(monkeypatch):
    replies = iter([_word_blob(8), _word_blob(240)])
    monkeypatch.setattr(mem, "_llm_complete", lambda messages, endpoint, *, session_id: next(replies))
    out = mem.summarize_call([{"role": "user", "content": "hi"}])
    assert len(out.split()) == 240


def test_summarize_still_short_raises_and_finalize_writes_nothing(tmp_path, monkeypatch, store):
    _personas_root(tmp_path, monkeypatch)
    monkeypatch.setattr(mem, "_llm_complete", lambda messages, endpoint, *, session_id: _word_blob(5))

    p = store.create_persona("Emma")
    session = store.create_session(p["id"])
    store.add_message(p["id"], "user", "hello", session_id=session["id"])

    with pytest.raises(RuntimeError, match="200-300"):
        mem.summarize_call([{"role": "user", "content": "hello"}])

    # finalize schluckt den Fehler und persistiert keine ungültige Memory-Eintrag
    mem.finalize_call_memory(store, session)
    assert not mem.memory_path(p["id"]).exists()
    assert not (store.get_session(session["id"]).get("summary") or "")


def test_summarize_long_output_truncated_to_300(monkeypatch):
    monkeypatch.setattr(mem, "_llm_complete", lambda messages, endpoint, *, session_id: _word_blob(400))
    out = mem.summarize_call([{"role": "user", "content": "hi"}])
    assert 200 <= len(out.split()) <= 300


def test_finalize_call_memory_end_to_end(tmp_path, monkeypatch, store):
    _personas_root(tmp_path, monkeypatch)
    p = store.create_persona("Emma")
    session = store.create_session(p["id"])
    store.add_message(p["id"], "user", "Hi Emma, how are you?", session_id=session["id"])
    store.add_message(p["id"], "assistant", "Great! What about you?", session_id=session["id"])

    monkeypatch.setattr(mem, "summarize_call",
                        lambda transcript, persona, *, session_id: "Summary: the user said hello to Emma.")
    mem.finalize_call_memory(store, session)

    row = store.get_session(session["id"])
    assert "the user said hello" in (row.get("summary") or "")
    assert "Summary: the user said hello to Emma." in mem.load_memory(p["id"])


def test_finalize_empty_call_writes_nothing(tmp_path, monkeypatch, store):
    _personas_root(tmp_path, monkeypatch)
    p = store.create_persona("Emma")
    session = store.create_session(p["id"])

    def no_llm(transcript):
        raise AssertionError("LLM darf bei leeren Calls nicht aufgerufen werden")

    monkeypatch.setattr(mem, "summarize_call", no_llm)
    mem.finalize_call_memory(store, session)

    assert not mem.memory_path(p["id"]).exists()


def test_schedule_call_finalization_thread(tmp_path, monkeypatch, store):
    _personas_root(tmp_path, monkeypatch)
    p = store.create_persona("Emma")
    session = store.create_session(p["id"])
    store.add_message(p["id"], "user", "Tell me about the weather.", session_id=session["id"])

    monkeypatch.setattr(mem, "summarize_call",
                        lambda transcript, persona, *, session_id: "Thread summary text.")
    thread = mem.schedule_call_finalization(store, session)
    thread.join(timeout=5)

    assert "Thread summary text." in mem.load_memory(p["id"])


def test_sessions_missing_summary_queues_only_calls_that_lost_theirs(tmp_path, monkeypatch, store):
    """The retry queue is exactly: finalized, transcript present, no summary."""
    _personas_root(tmp_path, monkeypatch)
    p = store.create_persona("Emma")

    done = store.create_session(p["id"])
    store.add_message(p["id"], "user", "summarized", session_id=done["id"])
    store.finalize_session_now(done["id"])
    store.set_session_summary(done["id"], "already summarized")

    nothing_said = store.create_session(p["id"])
    store.finalize_session_now(nothing_said["id"])

    still_running = store.create_session(p["id"])
    store.add_message(p["id"], "user", "in progress", session_id=still_running["id"])

    lost = store.create_session(p["id"])
    store.add_message(p["id"], "user", "lost call", session_id=lost["id"])
    store.finalize_session_now(lost["id"])

    other_lane = store.create_session(p["id"], transport="telegram")
    store.add_message(p["id"], "user", "other lane", session_id=other_lane["id"])
    store.finalize_session_now(other_lane["id"])

    assert [row["id"] for row in store.sessions_missing_summary("facetime")] == [lost["id"]]
    assert {row["id"] for row in store.sessions_missing_summary()} == {lost["id"], other_lane["id"]}
    assert store.sessions_missing_summary("facetime", exclude_ids=(lost["id"],)) == []
    assert len(store.sessions_missing_summary(limit=1)) == 1
    # A call that just ended may still be summarizing: the grace window skips it.
    assert store.sessions_missing_summary("facetime", older_than=3600) == []


def test_retry_missing_summaries_writes_the_entry_that_was_lost(tmp_path, monkeypatch, store):
    _personas_root(tmp_path, monkeypatch)
    p = store.create_persona("Emma")
    lost = store.create_session(p["id"])
    store.add_message(p["id"], "user", "The call that lost its summary.", session_id=lost["id"])
    store.finalize_session_now(lost["id"])
    _age(store, lost["id"])                      # the failure happened a while ago
    monkeypatch.setattr(mem, "summarize_call",
                        lambda transcript, persona, *, session_id: "Recovered summary text.")

    assert mem.retry_missing_summaries(store, transport="facetime") == 1

    assert (store.get_session(lost["id"]).get("summary") or "") == "Recovered summary text."
    assert "Recovered summary text." in mem.load_memory(p["id"])
    assert mem.retry_missing_summaries(store, transport="facetime") == 0   # nothing left


def test_retry_missing_summaries_leaves_a_running_summary_alone(tmp_path, monkeypatch, store):
    _personas_root(tmp_path, monkeypatch)
    p = store.create_persona("Emma")
    fresh = store.create_session(p["id"])
    store.add_message(p["id"], "user", "Still being summarized.", session_id=fresh["id"])
    store.finalize_session_now(fresh["id"])
    called = []

    def capture(transcript, persona, *, session_id):
        called.append(session_id)
        return "Late summary"

    monkeypatch.setattr(mem, "summarize_call", capture)

    assert mem.retry_missing_summaries(store, transport="facetime", older_than=3600) == 0
    assert called == []
    # Draining on demand stays possible with an open grace window.
    assert mem.retry_missing_summaries(store, transport="facetime", older_than=0) == 1
    assert called == [fresh["id"]]


def test_finalize_call_memory_uses_the_session_model_override(tmp_path, monkeypatch, store):
    """A /model choice is the model the call ran on, so it summarizes that call."""
    _personas_root(tmp_path, monkeypatch)
    p = store.create_persona("Emma")
    session = store.create_session(p["id"])
    store.update_session_model(session["id"], llm_provider="opencode-go",
                              llm_base_url="", llm_model="chosen-model")
    store.add_message(p["id"], "user", "Hi Emma, how are you?", session_id=session["id"])
    seen = {}

    def capture(transcript, persona, *, session_id):
        seen["persona"] = persona
        return "Summary via the session model."

    monkeypatch.setattr(mem, "summarize_call", capture)
    mem.finalize_call_memory(store, store.get_session(session["id"]))

    assert seen["persona"]["llm_provider"] == "opencode-go"
    assert seen["persona"]["llm_model"] == "chosen-model"
    assert seen["persona"]["name"] == "Emma"          # the rest of the persona survives
    assert "Summary via the session model." in mem.load_memory(p["id"])


def test_finalize_call_memory_keeps_persona_settings_without_an_override(tmp_path, monkeypatch, store):
    _personas_root(tmp_path, monkeypatch)
    p = store.create_persona("Emma")
    session = store.create_session(p["id"])
    store.add_message(p["id"], "user", "hello", session_id=session["id"])
    seen = {}

    def capture(transcript, persona, *, session_id):
        seen["persona"] = persona
        return "Summary via the persona."

    monkeypatch.setattr(mem, "summarize_call", capture)
    mem.finalize_call_memory(store, store.get_session(session["id"]))

    assert seen["persona"]["llm_provider"] == p["llm_provider"]
    assert seen["persona"]["llm_model"] == p["llm_model"]


def test_finalize_call_memory_survives_a_memory_write_failure(tmp_path, monkeypatch, store):
    """A failing memory write must not kill the daemon thread with a traceback."""
    _personas_root(tmp_path, monkeypatch)
    p = store.create_persona("Emma")
    session = store.create_session(p["id"])
    store.add_message(p["id"], "user", "Preserve this.", session_id=session["id"])
    monkeypatch.setattr(mem, "summarize_call",
                        lambda transcript, persona, *, session_id: "Summary text.")

    def fail_write(*args, **kwargs):
        raise OSError("private-provider-secret")

    monkeypatch.setattr(mem, "append_summary", fail_write)
    died = []
    monkeypatch.setattr(threading, "excepthook", lambda args: died.append(args))
    thread = mem.schedule_call_finalization(store, session)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert died == []                                     # no dead thread, no traceback
    assert (store.get_session(session["id"]).get("summary") or "") == "Summary text."
    assert not mem.memory_path(p["id"]).exists()


def test_system_prompt_injects_memory(tmp_path, monkeypatch):
    _personas_root(tmp_path, monkeypatch)
    mem.append_summary("pid123", "Persistent memory: user likes coffee.")

    prompt = build_system_prompt(
        {"id": "pid123", "personality_prompt": "You are Emma.", "language": "German"},
        [],
    )
    assert "Persistent memory: user likes coffee." in prompt


def test_system_prompt_without_memory_unchanged(tmp_path, monkeypatch):
    _personas_root(tmp_path, monkeypatch)
    prompt = build_system_prompt(
        {"id": "nope", "personality_prompt": "You are Emma.", "language": "German"},
        [],
    )
    assert "Persistent memory" not in prompt
