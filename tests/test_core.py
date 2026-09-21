"""Store + agent unit tests — no network, no GPU."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio.core.store import Store
from studio.agent.dialog import build_system_prompt


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


def test_persona_crud(store):
    p = store.create_persona("Emma", personality_prompt="You are Emma.", language="German")
    assert p["name"] == "Emma"
    p2 = store.update_persona(p["id"], ref_image="/tmp/x.png", voice_sample="/tmp/v.mp3")
    assert p2["ref_image"] == "/tmp/x.png"
    assert any(x["id"] == p["id"] for x in store.list_personas())


def test_job_lifecycle(store):
    p = store.create_persona("E")
    j = store.create_job(p["id"], "video_message", "say hello")
    assert j["state"] == "draft"
    j = store.set_job_state(j["id"], "scripted", script="...")
    assert j["state"] == "scripted"
    j = store.set_job_state(j["id"], "done", render_path="/tmp/out.mp4", speech_mode="tts_pre")
    assert j["render_path"] == "/tmp/out.mp4"
    assert store.list_jobs()[0]["id"] == j["id"]
    # invalid state rejected by CHECK constraint
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        store.set_job_state(j["id"], "bogus")


def test_facts_fts_and_scoping(store):
    store.add_fact("User has a dog named Bruno", persona_id=None)
    pid = store.create_persona("X")["id"]
    store.add_fact("Persona X likes jazz", persona_id=pid)
    hits = store.search_facts("Bruno")
    assert len(hits) == 1 and "Bruno" in hits[0]["fact"]
    # persona-scoped search includes global facts, ranks persona facts first
    hits = store.search_facts("jazz OR Bruno", persona_id=pid)
    assert hits[0]["persona_id"] == pid
    # garbage query must not raise (may still return semantic hits)
    result = store.search_facts('"unbalanced (query')
    assert isinstance(result, list)
    assert store.count_facts() == 2
    fid = store.search_facts("Bruno")[0]["id"]
    assert store.delete_fact(fid) is True


def test_system_prompt_composition(store):
    persona = {"personality_prompt": "You are Emma.", "language": "German"}
    facts = [{"fact": "User has a dog"}]
    search_results = [{"title": "T", "snippet": "S"}]
    sp = build_system_prompt(persona, facts, search_results)
    assert "You are Emma." in sp and "German" in sp
    assert "User has a dog" in sp and "looked it up" in sp
