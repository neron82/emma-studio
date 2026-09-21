"""Pipeline + router tests — mocked LLM/TTS/Comfy, no network, no GPU."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio.core.store import Store
from studio.core import pipeline


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "t.db")
    yield s
    s.close()


@pytest.fixture()
def persona(store):
    return store.create_persona("Emma", personality_prompt="You are Emma.")


def _mock_tts():
    """Patch tts.synthesize to write a fake wav."""
    async def fake(text, *, voice, out_path, language=None):
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"RIFF" + b"\0" * 2000)
        return p
    return fake


def test_call_turn_with_facts(store, persona):
    store.add_fact("User has a dog named Bruno", persona_id=persona["id"])
    job = store.create_job(persona["id"], "call_turn", "Wie heißt mein Hund?")

    async def fake_chat(persona, history, msg, facts, search_results=None):
        assert any("Bruno" in f["fact"] for f in facts), "fact card must reach the agent"
        assert search_results is None, "no web search when facts match"
        return "Dein Hund heißt Bruno."

    with patch.object(pipeline.agent_dialog, "chat", fake_chat), \
         patch.object(pipeline.tts, "synthesize", _mock_tts()):
        done = __import__("asyncio").run(pipeline.run_call_turn(store, job, []))
    assert done["state"] == "done"
    assert done["script"] == "Dein Hund heißt Bruno."
    assert done["speech_mode"] == "tts_pre"
    assert Path(done["speech_path"]).exists()


def test_call_turn_web_fallback_on_question(store, persona):
    job = store.create_job(persona["id"], "call_turn", "Wie spät ist es in Tokio?")

    async def fake_search(q, **kw):
        return [{"title": "Time", "snippet": "Tokio: UTC+9"}]

    async def fake_chat(persona, history, msg, facts, search_results=None):
        assert search_results and "Tokio" in search_results[0]["snippet"]
        return "In Tokio ist es jetzt X."

    with patch.object(pipeline.agent_dialog, "chat", fake_chat), \
         patch.object(pipeline.web_search, "search", fake_search), \
         patch.object(pipeline.tts, "synthesize", _mock_tts()):
        done = __import__("asyncio").run(pipeline.run_call_turn(store, job, []))
    assert done["state"] == "done"
    assert done["search_used"] == 1


def test_call_turn_no_search_for_smalltalk(store, persona):
    job = store.create_job(persona["id"], "call_turn", "Hallo!")

    async def fail_search(q, **kw):
        raise AssertionError("search must not run for smalltalk")

    async def fake_chat(persona, history, msg, facts, search_results=None):
        return "Hallo!"

    with patch.object(pipeline.agent_dialog, "chat", fake_chat), \
         patch.object(pipeline.web_search, "search", fail_search), \
         patch.object(pipeline.tts, "synthesize", _mock_tts()):
        done = __import__("asyncio").run(pipeline.run_call_turn(store, job, []))
    assert done["state"] == "done" and done["search_used"] == 0


def test_video_message_full_pipeline_mocked(store, persona, tmp_path):
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"png")
    store.update_persona(persona["id"], ref_image=str(ref))
    job = store.create_job(persona["id"], "video_message", "Grüß den User")

    class FakeResp:
        status_code = 200
        text = ""
        def json(self):
            return {"choices": [{"message": {"content": "Emma smiles and says hello"}}]}

    async def fake_render(**kw):
        out = Path(kw["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"mp4data")
        return out

    import httpx as _httpx
    with patch.object(_httpx.AsyncClient, "post", AsyncMock(return_value=FakeResp())), \
         patch.object(pipeline.speech_router, "choose_speech_mode",
                      AsyncMock(return_value="h3_native")), \
         patch.object(pipeline.comfy, "render", fake_render):
        done = __import__("asyncio").run(pipeline.run_video_message(store, job))

    assert done["state"] == "done"
    assert done["speech_mode"] == "h3_native"
    assert Path(done["render_path"]).exists()


def test_video_message_requires_ref_image(store, persona):
    job = store.create_job(persona["id"], "video_message", "test")

    class FakeResp:
        status_code = 200
        text = ""
        content = b"x" * 2000
        def json(self):
            return {"choices": [{"message": {"content": "script"}}]}

    import httpx as _httpx
    from unittest.mock import patch as _patch, AsyncMock as _AsyncMock
    with _patch.object(_httpx.AsyncClient, "post", _AsyncMock(return_value=FakeResp())), \
         _patch.object(pipeline.speech_router, "choose_speech_mode",
                       _AsyncMock(return_value="h3_native")):
        done = __import__("asyncio").run(pipeline.run_video_message(store, job))
    assert done["state"] == "failed"
    assert "ref_image" in done["error"]
