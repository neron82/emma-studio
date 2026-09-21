"""Realtime WS protocol integration tests with stubbed pipeline handlers.

Proves router ↔ service ↔ client event flow without GPU or models: a
hand-built PipelineUnit (no handler threads) whose queues are driven
directly, standing in for VAD/STT/LLM/TTS output. Covers: persona seeding
via session.update extras, implicit response on final transcription, audio
delta framing, emma.visemes translation, barge-in response.cancel, and
message persistence.
"""
from __future__ import annotations

import base64
import threading
from queue import Empty, Queue
from threading import Event

import numpy as np
import pytest
from fastapi.testclient import TestClient

from studio.core.store import Store
from studio.deliver.realtime.pipeline_unit import PipelineUnit
from studio.deliver.realtime.websocket_router import create_app
from studio.speech.core.cancel_scope import CancelScope
from studio.speech.core.events import (
    AssistantOutputEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TranscriptionCompletedEvent,
)
from studio.speech.core.messages import AUDIO_RESPONSE_DONE, AudioOutput
from studio.speech.core.speculative_turns import SpeculativeTurnTracker
from studio.speech.engine import EmmaRealtimeService, VisemeTimelineEvent


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    p = s.create_persona("Emma Test", personality_prompt="Du bist Emma.", language="German")
    s.add_message(p["id"], "user", "Hallo Emma")
    s.add_message(p["id"], "assistant", "Hallo zurück!")
    return s, p["id"]


def _make_unit(store: Store) -> PipelineUnit:
    should_listen = Event()
    text_prompt_queue = Queue()
    text_output_queue = Queue()
    return PipelineUnit(
        index=0,
        service=EmmaRealtimeService(
            store=store,
            default_voice="emma_test",
            text_prompt_queue=text_prompt_queue,
            should_listen=should_listen,
            chat_size=4,
            speculative_turns=SpeculativeTurnTracker(),
            default_instructions=None,
        ),
        cancel_scope=CancelScope(),
        should_listen=should_listen,
        response_playing=Event(),
        input_queue=Queue(),
        output_queue=Queue(),
        text_output_queue=text_output_queue,
        text_prompt_queue=text_prompt_queue,
        handlers=[],
    )

@pytest.fixture
def session(store):
    db, persona_id = store
    unit = _make_unit(db)
    stop_event = Event()

    # Pass-through pump: consumes input_queue and forwards to output_queue so
    # the SESSION_END control message emitted on disconnect reaches the send
    # loop and the release completes (there are no real handler threads).
    def _pump() -> None:
        while not stop_event.is_set():
            try:
                item = unit.input_queue.get(timeout=0.05)
            except Empty:
                continue
            unit.output_queue.put(item)

    pump = threading.Thread(target=_pump, daemon=True)
    pump.start()
    app = create_app([unit], stop_event)
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime", headers={"sec-websocket-protocol": "realtime"}) as ws:
            created = ws.receive_json()
            assert created["type"] == "session.created"
            session_id = created["session"]["id"]
            ws.send_json({
                "type": "session.update",
                "event_id": "evt_setup",
                "session": {"type": "realtime", "persona_id": persona_id},
            })
            updated = ws.receive_json()
            assert updated["type"] == "session.updated"
            yield ws, unit, db, persona_id, session_id
    stop_event.set()
    pump.join(timeout=2)


def _recv_until(ws, wanted: str, *, max_events: int = 24) -> list[dict]:
    """Collect events until one of type *wanted* arrives (bounded for safety)."""
    events = []
    while len(events) < max_events:
        ev = ws.receive_json()
        events.append(ev)
        if ev["type"] == wanted:
            return events
    raise AssertionError(f"never saw {wanted}; got {[e['type'] for e in events]}")


def test_persona_seeds_instructions_history_and_voice(session):
    ws, unit, db, persona_id, session_id = session
    st = unit.service._state(session_id)
    cfg = st.runtime_config
    assert cfg.session.instructions and "Du bist Emma." in cfg.session.instructions
    # history seeded chronologically: user, assistant (+ persona system prompt)
    roles = [getattr(item, "role", None) for item in cfg.chat.buffer]
    assert "user" in roles and "assistant" in roles
    first_user = next(i for i in cfg.chat.buffer if getattr(i, "role", None) == "user")
    assert "Hallo Emma" in str(first_user)


def test_transcription_triggers_response_audio_and_persistence(session):
    ws, unit, db, persona_id, session_id = session
    unit.text_output_queue.put(SpeechStartedEvent(turn_id="t1", turn_revision=0))
    unit.text_output_queue.put(SpeechStoppedEvent(duration_s=1.2, audio_end_ms=1200, turn_id="t1", turn_revision=0))
    unit.text_output_queue.put(
        TranscriptionCompletedEvent(transcript="Hallo Emma", turn_id="t1", turn_revision=0)
    )

    events = _recv_until(ws, "conversation.item.input_audio_transcription.completed")
    types = [e["type"] for e in events]
    assert "input_audio_buffer.speech_started" in types
    assert "input_audio_buffer.speech_stopped" in types
    assert events[-1]["transcript"] == "Hallo Emma"

    # an LM request tagged with the pending response key was queued
    request = unit.text_prompt_queue.get_nowait()
    key = request.response_key
    gen = unit.cancel_scope.generation
    assert unit.service._state(session_id).response_pending
    # user turn persisted to the store
    history = db.get_history(persona_id, limit=10)
    assert any(m["role"] == "user" and m["content"] == "Hallo Emma" for m in history)

    # stub LM + TTS output tagged with the response key
    unit.text_output_queue.put(
        AssistantOutputEvent(
            text="Antwort.", turn_id="t1", turn_revision=0,
            cancel_generation=gen, response_key=key, output_sequence=1,
        )
    )
    block = np.arange(512, dtype=np.int16)
    unit.output_queue.put(AudioOutput(audio=block, cancel_generation=gen, response_key=key))
    unit.output_queue.put(AudioOutput(audio=AUDIO_RESPONSE_DONE, cancel_generation=gen, response_key=key))

    events = _recv_until(ws, "response.done")
    types = [e["type"] for e in events]
    assert "response.created" in types
    delta = next(e for e in events if e["type"] == "response.output_audio.delta")
    assert base64.b64decode(delta["delta"]) == block.tobytes()
    assert any(e["type"] == "response.output_audio.done" for e in events)
    assert events[-1]["response"]["status"] == "completed"


def test_viseme_event_translated_before_audio(session):
    ws, unit, db, persona_id, session_id = session
    unit.text_output_queue.put(SpeechStartedEvent(turn_id="t1", turn_revision=0))
    unit.text_output_queue.put(SpeechStoppedEvent(duration_s=1.0, audio_end_ms=1000, turn_id="t1", turn_revision=0))
    unit.text_output_queue.put(
        TranscriptionCompletedEvent(transcript="Test", turn_id="t1", turn_revision=0)
    )
    _recv_until(ws, "conversation.item.input_audio_transcription.completed")
    request = unit.text_prompt_queue.get_nowait()
    key = request.response_key
    gen = unit.cancel_scope.generation
    unit.text_output_queue.put(
        AssistantOutputEvent(
            text="Satz eins.", turn_id="t1", turn_revision=0,
            cancel_generation=gen, response_key=key, output_sequence=1,
        )
    )
    events = _recv_until(ws, "response.created")
    resp_id = events[-1]["response"]["id"]

    unit.text_output_queue.put(
        VisemeTimelineEvent(
            text="Satz eins.",
            seq=1,
            duration_s=0.8,
            timeline=[{"v": "e", "t0": 0.1, "t1": 0.3}, {"v": "o", "t0": 0.4, "t1": 0.6}],
            turn_id="t1",
            turn_revision=0,
            response_key=key,
        )
    )
    unit.output_queue.put(AudioOutput(audio=np.zeros(512, dtype=np.int16), cancel_generation=gen, response_key=key))
    unit.output_queue.put(AudioOutput(audio=AUDIO_RESPONSE_DONE, cancel_generation=gen, response_key=key))

    events = _recv_until(ws, "response.done")
    types = [e["type"] for e in events]
    assert "emma.visemes" in types
    assert types.index("emma.visemes") < types.index("response.output_audio.delta")
    vis = next(e for e in events if e["type"] == "emma.visemes")
    assert vis["seq"] == 1 and vis["text"] == "Satz eins."
    assert vis["response_id"] == resp_id
    assert vis["duration_s"] == pytest.approx(0.8)
    assert vis["timeline"][0] == {"v": "e", "t0": 0.1, "t1": 0.3}


def test_barge_in_speech_started_cancels_active_response(session):
    ws, unit, db, persona_id, session_id = session
    unit.text_output_queue.put(SpeechStartedEvent(turn_id="t1", turn_revision=0))
    unit.text_output_queue.put(SpeechStoppedEvent(duration_s=1.0, audio_end_ms=1000, turn_id="t1", turn_revision=0))
    unit.text_output_queue.put(
        TranscriptionCompletedEvent(transcript="Hallo", turn_id="t1", turn_revision=0)
    )
    _recv_until(ws, "conversation.item.input_audio_transcription.completed")
    request = unit.text_prompt_queue.get_nowait()
    key = request.response_key
    gen = unit.cancel_scope.generation
    unit.text_output_queue.put(
        AssistantOutputEvent(
            text="Antwort.", turn_id="t1", turn_revision=0,
            cancel_generation=gen, response_key=key, output_sequence=1,
        )
    )
    _recv_until(ws, "response.created")
    unit.output_queue.put(
        AudioOutput(audio=np.zeros(512, dtype=np.int16), cancel_generation=gen, response_key=key)
    )  # response now audible

    # user interrupts mid-response: server-VAD semantics terminate the
    # response with response.done(status="cancelled") — response.cancel is a
    # client-only event in the OpenAI Realtime protocol.
    unit.text_output_queue.put(SpeechStartedEvent(turn_id="t2", turn_revision=1))
    events = _recv_until(ws, "response.done")
    assert events[-1]["response"]["status"] == "cancelled"
    # speech_started follows the cancelled terminal (finish_response runs first)
    nxt = ws.receive_json()
    assert nxt["type"] == "input_audio_buffer.speech_started"


def test_invalid_client_event_gets_error(session):
    ws, unit, db, persona_id, session_id = session
    ws.send_json({"type": "bogus.event", "event_id": "evt_bad"})
    ev = ws.receive_json()
    assert ev["type"] == "error"
    assert ev.get("error") or ev.get("message")


def test_camera_frame_reaches_prompt_as_visual_context(session, monkeypatch):
    """Kamera-Frame → Analyzer → rollierende Summary → Injektion in die Instructions.

    Fake-VLM, damit der Test ohne GPU läuft; der Analyzer-Thread wird über ein
    sehr kurzes Intervall angesteuert (deckt Thread + Summary-Pfad mit ab).
    """
    import time as _time

    from studio import config
    from studio.speech import engine as engine_mod

    ws, unit, db, persona_id, session_id = session
    seen: dict = {}

    def fake_describe(jpeg: bytes) -> str:
        seen["jpeg"] = jpeg
        return f"Observation {len(seen)}: the person is typing."

    monkeypatch.setattr(engine_mod.vlm, "describe", fake_describe)
    monkeypatch.setattr(engine_mod.vlm, "summarize_temporal",
                        lambda obs: f"Temporal summary of {len(obs)} observations: typing at a desk.")
    monkeypatch.setattr(config, "VLM_INTERVAL_S", 0.05)
    monkeypatch.setattr(config, "VLM_FRAME_MAX_AGE_S", 5.0)
    monkeypatch.setattr(config, "VLM_CYCLE_S", 0.05)

    def _wait(pred, timeout=5.0) -> bool:
        end = _time.time() + timeout
        while _time.time() < end:
            if pred():
                return True
            _time.sleep(0.02)
        return False

    frame = base64.b64encode(b"\xff\xd8\xff\xe0fake-jpeg-payload").decode()
    ws.send_json({"type": "emma.video_frame", "event_id": "evt_frame", "frame": frame})
    assert _wait(lambda: session_id in unit.service._frame), "Frame kam nicht im Service an"
    assert _wait(lambda: seen.get("jpeg") == b"\xff\xd8\xff\xe0fake-jpeg-payload"), "Frame-Bytes verfälscht"
    # zweiter Frame lässt das (im Test 0,05 s kurze) Verlaufsfenster ablaufen
    _time.sleep(0.08)
    frame2 = base64.b64encode(b"\xff\xd8\xff\xe0fake-jpeg-payload-2").decode()
    ws.send_json({"type": "emma.video_frame", "event_id": "evt_frame2", "frame": frame2})
    assert _wait(lambda: bool(unit.service._cycles.get(session_id, None)
                              and unit.service._cycles[session_id].history())), "kein Verlauf erzeugt"

    # Turn abschließen → Instructions müssen den gerahmten Kamera-Block tragen
    unit.text_output_queue.put(SpeechStartedEvent(turn_id="t1", turn_revision=0))
    unit.text_output_queue.put(SpeechStoppedEvent(duration_s=1.0, audio_end_ms=1000,
                                                  turn_id="t1", turn_revision=0))
    unit.text_output_queue.put(TranscriptionCompletedEvent(transcript="Hallo", turn_id="t1", turn_revision=0))
    _recv_until(ws, "conversation.item.input_audio_transcription.completed")

    cfg = unit.service._state(session_id).runtime_config
    instructions = cfg.session.instructions
    assert engine_mod.VISUAL_FRAMING in instructions
    assert "Camera history (oldest first):" in instructions
    assert "typing at a desk" in instructions                # Verlaufstext im Prompt

    # Statuszeilen für das Debug-Overlay müssen beim Client ankommen
    status = _recv_until(ws, "emma.status")
    payloads = [e for e in status if e["type"] == "emma.status"]
    assert payloads, "kein emma.status Event erhalten"
    assert payloads[-1]["stage"] in ("VLM", "PROMPT", "STT", "LM", "RESP")
    assert payloads[-1]["detail"]


def test_unregister_cleans_visual_state(session):
    """Session-Ende räumt den Kamera-Zustand auf (und wirft nicht).

    Deckt einen echten Bug ab: ``_vlm_warned`` ist ein set, wurde aber wie ein
    dict behandelt (``set.pop(key, default)`` → TypeError beim Aufräumen).
    """
    import time as _time
    from collections import deque

    from studio.speech.visual_cycles import Cycle, VisualCycles

    ws, unit, db, persona_id, session_id = session
    svc = unit.service
    svc._frame[session_id] = (_time.monotonic(), b"jpeg")
    svc._cycles[session_id] = VisualCycles(cycle_s=0.05)
    svc._pending_cycles[session_id] = deque([Cycle(started_at=0.0)])
    svc._silent_counters[session_id] = {"answered": 1, "discarded": 3}
    svc._persona_cache[session_id] = {"id": persona_id}
    svc._silent_keys["resp-x"] = session_id
    svc._vlm_warned.add(session_id)

    svc.unregister(session_id)          # darf nicht werfen

    assert session_id not in svc._frame
    assert session_id not in svc._cycles
    assert session_id not in svc._pending_cycles
    assert session_id not in svc._silent_counters
    assert session_id not in svc._persona_cache
    assert "resp-x" not in svc._silent_keys
    assert session_id not in svc._vlm_warned


def test_prompt_has_no_camera_block_without_frames(session):
    """Ohne Frames bleibt der Prompt sauber — kein leerer Kamera-Block."""
    ws, unit, db, persona_id, session_id = session
    unit.text_output_queue.put(SpeechStartedEvent(turn_id="t1", turn_revision=0))
    unit.text_output_queue.put(SpeechStoppedEvent(duration_s=1.0, audio_end_ms=1000, turn_id="t1", turn_revision=0))
    unit.text_output_queue.put(TranscriptionCompletedEvent(transcript="Hallo", turn_id="t1", turn_revision=0))
    _recv_until(ws, "conversation.item.input_audio_transcription.completed")

    instructions = unit.service._state(session_id).runtime_config.session.instructions
    assert "Camera observations:" not in instructions


def test_silent_cycle_triggers_turn_with_decision_framing(session, monkeypatch):
    """Fenster ohne Gate → Verlaufsbeschreibung als Silent-Turn in den Prompt."""
    import time as _time

    from studio import config
    from studio.speech import engine as engine_mod

    ws, unit, db, persona_id, session_id = session
    svc = unit.service
    monkeypatch.setattr(engine_mod.vlm, "describe", lambda jpeg: "The person looks neutral.")
    monkeypatch.setattr(engine_mod.vlm, "summarize_temporal",
                        lambda obs: "First neutral, then she smiles and holds up a pen.")
    monkeypatch.setattr(config, "VLM_CYCLE_S", 0.05)      # 5-s-Fenster für den Test

    def _send_frame(tag: str) -> None:
        payload = base64.b64encode(b"\xff\xd8\xff\xe0" + tag.encode()).decode()
        ws.send_json({"type": "emma.video_frame", "event_id": f"evt_{tag}", "frame": payload})

    def _wait(pred, timeout=5.0) -> bool:
        end = _time.time() + timeout
        while _time.time() < end:
            if pred():
                return True
            _time.sleep(0.02)
        return False

    _send_frame("a")
    assert _wait(lambda: session_id in svc._frame)
    svc._analyze_once(session_id)
    _time.sleep(0.08)                                     # Fenster ablaufen lassen
    _send_frame("b")
    assert _wait(lambda: svc._frame[session_id][1].endswith(b"b"))
    svc._analyze_once(session_id)                         # schließt Zyklus + Silent-Turn

    assert _wait(lambda: not unit.text_prompt_queue.empty()), "kein Silent-Turn angefordert"
    request = unit.text_prompt_queue.get_nowait()
    assert svc.is_silent_turn(request.response_key)

    instructions = svc._state(session_id).runtime_config.session.instructions
    assert "has NOT spoken" in instructions
    assert config.SILENT_TAG in instructions
    assert "holds up a pen" in instructions                # Verlauf ist drin


def test_silent_ratio_enforces_minimum_discards(session):
    """Quote 0,8: vier von fünf Silent-Turns müssen verworfen werden."""
    ws, unit, db, persona_id, session_id = session
    svc = unit.service
    verdicts = []
    for i in range(5):
        key = f"resp-silent-{i}"
        svc._silent_keys[key] = session_id
        verdicts.append(svc.decide_silent(key, f"Ein echter Kommentar {i}"))
    assert verdicts == ["discard", "discard", "discard", "discard", "answer"]


def test_silent_tag_and_unknown_text_are_discarded(session):
    ws, unit, db, persona_id, session_id = session
    svc = unit.service
    for i, text in enumerate(["no_answer_needed", "", "NO_ANSWER_NEEDED", "ok no_answer_needed"]):
        key = f"resp-tag-{i}"
        svc._silent_keys[key] = session_id
        assert svc.decide_silent(key, text) == "discard"


def test_lazy_build_failure_surfaces_engine_unavailable():
    """First connect triggers build_pool; failure → error event + 1008 close."""
    from fastapi import FastAPI
    from contextlib import asynccontextmanager

    from starlette.websockets import WebSocketDisconnect

    from studio.deliver.realtime.websocket_router import register_realtime

    def raiser(pool: list) -> None:
        raise RuntimeError("models offline")

    holder: dict = {}
    stop_event = Event()

    @asynccontextmanager
    async def _lifespan(_: FastAPI) -> None:
        async with holder["cm"]():
            yield

    app = FastAPI(lifespan=_lifespan)
    holder["cm"] = register_realtime(app, [], stop_event, build_pool=raiser)
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/v1/realtime") as ws:
                ev = ws.receive_json()
                assert ev["type"] == "error"
                assert ev["error"]["type"] == "engine_unavailable"
                assert "models offline" in ev["error"]["message"]
                ws.receive_json()  # server closed after the error → raises


def test_typed_user_message_gets_facts_and_persistence(session):
    """Typed input bypasses STT, so it must get the fact refresh + history row itself."""
    ws, unit, db, persona_id, session_id = session
    db.add_fact("Der Aktivierungscode lautet BLAU-42.", persona_id=persona_id)

    ws.send_json({
        "type": "conversation.item.create",
        "event_id": "evt_item",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Wie lautet der Aktivierungscode?"}],
        },
    })
    events = _recv_until(ws, "conversation.item.created")
    assert events[-1]["item"]["role"] == "user"

    st = unit.service._state(session_id)
    assert "BLAU-42" in st.runtime_config.session.instructions
    assert any(
        m["role"] == "user" and "Aktivierungscode" in m["content"]
        for m in db.get_history(persona_id, limit=5)
    )