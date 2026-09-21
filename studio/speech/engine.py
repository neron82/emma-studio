"""Realtime speech-to-speech engine: builds Emma's pipeline unit pool.

Wiring ported from the huggingface/speech-to-speech demo (s2s_pipeline.py),
adapted to Emma's stack:
  VAD (silero + smart-turn) → faster-whisper STT (resident singleton shared
  with /api/stt) → chat-completions LLM → OmniVoice OpenAI-compatible
  streaming TTS (PCM resampled to 16 kHz blocks).

The demo's speculative-turn / barge-in cancel semantics are preserved
untouched. Emma-specific pieces live here: persona/RAG/history wiring in
EmmaRealtimeService (persona_id on session.update, per-turn fact refresh,
message persistence) and the custom emma.visemes lip-sync timeline emitted
by VisemeTTSHandler.

The pool is built lazily on the first WebSocket connection (build_into) so
heavy model loading stays off the import path; a failed build surfaces as a
protocol error and is retried on the next connect.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from queue import Queue
from threading import Event
from typing import Any, Literal, Optional

import httpx
import numpy as np
from openai.types.realtime import ConversationItemCreateEvent, RealtimeErrorEvent
from pydantic import BaseModel, Field

from studio import config
from studio.agent.dialog import build_system_prompt
from studio.agent.memory import schedule_call_finalization
from studio.core.store import Store
from studio.deliver.realtime.emma_events import EmmaStatusEvent, EmmaVideoFrameEvent
from studio.deliver.realtime.pipeline_unit import PipelineUnit
from studio.deliver.realtime.service import RealtimeService
from studio.speech import stt, visemes, vlm
from studio.speech.chat import make_assistant_message, make_user_message
from studio.speech.chat_completions_language_model import ChatCompletionsApiModelHandler
from studio.speech.silent_turn_processor import SilentTurnProcessor
from studio.speech.visual_cycles import ALL_GATES, Cycle, VisualCycles
from studio.speech.core.cancel_scope import CancelScope
from studio.speech.core.events import AssistantOutputEvent, PipelineEvent
from studio.speech.core.messages import EndOfResponse, GenerateResponseRequest, TTSInput
from studio.speech.core.speculative_turns import SpeculativeTurnTracker
from studio.speech.core.thread_manager import ThreadManager
from studio.speech.core.transcript_logging import log_exception
from studio.speech.faster_whisper_handler import FasterWhisperSTTHandler
from studio.speech.openai_compatible_handler import OpenAICompatibleTTSHandler
from studio.speech.transcription_notifier import TranscriptionNotifier
from studio.speech.vad_handler import VADHandler

logger = logging.getLogger(__name__)

PIPELINE_SAMPLE_RATE = 16000  # VAD input rate and TTS output rate after resampling


# ── custom events ─────────────────────────────────────────────────────────────


class VisemeTimelineEvent(PipelineEvent):
    """Internal event: lip-sync timeline for one synthesized sentence.

    Travels on text_output_queue ahead of the sentence's audio blocks;
    EmmaRealtimeService translates it to the wire-level emma.visemes event.
    """

    type: Literal["emma_viseme_timeline"] = "emma_viseme_timeline"
    text: str = ""
    seq: int = 0
    duration_s: float = 0.0
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    turn_id: str | None = None
    turn_revision: int | None = None
    response_key: str | None = None


class StatusEvent(PipelineEvent):
    """Intern: eine Pipeline-Statuszeile → ``emma.status`` im Client.

    Wird von beliebigen Threads (Analyzer, TTS) auf die text_output_queue
    gelegt und vom Send-Loop in ein Wire-Event übersetzt.
    """

    type: Literal["emma_status"] = "emma_status"
    stage: str = ""
    detail: str = ""
    ms: float | None = None


class EmmaVisemeStamp(BaseModel):
    v: str
    t0: float
    t1: float


class EmmaVisemesEvent(BaseModel):
    """Server→client lip-sync timeline for one response sentence.

    Custom (non-OpenAI) protocol extension: the browser feeds the timeline
    to facetime-engine.setVisemeTimeline; offsets accumulate via seq +
    duration_s across sentences of one response.
    """

    type: Literal["emma.visemes"] = "emma.visemes"
    event_id: str
    response_id: Optional[str] = None
    item_id: Optional[str] = None
    seq: int = 0
    text: str = ""
    duration_s: float = 0.0
    timeline: list[EmmaVisemeStamp] = Field(default_factory=list)


# ── handler subclasses ────────────────────────────────────────────────────────


class SharedWhisperSTTHandler(FasterWhisperSTTHandler):
    """STT handler using Emma's resident whisper singleton.

    The demo handler builds its own WhisperModel (without device_index) and
    deletes it on cleanup; Emma pins the model to one GPU and shares it with
    the batch /api/stt path and the viseme transcriptions.
    """

    def setup(
        self,
        model_name: str = "",
        device: str = "",
        compute_type: str = "",
        gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        # model/device/compute are ignored: stt.get_model() owns them via config.
        gen_kwargs = self.adapt_gen_kwargs(dict(gen_kwargs or {}))
        self.start_language = self._normalize_language(gen_kwargs.get("language"))
        if self.start_language is None:
            gen_kwargs.pop("language", None)
        self.gen_kwargs = gen_kwargs
        stt.get_model()  # resident load, on the model's owner thread
        self.model = stt.bound_model()  # transcribe() forwards to that thread

    def cleanup(self) -> None:
        logger.info("Stopping SharedWhisperSTTHandler (resident model stays loaded)")


class VisemeTTSHandler(OpenAICompatibleTTSHandler):
    """Streaming TTS that prefixes each sentence with an emma.visemes timeline.

    To guarantee the timeline arrives before the sentence's first audio
    block, blocks are buffered until synthesis completes; whisper then stamps
    the assembled PCM with word timestamps, visemes.build_viseme_timeline
    converts them, and the audio is released. STUDIO_VISEMES=0 restores pure
    incremental streaming.
    """

    def setup(self, *args: Any, viseme_queue: Queue | None = None,
              status_cb: Any = None, **kwargs: Any) -> None:
        self.viseme_queue = viseme_queue
        self.status_cb = status_cb
        self._seq_by_response: dict[str, int] = {}
        super().setup(*args, **kwargs)

    def on_session_end(self) -> None:
        self._seq_by_response.clear()
        super().on_session_end()

    def process(self, tts_input):
        if isinstance(tts_input, EndOfResponse) and tts_input.response_key:
            self._seq_by_response.pop(tts_input.response_key, None)
        if self.viseme_queue is None or not isinstance(tts_input, TTSInput):
            yield from super().process(tts_input)
            return

        blocks: list[np.ndarray] = []
        for out in super().process(tts_input):
            blocks.append(out)  # cancellation/staleness ends super() early
        if blocks and not self.stop_event.is_set():
            event = self._build_event(tts_input, blocks)
            if event is not None:
                self.viseme_queue.put(event)
        yield from blocks

    def _build_event(self, tts_input: TTSInput, blocks: list[np.ndarray]) -> VisemeTimelineEvent | None:
        audio = np.concatenate(blocks)
        if audio.dtype == np.int16:
            floats = audio.astype(np.float32) / 32768.0
        else:
            floats = np.asarray(audio, dtype=np.float32)
        lang = (tts_input.language_code or config.STT_LANGUAGE).split("-")[0]
        try:
            words = stt.transcribe_words_array(floats, language=lang)
        except Exception as exc:
            log_exception(logger, "viseme transcription failed", exc)
            return None
        timeline = visemes.build_viseme_timeline(words)
        if not timeline:
            return None
        key = tts_input.response_key or ""
        seq = self._seq_by_response.get(key, 0) + 1
        self._seq_by_response[key] = seq
        if self.status_cb is not None:
            try:
                self.status_cb(
                    "TTS",
                    f"Satz {seq} synthetisiert: {tts_input.text[:48]!r} → "
                    f"{audio.shape[0] / PIPELINE_SAMPLE_RATE:.1f}s Audio, {len(timeline)} Viseme",
                )
            except Exception:                                 # noqa: BLE001
                pass
        return VisemeTimelineEvent(
            text=tts_input.text,
            seq=seq,
            duration_s=float(audio.shape[0]) / PIPELINE_SAMPLE_RATE,
            timeline=timeline,
            turn_id=tts_input.turn_id,
            turn_revision=tts_input.turn_revision,
            response_key=tts_input.response_key,
        )


# ── service subclass ──────────────────────────────────────────────────────────


_QUESTION_MARKERS = (
    "?", "wer ", "was ", "wann ", "wo ", "wie ", "warum ", "welche",
    "who ", "what ", "when ", "where ", "how ", "why ",
)  # fmt: skip


def _looks_like_question(text: str) -> bool:
    t = text.strip().lower()
    return any(m in t for m in _QUESTION_MARKERS)


def _web_search_sync(query: str) -> list[dict]:
    """Synchronous SearXNG fetch (mirror of studio.agent.search.search)."""
    if not config.SEARCH_ENABLED:
        return []
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(
                f"{config.SEARXNG_URL.rstrip('/')}/search",
                params={"q": query, "format": "json"},
            )
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError):
        return []
    return [
        {"title": r.get("title", ""), "snippet": r.get("content", "")}
        for r in (data.get("results") or [])[: config.SEARCH_TOP_K]
    ]


def _user_item_text(item: Any) -> str:
    """Plain text of a client-created user message item (typed input)."""
    if item is None or getattr(item, "role", None) != "user":
        return ""
    parts = getattr(item, "content", None) or []
    chunks = [getattr(p, "text", None) for p in parts]
    return " ".join(c for c in chunks if isinstance(c, str)).strip()


# Rahmung für den Kamera-Block im System-Prompt (bewusst so formuliert, dass
# Emma die Beobachtungen nicht in jeder Antwort kommentiert).
VISUAL_FRAMING = (
    "You will receive summarized descriptions of what the user is doing on their "
    "camera. Treat them as additional info, not as something you necessarily have "
    "to react to on every response."
)

# Rahmung für Silent-Turns: der User hat *nichts* gesagt, es liegt nur eine
# Zustandsbeschreibung vor. Das Modell entscheidet selbst (Persona!), ob es
# reagiert — Standard ist Stille, ausgedrückt durch ein Tag, das das Backend
# verwirft (kein TTS, kein Audio, keine sichtbare Änderung).
SILENT_FRAMING = (
    "The user has NOT spoken. What follows is a pure state description of what their camera "
    "showed over the last seconds — it is not a message addressed to you.\n"
    "Decide from your persona whether this is worth speaking up about. The default is NOT to "
    "answer: stay silent unless your persona would genuinely comment on it.\n"
    "If you stay silent, reply with EXACTLY this token and nothing else: {tag}\n"
    "Roughly {percent}% of these silent moments should stay unanswered. "
    "If you do speak, keep it short and in character, in the user's language."
)


def _ratio_of(persona: dict | None) -> float:
    """Silent-Verwurfsquote (0..1) der Persona, sonst globaler Default.

    Der Store liefert die Spalte bereits als REAL in 0..1 — hier wird nur noch
    gegen Ausreißer/None abgesichert.
    """
    try:
        return min(1.0, max(0.0, float((persona or {}).get("silent_ratio"))))
    except (TypeError, ValueError):
        return config.VLM_SILENT_RATIO_DEFAULT


class EmmaRealtimeService(RealtimeService):
    """RealtimeService plus Emma persona/RAG/history persistence.

    session.update may carry a custom ``persona_id`` (extra field): the chat
    is then seeded from the store's history, instructions come from the
    persona prompt, and the output voice becomes the persona's OmniVoice
    profile. Before each response is queued, fact cards are re-searched
    against the latest transcript (inline — fast, local); SearXNG fallback
    runs on a daemon thread and lands on the NEXT turn's instructions,
    because a blocking web call would stall the send loop's audio delivery.
    """

    def __init__(self, *, store: Store, default_voice: str = "emma_alicia", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._store = store
        self._default_voice = default_voice
        # Statuszeilen für das Debug-Overlay liegen bewusst NICHT auf der
        # text_output_queue: dort würden sie das Visem-Event pro Loop-Tick
        # verzögern (Audio könnte das Visem überholen). Sie haben eine eigene
        # Spur, die der Send-Loop zuletzt leert.
        self._status_pending: deque[tuple[str, str, float | None]] = deque(maxlen=200)
        self._status_lock = threading.Lock()
        self._lm_text_seen: dict[str, set[str]] = {}
        self._persona_by_conn: dict[str, str] = {}
        self._call_session_by_conn: dict[str, str] = {}  # conn_id → DB-Call-Session
        self._call_release_events: dict[str, threading.Event] = {}   # conn_id → set, wenn unregister abgeschlossen
        self._last_persisted: dict[str, str] = {}
        # Visueller Kontext: Verlaufs-Zyklen (5 s) statt einer Rollen-Summary.
        # Der Analyzer-Thread füllt sie; Gates markieren Aktivität; silent
        # cycles dürfen als Silent-Turn in den Prompt.
        self._frame: dict[str, tuple[float, bytes]] = {}          # (monotonic, JPEG)
        self._described_frame_at: dict[str, float] = {}           # gegen Doppelbeschreibung
        self._cycles: dict[str, VisualCycles] = {}
        self._pending_cycles: dict[str, deque[Cycle]] = {}        # noch zu beschreiben
        self._silent_keys: dict[str, str] = {}                    # response_key → conn_id
        self._silent_user_items: dict[str, tuple[str, str]] = {}  # response_key → (conn, item)
        self._silent_counters: dict[str, dict[str, int]] = {}     # answered/discarded
        self._silent_seq = 0
        self._persona_cache: dict[str, dict] = {}
        self._analyzer_thread: dict[str, threading.Thread] = {}
        self._analyzer_stop: dict[str, Event] = {}
        self._vlm_warned: set[str] = set()

    # ── persona ──────────────────────────────────────────────────────────

    def handle_session_update(self, conn_id: str, event) -> Optional[RealtimeErrorEvent]:
        error = super().handle_session_update(conn_id, event)
        if error is not None:
            return error
        extra = getattr(event.session, "model_extra", None) or {}
        persona_id = extra.get("persona_id")
        if isinstance(persona_id, str) and persona_id:
            error = self._apply_persona(conn_id, persona_id)
            if error is not None:
                return error
        # Call-Session-Bindung (extra Feld ``session_id``): alle Nachrichten dieser
        # Verbindung werden mit der DB-Session getaggt; bei Trennung wird die
        # Session defensiv finalisiert.
        self._apply_call_session(conn_id, extra.get("session_id"))
        return None

    def _apply_call_session(self, conn_id: str, session_id: Any) -> None:
        """Verbindung mit einer DB-Call-Session binden (best effort).

        Nur aktive Sessions derselben Persona werden gebunden; alles andere wird
        abgelehnt und protokolliert. Die Bindung ist die Grundlage für das
        Session-Tagging der persistierten Nachrichten und die defensive
        Finalisierung bei Trennung.
        """
        if not isinstance(session_id, str) or not session_id:
            return
        row = self._store.get_session(session_id)
        if row is None:
            logger.warning("session.update: unknown session_id %r — not bound", session_id)
            return
        if row["state"] != "active":
            logger.warning("session.update: session %s is not active — not bound", session_id)
            return
        persona_id = self._persona_by_conn.get(conn_id)
        if persona_id and row["persona_id"] != persona_id:
            logger.warning("session.update: session %s belongs to another persona — not bound", session_id)
            return
        self._call_release_events.setdefault(conn_id, threading.Event())
        self._call_session_by_conn[conn_id] = session_id
        logger.info("Connection %s bound to call session %s", conn_id, session_id)

    def _apply_persona(self, conn_id: str, persona_id: str) -> Optional[RealtimeErrorEvent]:
        persona = self._store.get_persona(persona_id)
        if persona is None:
            return self.make_error(f"Unknown persona_id {persona_id!r}", "unknown_persona")
        st = self._state(conn_id)
        cfg = st.runtime_config
        self._persona_by_conn[conn_id] = persona_id
        self._persona_cache[conn_id] = persona
        seeded = 0
        for row in reversed(self._store.get_history(persona_id, limit=2 * self._chat_size)):
            content = (row.get("content") or "").strip()
            if not content:
                continue
            if row.get("role") == "user":
                cfg.chat.add_item(make_user_message(content))
            else:
                cfg.chat.add_item(make_assistant_message(content))
            seeded += 1
        cfg.session.instructions = build_system_prompt(persona, [], None)
        voice = persona.get("voice_profile") or self._default_voice
        if cfg.session.audio is not None and cfg.session.audio.output is not None:
            cfg.session.audio.output.voice = voice
        logger.info(f"Persona {persona_id!r} applied ({seeded} history rows seeded)")
        return None

    def unregister(self, conn_id: str) -> None:
        self._finalize_call_session(conn_id)
        self._persona_by_conn.pop(conn_id, None)
        self._last_persisted.pop(conn_id, None)
        self._stop_analyzer(conn_id)
        for store_ in (self._frame, self._described_frame_at, self._cycles,
                       self._pending_cycles, self._silent_counters, self._persona_cache):
            store_.pop(conn_id, None)
        for key in [k for k, owner in self._silent_keys.items() if owner == conn_id]:
            self._silent_keys.pop(key, None)
        for key in [k for k, (owner, _item) in self._silent_user_items.items() if owner == conn_id]:
            self._silent_user_items.pop(key, None)
        self._vlm_warned.discard(conn_id)      # set, kein dict
        super().unregister(conn_id)
        ev = self._call_release_events.pop(conn_id, None)
        if ev is not None:
            ev.set()

    def _finalize_call_session(self, conn_id: str) -> None:
        """Defensive Call-Finalisierung bei Verbindungstrennung (Tab zu, Crash):
        Wenn noch eine aktive Call-Session angebunden ist, wird sie finalisiert und
        die Memory-Aufarbeitung gestartet. Das explizite ``POST /api/calls/end``
        hat Vorrang — der Übergangs-Guard in ``finalize_session_now`` verhindert
        Doppel-Einträge in der Memory.
        """
        session_id = self._call_session_by_conn.pop(conn_id, None)
        if not session_id:
            return
        try:
            row = self._store.finalize_session_now(session_id)
        except Exception as exc:  # noqa: BLE001 — Trennungspfad darf nicht sterben
            log_exception(logger, "defensive session finalization failed", exc)
            return
        if row is None:
            return
        logger.info("Call %s bei Verbindungstrennung finalisiert (kein explizites Ende)", session_id)
        try:
            schedule_call_finalization(self._store, row)
        except Exception as exc:  # noqa: BLE001
            log_exception(logger, "memory scheduling failed", exc)

    def conn_for_call_session(self, session_id: str) -> Optional[str]:
        """conn_id der an *session_id* gebundenen Verbindung (oder None)."""
        for conn_id, sid in self._call_session_by_conn.items():
            if sid == session_id:
                return conn_id
        return None

    def wait_call_session_released(self, conn_id: str, timeout: float) -> bool:
        """Blockiert, bis *conn_id* unregister'd ist (Drain komplett → das
        Transkript ist vollständig). ``True`` bei Timeout oder fehlendem Event."""
        ev = self._call_release_events.get(conn_id)
        if ev is None:
            return True
        return ev.wait(timeout)

    # ── Statuszeilen für das Debug-Overlay ───────────────────────────────

    def _status(self, stage: str, detail: str, ms: float | None = None) -> None:
        """Eine Statuszeile vormerken (thread-sicher, best effort).

        Der Send-Loop holt sie mit ``drain_status_events`` ab — nach Text und
        Audio, damit Diagnose-Zeilen die Pipeline nie bremsen.
        """
        with self._status_lock:
            self._status_pending.append((stage, detail, ms))

    def drain_status_events(self, conn_id: str) -> list:
        """Vorgemerkte Statuszeilen als Wire-Events (max. 20 pro Aufruf)."""
        with self._status_lock:
            pending = list(self._status_pending)[:20]
            for _ in pending:
                self._status_pending.popleft()
        return [
            EmmaStatusEvent(event_id=self._next_event_id(), stage=stage,
                            detail=detail, ms=ms, at=time.time())
            for stage, detail, ms in pending
        ]

    # ── visual context (VLM) ─────────────────────────────────────────────

    def handle_video_frame(self, conn_id: str, event: EmmaVideoFrameEvent) -> list:
        """Custom event: keep the newest camera frame and start the analyzer.

        Only the newest frame is held (drop-oldest): describing a backlog would
        waste GPU on stale pictures, and the integration only ever wants the
        current one.
        """
        if not config.VLM_ENABLED:
            return []
        try:
            jpeg = vlm.decode_frame(event.frame)
        except vlm.VLMError as exc:
            if conn_id not in self._vlm_warned:
                self._vlm_warned.add(conn_id)
                logger.warning("VLM: unbrauchbarer Frame verworfen (%s)", exc)
            return []
        self._frame[conn_id] = (time.monotonic(), jpeg)
        self._ensure_analyzer(conn_id)
        return []

    def _ensure_analyzer(self, conn_id: str) -> None:
        thread = self._analyzer_thread.get(conn_id)
        if thread is not None and thread.is_alive():
            return
        stop = Event()
        self._analyzer_stop[conn_id] = stop
        thread = threading.Thread(
            target=self._analyze_loop, args=(conn_id, stop),
            name=f"vlm-analyze-{conn_id[-6:]}", daemon=True,
        )
        self._analyzer_thread[conn_id] = thread
        thread.start()

    def _stop_analyzer(self, conn_id: str) -> None:
        stop = self._analyzer_stop.pop(conn_id, None)
        if stop is not None:
            stop.set()
        self._analyzer_thread.pop(conn_id, None)

    def _analyze_loop(self, conn_id: str, stop: Event) -> None:
        """Einmal pro Sekunde den neuesten Frame beschreiben + Summary pflegen.

        Läuft abseits des Send-Loops: die VLM-Aufrufe (150-250 ms) dürfen die
        Audio-Auslieferung nicht blockieren.
        """
        while not stop.wait(config.VLM_INTERVAL_S):
            try:
                self._analyze_once(conn_id)
            except Exception as exc:                          # noqa: BLE001
                self._warn_once(conn_id, f"Analyzer-Fehler: {type(exc).__name__}: {exc}")

    def _analyze_once(self, conn_id: str) -> None:
        """Ein Analyse-Schritt — direkt aufrufbar (Tests) ohne Thread."""
        frame = self._frame.get(conn_id)
        if frame is None:
            return
        frame_at, jpeg = frame
        if frame_at == self._described_frame_at.get(conn_id):
            return                                            # noch kein neuer Frame
        if time.monotonic() - frame_at > config.VLM_FRAME_MAX_AGE_S:
            return                                            # Kamera aus oder Netz hängt
        loading = not vlm.status()["loaded"]
        if loading:
            self._status("VLM", "Modell wird geladen … (einmalig, ~15 s)")
        t0 = time.monotonic()
        text = vlm.describe(jpeg)
        ms = (time.monotonic() - t0) * 1000
        if loading:
            self._status("VLM", f"Modell bereit ({vlm.status()['model']}, {vlm.status()['device']})")
        self._described_frame_at[conn_id] = frame_at
        if not text:
            return
        self._status("VLM", text, ms)

        cycles = self._cycles_for(conn_id)
        cycles.add_observation(text)
        closed = cycles.tick()                                # 5-s-Fenster abgelaufen?
        if closed is not None:
            self._queue_cycle(conn_id, closed)
        self._summarize_pending(conn_id)                      # Verläufe + Silent-Turns

    # ── Zyklen, Gates, Silent-Turns ──────────────────────────────────────

    def _cycles_for(self, conn_id: str) -> VisualCycles:
        cycles = self._cycles.get(conn_id)
        if cycles is None:
            cycles = VisualCycles(cycle_s=config.VLM_CYCLE_S, max_cycles=config.VLM_HISTORY_MAX)
            self._cycles[conn_id] = cycles
        return cycles

    def _queue_cycle(self, conn_id: str, cycle: Cycle) -> None:
        """Abgeschlossenen Zyklus zur Beschreibung vormerken (nie im Send-Loop!)."""
        self._pending_cycles.setdefault(conn_id, deque()).append(cycle)

    def gate(self, conn_id: str, gate: str) -> None:
        """Aktivitäts-Gate öffnen: laufender Zyklus wird sofort abgeschlossen.

        End-of-turn (``finish_response``) schließt den Aktivitätszyklus ab — das
        ist die "unterbrochen und abgeschlossen"-Semantik aus dem Konzept.
        """
        if not config.VLM_ENABLED:
            return
        cycles = self._cycles.get(conn_id)
        if cycles is None:
            return
        closed = cycles.gate(gate, True)
        if closed is not None:
            self._queue_cycle(conn_id, closed)

    def close_activity_cycle(self, conn_id: str) -> None:
        """Turn-Ende: offenen Zyklus abschließen und beschreiben lassen."""
        cycles = self._cycles.get(conn_id)
        if cycles is None:
            return
        closed = cycles.close_all()
        if closed is not None:
            self._queue_cycle(conn_id, closed)

    def _summarize_pending(self, conn_id: str) -> None:
        """Vorgemerkte Zyklen in Verlaufsbeschreibungen verwandeln (Analyzer-Thread)."""
        pending = self._pending_cycles.get(conn_id)
        if not pending:
            return
        cycles = self._cycles_for(conn_id)
        while pending:
            cycle = pending.popleft()
            try:
                t0 = time.monotonic()
                summary = vlm.summarize_temporal(cycle.texts())
                ms = (time.monotonic() - t0) * 1000
            except vlm.VLMError as exc:
                self._warn_once(conn_id, f"Verlaufsbeschreibung fehlgeschlagen: {exc}")
                continue
            if not summary:
                continue
            cycles.record(cycle, summary)
            kind = "silent" if cycle.silent else "aktiv"
            # Auch im Journal sichtbar: so ist nachvollziehbar, was Emma "sieht".
            logger.info("Verlauf (%s, %.0fs): %s", kind, cycle.seconds, summary)
            self._status("VLM", f"Verlauf ({kind}, {cycle.seconds:.0f}s): {summary}", ms)
            if cycle.silent:
                self._maybe_silent_turn(conn_id, cycle)

    def _maybe_silent_turn(self, conn_id: str, cycle: Cycle) -> None:
        """Silent-Turn: Zustandsbeschreibung ohne User-Beitrag an das Modell geben.

        Das Modell entscheidet selbst (auf Basis der Persona), ob es reagiert.
        Die Entscheidung wird unsichtbar ausgewertet — siehe
        :class:`SilentTurnProcessor` und :meth:`decide_silent`.
        """
        persona = self._persona_for(conn_id)
        if not persona or not bool(persona.get("silent_summary")):
            return
        st = self._state(conn_id)
        if st.in_response or st.response_pending:
            return                                            # läuft schon etwas → nicht stören
        cfg = st.runtime_config
        cfg.session.instructions = (
            build_system_prompt(persona, [], None)
            + "\n\n"
            + self._silent_framing(conn_id, persona)
        )
        self._silent_seq += 1
        request = GenerateResponseRequest(
            runtime_config=cfg,
            language_code=persona.get("language"),
            turn_id=f"silent-{self._silent_seq}",
            turn_revision=None,
        )
        # Der LM-Request braucht einen User-Turn (Qwen-Templates lehnen eine
        # Konversation ab, die mit einer Assistant-Nachricht endet). Dieser Turn
        # ist rein technisch und wird nach der Entscheidung wieder entfernt,
        # damit der Chat nicht mit Still-Momenten vollläuft.
        item = cfg.chat.add_item(make_user_message(self._silent_prompt(conn_id, persona)))
        self._silent_user_items[request.response_key] = (conn_id, item.id)
        self._silent_keys[request.response_key] = conn_id
        st.mark_response_pending(request.response_key)
        queue = self.text_prompt_queue
        if queue is None:
            return
        queue.put(request)
        self._status("SILENT", f"Zustandsbeschreibung gesendet ({cycle.seconds:.0f}s Verlauf)")

    def _silent_prompt(self, conn_id: str, persona: dict) -> str:
        """Zustandsbericht als ephemere User-Nachricht (Verlauf, älteste zuerst).

        Die Instruktionen (Default: keine Antwort, Tag) stehen im System-Prompt —
        hier steht nur, *was* die Kamera gezeigt hat.
        """
        history = self._cycles_for(conn_id).silent_history(config.VLM_INJECT_MAX)
        lines = "\n".join(f"- {c.summary}" for c in history) or "- (nothing notable)"
        return "Camera state report (I have not said anything):\n" + lines

    def _drop_silent_user_item(self, response_key: str) -> None:
        """Ephemeren User-Turn wieder entfernen (nach der Entscheidung)."""
        entry = self._silent_user_items.pop(response_key, None)
        if entry is None:
            return
        conn_id, item_id = entry
        cfg = self._state(conn_id).runtime_config
        try:
            cfg.chat.remove_user_message(item_id)
        except Exception as exc:                              # noqa: BLE001
            logger.debug("Silent-User-Turn konnte nicht entfernt werden: %s", exc)

    def _silent_framing(self, conn_id: str, persona: dict) -> str:
        """Prompt-Block: reine Zustandsbeschreibung + Entscheidungsauftrag + Tag."""
        ratio = _ratio_of(persona)
        history = self._cycles_for(conn_id).silent_history(config.VLM_INJECT_MAX)
        lines = "\n".join(f"- {c.summary}" for c in history) or "- (keine)"
        return (
            SILENT_FRAMING.format(tag=config.SILENT_TAG, percent=round(ratio * 100))
            + "\n\nCamera history of the last minute (oldest first):\n" + lines
        )

    def _persona_for(self, conn_id: str) -> dict | None:
        cached = self._persona_cache.get(conn_id)
        if cached is not None:
            return cached
        persona_id = self._persona_by_conn.get(conn_id)
        if not persona_id:
            return None
        persona = self._store.get_persona(persona_id)
        if persona is not None:
            self._persona_cache[conn_id] = persona
        return persona

    # ── Entscheidung über Silent-Antworten (Ratio-Enforcement) ───────────

    def is_silent_turn(self, response_key: str) -> bool:
        return response_key in self._silent_keys

    def decide_silent(self, response_key: str, text: str) -> str:
        """``"answer"`` oder ``"discard"`` — inkl. Mindest-Verwurfsquote.

        Hält die Persona-Quote als *Untergrenze*: das Modell darf seltener
        antworten als erlaubt (mehr Verwerfungen sind fein), aber nicht
        häufiger. Ohne erkannten Tag wird ebenfalls verworfen.
        """
        conn_id = self._silent_keys.pop(response_key, None)
        self._drop_silent_user_item(response_key)
        persona = self._persona_for(conn_id) if conn_id else None
        ratio = _ratio_of(persona)
        counters = self._silent_counters.setdefault(
            conn_id or "-", {"answered": 0, "discarded": 0}
        )
        wants_answer = bool((text or "").strip()) and config.SILENT_TAG not in (text or "").lower()
        total = counters["answered"] + counters["discarded"]
        # Quote NACH dieser Entscheidung: verworfene / (bisherige + diese)
        allowed = counters["discarded"] / (total + 1) >= ratio
        if wants_answer and allowed:
            counters["answered"] += 1
            self._status("SILENT", f"Antwort freigegeben ({counters['answered']} von {total + 1})")
            return "answer"
        counters["discarded"] += 1
        reason = "Tag no_answer_needed" if not wants_answer else f"Quote {ratio:.0%} erzwingt Stille"
        self._status(
            "SILENT",
            f"verworfen — {reason} ({counters['discarded']} von {total + 1} verworfen)",
        )
        return "discard"

    def _visual_context(self, conn_id: str) -> str:
        """Prompt-Block für *User*-Turns: Verlauf der letzten Zyklen.

        Die Rahmung ist wichtig: ohne sie kommentiert Emma die Kamera ständig
        ("ich sehe du sitzt vor dem Rechner"), obwohl das für den Dialog nur
        Hintergrundwissen ist.
        """
        if not config.VLM_ENABLED:
            return ""
        cycles = self._cycles.get(conn_id)
        if cycles is None:
            return ""
        recent = [c for c in cycles.history(config.VLM_INJECT_MAX) if c.summary]
        if not recent:
            return ""
        lines = "\n".join(f"- {cycle.summary}" for cycle in recent)
        return f"{VISUAL_FRAMING}\n\nCamera history (oldest first):\n{lines}"

    def _warn_once(self, conn_id: str, message: str) -> None:
        if conn_id not in self._vlm_warned:
            self._vlm_warned.add(conn_id)
            logger.warning("%s (Kamera-Kontext bleibt bis zum nächsten Versuch leer)", message)

    # ── per-turn RAG + user persistence ──────────────────────────────────

    def _call_session_id(self, conn_id: str) -> str | None:
        """Aktive DB-Call-Session der Verbindung (sonst ``None``)."""
        return self._call_session_by_conn.get(conn_id)

    def _on_transcription_completed(self, conn_id: str, event) -> list:
        persona_id = self._persona_by_conn.get(conn_id)
        transcript = (event.transcript or "").strip()
        if persona_id and transcript:
            self._refresh_instructions(conn_id, persona_id, transcript)
        events = super()._on_transcription_completed(conn_id, event)
        if persona_id and transcript:
            self._status("STT", f"final: {transcript[:90]}", None)
            try:
                self._store.add_message(persona_id, "user", transcript, session_id=self._call_session_id(conn_id))
            except Exception as exc:
                log_exception(logger, "user message persist failed", exc)
        return events

    def handle_conversation_item_create(self, conn_id: str, event: ConversationItemCreateEvent) -> list:
        """Client-created items (typed input) get the same treatment as voice turns.

        A typed message never reaches _on_transcription_completed, so without
        this hook it would miss the fact refresh and the history row — the LM
        would answer from the persona prompt alone.
        """
        events = super().handle_conversation_item_create(conn_id, event)
        persona_id = self._persona_by_conn.get(conn_id)
        text = _user_item_text(getattr(event, "item", None))
        if persona_id and text:
            self._refresh_instructions(conn_id, persona_id, text)
            try:
                self._store.add_message(persona_id, "user", text, session_id=self._call_session_id(conn_id))
            except Exception as exc:
                log_exception(logger, "user message persist failed", exc)
        return events

    def _refresh_instructions(self, conn_id: str, persona_id: str, transcript: str) -> None:
        cfg = self._state(conn_id).runtime_config
        persona = self._store.get_persona(persona_id)
        if persona is None:
            return
        try:
            facts = self._store.search_facts(transcript, persona_id=persona_id, limit=config.SEARCH_TOP_K)
        except Exception as exc:
            # Degrade to prompt-only rather than failing the turn, but never
            # silently: a programming error here would look like "no facts".
            log_exception(logger, "fact search failed", exc)
            facts = []
        cfg.session.instructions = build_system_prompt(persona, facts, None)
        visual = self._visual_context(conn_id)
        if visual:
            # Turn-complete-Injektion: der Analyzer-Thread hat die Zeile schon
            # fertig, hier wird nur gelesen (kein VLM-Aufruf im Send-Loop).
            cfg.session.instructions += "\n\n" + visual
            logger.info("Blickkontext für diesen Turn: %s", visual.replace("\n", " | "))
        self._status(
            "PROMPT",
            f"Prompt gebaut: {len(facts)} Fakten, Blickkontext {'ja' if visual else 'nein'}, "
            f"{len(cfg.session.instructions)} Zeichen",
        )
        if _looks_like_question(transcript) and config.SEARCH_ENABLED:
            def _bg_search() -> None:
                results = _web_search_sync(transcript)
                if results:
                    try:
                        base = build_system_prompt(persona, facts, results)
                        cfg.session.instructions = base + ("\n\n" + visual if visual else "")
                    except Exception:
                        pass
            threading.Thread(target=_bg_search, daemon=True, name="emma-web-search").start()

    # ── assistant persistence ────────────────────────────────────────────

    def finish_response(self, conn_id: str, status="completed", reason=None, response_key=None) -> list:
        events = super().finish_response(conn_id, status, reason, response_key=response_key)
        self._persist_assistant(conn_id)
        self._status("RESP", f"Antwort {status}" + (f" ({reason})" if reason else ""))
        self._lm_text_seen.pop(conn_id, None)
        # End-of-turn: Aktivitätszyklus abschließen (wird im Analyzer-Thread
        # beschrieben — hier nur vormerken, der Send-Loop darf nicht warten).
        self.close_activity_cycle(conn_id)
        return events

    def _persist_assistant(self, conn_id: str) -> None:
        persona_id = self._persona_by_conn.get(conn_id)
        if not persona_id:
            return
        try:
            buffer = self._state(conn_id).runtime_config.chat.buffer
            text = ""
            for item in reversed(buffer):
                if getattr(item, "role", None) == "assistant":
                    text = " ".join(
                        p.text for p in (getattr(item, "content", None) or []) if getattr(p, "text", None)
                    ).strip()
                    break
            if not text or self._last_persisted.get(conn_id) == text:
                return
            self._last_persisted[conn_id] = text
            self._store.add_message(persona_id, "assistant", text, session_id=self._call_session_id(conn_id))
        except Exception as exc:
            log_exception(logger, "assistant message persist failed", exc)

    # ── viseme translation ───────────────────────────────────────────────

    def _dispatch_pipeline_event(self, conn_id: str, event: PipelineEvent, *, wait_for_pending_reopen: bool):
        if isinstance(event, VisemeTimelineEvent):
            return self._on_visemes(conn_id, event)
        # Aktivitäts-Gate: öffnet ein Gate, wird der laufende Verlaufszyklus
        # sofort abgeschlossen (sein Verlauf ist damit festgehalten); der neue
        # Zyklus trägt das Gate und wird am Turn-Ende abgeschlossen.
        gate = VisualCycles.gate_for_event(getattr(event, "type", ""))
        if gate is not None:
            self.gate(conn_id, gate)
        if isinstance(event, AssistantOutputEvent):
            seen = self._lm_text_seen.setdefault(conn_id, set())
            key = getattr(event, "response_key", None) or "-"
            if key not in seen:
                seen.add(key)
                self._status("LM", f"erster Text vom Modell ({len(event.text)} Zeichen)")
        return super()._dispatch_pipeline_event(conn_id, event, wait_for_pending_reopen=wait_for_pending_reopen)

    def _on_visemes(self, conn_id: str, event: VisemeTimelineEvent) -> list:
        st = self._state(conn_id)
        if event.response_key is not None and event.response_key in st.closed_response_keys:
            return []
        response_id = st.current_response_id if event.response_key == st.current_response_key else None
        timeline = [
            EmmaVisemeStamp(v=str(s.get("v", "")), t0=float(s.get("t0", 0.0)), t1=float(s.get("t1", 0.0)))
            for s in event.timeline
        ]
        return [
            EmmaVisemesEvent(
                event_id=self._next_event_id(),
                response_id=response_id,
                item_id=st.pending_assistant_item_id,
                seq=event.seq,
                text=event.text,
                duration_s=event.duration_s,
                timeline=timeline,
            )
        ]


# ── unit + pool construction ──────────────────────────────────────────────────


def _build_unit(index: int, stop_event: Event) -> PipelineUnit:
    """One isolated pipeline: queues, service, handler chain (not threaded yet)."""
    should_listen = Event()
    response_playing = Event()
    cancel_scope = CancelScope()
    speculative_turns = SpeculativeTurnTracker()
    recv_audio_chunks_queue: Queue = Queue()
    spoken_prompt_queue: Queue = Queue()
    stt_output_queue: Queue = Queue()
    text_prompt_queue: Queue = Queue()
    lm_response_queue: Queue = Queue()
    lm_processed_queue: Queue = Queue()
    send_audio_chunks_queue: Queue = Queue()
    text_output_queue: Queue = Queue()

    service = EmmaRealtimeService(
        store=Store(config.DB_PATH),
        default_voice=config.TTS_VOICE_DEFAULT,
        text_prompt_queue=text_prompt_queue,
        should_listen=should_listen,
        chat_size=config.REALTIME_CHAT_SIZE,
        speculative_turns=speculative_turns,
        default_instructions=None,
    )

    vad = VADHandler(
        stop_event,
        queue_in=recv_audio_chunks_queue,
        queue_out=spoken_prompt_queue,
        setup_args=(should_listen,),
        setup_kwargs={
            "thresh": config.VAD_THRESH,
            "sample_rate": config.VAD_SAMPLE_RATE,
            "min_silence_ms": config.VAD_MIN_SILENCE_MS,
            "min_speech_ms": config.VAD_MIN_SPEECH_MS,
            "min_speech_continuation_ms": config.VAD_CONTINUATION_MS,
            "speech_pad_ms": config.VAD_SPEECH_PAD_MS,
            "enable_realtime_transcription": config.LIVE_TRANSCRIPTION,
            "realtime_processing_pause": config.LIVE_TRANSCRIPTION_INTERVAL_S,
            "speculative_reopen_ms": config.VAD_SPECULATIVE_REOPEN_MS,
            "unanswered_reopen_ms": config.VAD_UNANSWERED_REOPEN_MS,
            "smart_turn": config.SMART_TURN,
            "smart_turn_model_path": config.SMART_TURN_MODEL_PATH,
            "smart_turn_threshold": config.SMART_TURN_THRESHOLD,
            "smart_turn_max_wait_ms": config.SMART_TURN_MAX_WAIT_MS,
            "smart_turn_incomplete_delay_ms": config.SMART_TURN_INCOMPLETE_DELAY_MS,
            "text_output_queue": text_output_queue,
            "speculative_turns": speculative_turns,
        },
    )

    stt_handler = SharedWhisperSTTHandler(
        stop_event,
        queue_in=spoken_prompt_queue,
        queue_out=stt_output_queue,
        setup_kwargs={
            "model_name": config.STT_MODEL,
            "device": config.STT_DEVICE,
            "compute_type": "float16" if config.STT_DEVICE == "cuda" else "int8",
            "gen_kwargs": {
                "language": config.STT_LANGUAGE,
                "beam_size": config.STT_BEAM_SIZE,
                "condition_on_previous_text": False,
                "vad_filter": False,
            },
        },
    )
    stt_handler.speculative_turns = speculative_turns

    notifier = TranscriptionNotifier(
        stop_event,
        queue_in=stt_output_queue,
        queue_out=text_prompt_queue,
        setup_kwargs={"text_output_queue": text_output_queue, "should_listen": should_listen},
    )

    lm = ChatCompletionsApiModelHandler(
        stop_event,
        queue_in=text_prompt_queue,
        queue_out=lm_response_queue,
        setup_kwargs={
            "model_name": config.LLM_MODEL,
            "base_url": config.LLM_BASE_URL,
            # Empty key → None so the client uses its local-endpoint sentinel
            # instead of sending an empty Bearer header.
            "api_key": config.LLM_API_KEY or None,
            "gen_kwargs": {},
            "stream": True,
            # Empirically (22.08.): chat_template_kwargs.enable_thinking=false
            # empties minimax-m3's final content — never send provider extras.
            "disable_thinking": False,
            "cancel_scope": cancel_scope,
            "speculative_turns": speculative_turns,
            "request_timeout_s": config.REALTIME_LM_TIMEOUT_S,
            "stream_batch_sentences": 3,
        },
    )

    lm_processor = SilentTurnProcessor(
        stop_event,
        queue_in=lm_response_queue,
        queue_out=lm_processed_queue,
        setup_kwargs={
            "speculative_turns": speculative_turns,
            "text_output_queue": text_output_queue,
            # Silent-Turns werden gepuffert und je nach Entscheidung freigegeben
            # oder unsichtbar verworfen (Persona-Quote).
            "is_silent": service.is_silent_turn,
            "decide": service.decide_silent,
        },
    )

    tts = VisemeTTSHandler(
        stop_event,
        queue_in=lm_processed_queue,
        queue_out=send_audio_chunks_queue,
        setup_args=(should_listen,),
        setup_kwargs={
            "base_url": f"{config.OMNIVOICE_URL.rstrip('/')}/v1",
            "api_key": None,
            "model": config.TTS_MODEL,
            "voice": config.TTS_VOICE_DEFAULT,
            "language": None,
            "response_format": "pcm",
            "sample_rate": 24000,
            "stream": config.TTS_STREAM,
            "timeout": config.TTS_TIMEOUT_S,
            "blocksize": config.TTS_BLOCKSIZE,
            "cancel_scope": cancel_scope,
            "speculative_turns": speculative_turns,
            "viseme_queue": text_output_queue if config.VISEMES else None,
            "status_cb": service._status,
        },
    )

    handlers = [vad, stt_handler, notifier, lm, lm_processor, tts]
    for h in handlers:
        h.pipeline_index = index

    return PipelineUnit(
        index=index,
        service=service,
        cancel_scope=cancel_scope,
        should_listen=should_listen,
        response_playing=response_playing,
        input_queue=recv_audio_chunks_queue,
        output_queue=send_audio_chunks_queue,
        text_output_queue=text_output_queue,
        text_prompt_queue=text_prompt_queue,
        handlers=handlers,
    )


_stop_event = Event()
_build_lock = threading.Lock()
_manager: ThreadManager | None = None


def _ensure_nltk() -> None:
    try:
        import nltk

        for resource in ("tokenizers/punkt_tab", "taggers/averaged_perceptron_tagger_eng"):
            try:
                nltk.data.find(resource)
            except (LookupError, OSError):
                nltk.download(resource.split("/")[1], quiet=True)
    except Exception as exc:
        logger.warning("NLTK resource ensure failed: %s", exc)


def build_into(pool: list[PipelineUnit]) -> None:
    """Build the pipeline pool into *pool* and start handler threads.

    Called lazily by the /v1/realtime route on the first connection. Raises
    if nothing could be built (services down / missing models); a partial
    failure aborts the whole build so the pool never runs a broken chain.
    """
    global _manager
    with _build_lock:
        if pool:
            return
        _ensure_nltk()
        units: list[PipelineUnit] = []
        error: Exception | None = None
        for index in range(max(1, config.REALTIME_POOL_SIZE)):
            try:
                units.append(_build_unit(index, _stop_event))
            except Exception as exc:
                logger.error(f"Realtime unit {index} build failed: {type(exc).__name__}: {exc}")
                error = exc
                units = []
                break
        if not units:
            raise error or RuntimeError("realtime pipeline build failed")
        _manager = ThreadManager([h for u in units for h in u.handlers])
        _manager.start()
        pool.extend(units)
        logger.info(f"Realtime engine ready: {len(units)} pipeline unit(s)")


def shutdown() -> None:
    """Stop handler threads (called from the app lifespan teardown)."""
    global _manager
    _stop_event.set()
    if _manager is not None:
        _manager.stop()
        _manager = None