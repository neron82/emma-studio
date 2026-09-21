"""Tests des Silent-Turn-Prozessors: Entscheidungen dürfen unsichtbar bleiben."""
from __future__ import annotations

from queue import Queue
from threading import Event

from studio.speech.core.messages import (
    AssistantTextPart,
    EndOfResponse,
    LLMResponseChunk,
    TTSInput,
)
from studio.speech.silent_turn_processor import SilentTurnProcessor


def _processor(decide, silent_keys):
    queue_out: Queue = Queue()
    proc = SilentTurnProcessor(
        Event(),
        queue_in=Queue(),
        queue_out=queue_out,
        setup_kwargs={
            "speculative_turns": None,
            "text_output_queue": Queue(),
            "is_silent": lambda key: key in silent_keys,
            "decide": decide,
        },
    )
    return proc, queue_out


def _chunk(text: str, key: str) -> LLMResponseChunk:
    return LLMResponseChunk(parts=[AssistantTextPart(text=text)], response_key=key)


def _collect(proc, items):
    out = []
    for item in items:
        out.extend(proc.process(item))
    return out


def test_normal_response_streams_unchanged():
    proc, _ = _processor(lambda key, text: "answer", silent_keys=set())
    out = _collect(proc, [_chunk("Hallo!", "resp-normal"), EndOfResponse(response_key="resp-normal")])
    assert any(isinstance(o, TTSInput) for o in out)
    assert any(isinstance(o, EndOfResponse) for o in out)
    assert not any(getattr(o, "cleanup_only", False) for o in out)


def test_silent_tag_is_discarded_without_any_output():
    calls: list[tuple[str, str]] = []

    def decide(key, text):
        calls.append((key, text))
        return "discard"

    proc, _ = _processor(decide, silent_keys={"resp-silent"})
    out = _collect(proc, [
        _chunk("no_answer_needed", "resp-silent"),
        EndOfResponse(response_key="resp-silent"),
    ])
    # Nur der Lifecycle-Aufräumer darf übrig bleiben: kein TTS, kein Text-Event.
    assert calls == [("resp-silent", "no_answer_needed")]
    assert not any(isinstance(o, TTSInput) for o in out)
    assert [type(o).__name__ for o in out] == ["EndOfResponse"]
    assert out[0].cleanup_only is True


def test_silent_answer_is_released_after_decision():
    proc, _ = _processor(lambda key, text: "answer", silent_keys={"resp-silent"})
    out = _collect(proc, [
        _chunk("Schöner Stift!", "resp-silent"),
        EndOfResponse(response_key="resp-silent"),
    ])
    assert any(isinstance(o, TTSInput) for o in out)
    assert any(isinstance(o, EndOfResponse) and not o.cleanup_only for o in out)


def test_silent_forced_discard_hides_generated_text():
    """Ratio erzwingt Stille: der bereits generierte Text darf nirgends auftauchen."""
    proc, _ = _processor(lambda key, text: "discard", silent_keys={"resp-silent"})
    out = _collect(proc, [
        _chunk("Ich sehe du hältst einen Stift", "resp-silent"),
        EndOfResponse(response_key="resp-silent"),
    ])
    assert not any(isinstance(o, TTSInput) for o in out)
    assert [type(o).__name__ for o in out] == ["EndOfResponse"]
