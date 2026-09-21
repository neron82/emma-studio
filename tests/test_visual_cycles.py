"""Tests der Verlaufs-Zyklen und Aktivitäts-Gates (reine Logik, kein Modell)."""
from __future__ import annotations

from studio.speech.visual_cycles import (
    GATE_RESPOND,
    GATE_SPEECH,
    GATE_TRANSCRIBE,
    VisualCycles,
)


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _cycles(clock: FakeClock, cycle_s: float = 5.0) -> VisualCycles:
    return VisualCycles(cycle_s=cycle_s, max_cycles=3, now=clock)


def test_cycle_closes_after_window_and_keeps_observations():
    clock = FakeClock()
    cycles = _cycles(clock)
    cycles.add_observation("A person sits in front of the camera.")
    clock.advance(2)
    cycles.add_observation("Her expression stays neutral.")
    assert cycles.tick() is None                    # Fenster noch nicht voll
    clock.advance(4)
    cycle = cycles.tick()
    assert cycle is not None
    assert cycle.seconds == 6.0
    assert cycle.texts() == ["A person sits in front of the camera.",
                             "Her expression stays neutral."]
    assert cycle.silent is True                     # kein Gate → Zustandsbeschreibung
    assert cycles.tick() is None                    # nichts mehr offen


def test_gate_open_closes_running_cycle_and_marks_activity():
    clock = FakeClock()
    cycles = _cycles(clock)
    cycles.add_observation("neutral look")
    clock.advance(1)
    closed = cycles.gate(GATE_SPEECH, True)         # User fängt an zu sprechen
    assert closed is not None
    assert closed.texts() == ["neutral look"]
    assert closed.silent is True                    # bis hierher war nichts los
    cycles.add_observation("she smiles widely")
    clock.advance(1)
    assert cycles.gate(GATE_SPEECH, True) is None    # Gate war schon offen
    activity = cycles.close_all()                    # End-of-turn
    assert activity is not None
    assert activity.gates == {GATE_SPEECH}
    assert activity.silent is False
    assert activity.texts() == ["she smiles widely"]


def test_second_gate_opens_new_cycle():
    clock = FakeClock()
    cycles = _cycles(clock)
    cycles.gate(GATE_SPEECH, True)
    cycles.add_observation("she smiles")
    closed = cycles.gate(GATE_RESPOND, True)         # Antwort läuft an
    assert closed is not None and closed.gates == {GATE_SPEECH}
    cycles.add_observation("she holds up a pen while answering")
    after = cycles.close_all()
    assert after is not None and after.gates == {GATE_RESPOND}
    assert after.silent is False


def test_empty_cycle_produces_nothing():
    clock = FakeClock()
    cycles = _cycles(clock)
    cycles.gate(GATE_TRANSCRIBE, True)
    clock.advance(1)
    assert cycles.close_all() is None                # keine Beobachtung → nichts


def test_history_order_limit_and_silent_filter():
    clock = FakeClock()
    cycles = _cycles(clock)
    for i in range(5):
        cycles.add_observation(f"obs {i}")
        clock.advance(6)
        cycle = cycles.tick()
        assert cycle is not None
        cycles.record(cycle, f"summary {i}")
    assert [c.summary for c in cycles.history()] == ["summary 2", "summary 3", "summary 4"]
    assert [c.summary for c in cycles.history(2)] == ["summary 3", "summary 4"]
    assert [c.summary for c in cycles.silent_history()] == ["summary 2", "summary 3", "summary 4"]

    # Aktivitätszyklus landet im Verlauf, aber nicht im Silent-Verlauf
    cycles.gate(GATE_RESPOND, True)
    cycles.add_observation("while answering")
    clock.advance(6)
    activity = cycles.tick()
    assert activity is not None
    cycles.record(activity, "active summary")
    assert cycles.history()[-1].summary == "active summary"
    assert "active summary" not in [c.summary for c in cycles.silent_history()]


def test_gate_for_event_mapping():
    assert VisualCycles.gate_for_event("speech_started") == GATE_SPEECH
    assert VisualCycles.gate_for_event("partial_transcription") == GATE_TRANSCRIBE
    assert VisualCycles.gate_for_event("assistant_text") == GATE_RESPOND
    assert VisualCycles.gate_for_event("emma_viseme_timeline") == GATE_RESPOND
    assert VisualCycles.gate_for_event("token_usage") is None
