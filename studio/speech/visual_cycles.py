"""Temporale Blick-Zyklen: Verlaufsbeschreibungen + Aktivitäts-Gates.

Alle ``cycle_s`` Sekunden (Default 5) wird aus den Sekunden-Beobachtungen eine
*Verlaufsbeschreibung* erzeugt und gesammelt — also "erst neutraler Blick, dann
ein breites Lächeln, dann hebt sie einen Stift" statt eines Zustandsbildes
("sie schaut lächelnd und hält einen Stift"). Die Zyklen sind der Verlauf, den
der Dialog-Prompt bekommt.

Gates (``speech``, ``transcribe``, ``respond``) markieren, dass in dem Fenster
etwas passiert ist:

* Öffnet ein Gate, wird der laufende Zyklus **sofort abgeschlossen** (sein
  Verlauf ist damit festgehalten) und ein neuer begonnen — der Aktivitätszyklus.
* End-of-turn (``close_all``) schließt diesen Aktivitätszyklus ab.
* Zyklen ohne jedes Gate sind **silent**: der User hat nichts gesagt, es liegt
  nur eine Zustandsbeschreibung vor. Nur die dürfen als Silent-Turn in den
  Prompt.

Die Klasse enthält reine Logik und ruft selbst kein Modell: der Summarizer wird
injiziert (Caller/Test entscheidet, wie beschrieben wird).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

GATE_SPEECH = "speech"
GATE_TRANSCRIBE = "transcribe"
GATE_RESPOND = "respond"
ALL_GATES = (GATE_SPEECH, GATE_TRANSCRIBE, GATE_RESPOND)

Summarizer = Callable[[list[str]], str]


@dataclass
class Cycle:
    """Ein abgeschlossenes (oder offenes) Beobachtungsfenster."""

    started_at: float
    observations: list[tuple[float, str]] = field(default_factory=list)
    gates: set[str] = field(default_factory=set)
    ended_at: float | None = None
    summary: str = ""

    @property
    def silent(self) -> bool:
        """Ohne jedes Gate → reine Zustandsbeschreibung, kein User-Beitrag."""
        return not self.gates

    @property
    def seconds(self) -> float:
        return max(0.0, (self.ended_at or self.started_at) - self.started_at)

    def texts(self) -> list[str]:
        return [text for _, text in self.observations]

    def summary_line(self, width: int = 90) -> str:
        return f"{int(self.seconds):2d}s: {self.summary[:width]}"


class VisualCycles:
    """Zustand einer Session: offener Zyklus + gesammelter Verlauf."""

    def __init__(
        self,
        *,
        cycle_s: float = 5.0,
        max_cycles: int = 12,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        # Untergrenze nur gegen Unsinn in der Config — Tests fahren kürzere
        # Fenster (ms), deshalb bewusst klein halten.
        self.cycle_s = max(0.01, cycle_s)
        self.max_cycles = max(1, max_cycles)
        self._now = now
        self._open: Cycle | None = None
        self._done: list[Cycle] = []

    # ── Beobachtungen ────────────────────────────────────────────────────

    def add_observation(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        if self._open is None:
            self._open = Cycle(started_at=self._now())
        self._open.observations.append((self._now(), text))

    # ── Gates ────────────────────────────────────────────────────────────

    def gate(self, name: str, active: bool) -> Cycle | None:
        """Gate-Zustand setzen. Liefert den Zyklus, der dadurch abzuschließen ist.

        Öffnen: laufender Zyklus wird geschlossen (der Verlauf bis hierher bleibt
        erhalten), der neue Zyklus trägt das Gate.
        Schließen (``active=False``): nur das Gate entfernen — abgeschlossen
        wird am Turn-Ende über :meth:`close_all`.
        """
        if active:
            if self._open is not None and name in self._open.gates:
                return None                     # Gate war in diesem Zyklus schon offen
            closed = self._close_open()
            self._open = Cycle(started_at=self._now(), gates={name})
            return closed
        if self._open is not None:
            self._open.gates.discard(name)
            if not self._open.gates and not self._open.observations:
                self._open = None          # leerer Aktivitätszyklus: nichts zu tun
        return None

    def close_all(self) -> Cycle | None:
        """End-of-turn: offenen (Aktivitäts-)Zyklus abschließen."""
        return self._close_open()

    def tick(self) -> Cycle | None:
        """Periodischer Abschluss: nach ``cycle_s`` Sekunden Verlauf einkassieren."""
        if self._open is None:
            return None
        if self._now() - self._open.started_at < self.cycle_s:
            return None
        return self._close_open()

    def _close_open(self) -> Cycle | None:
        cycle, self._open = self._open, None
        if cycle is None:
            return None
        cycle.ended_at = self._now()
        if not cycle.observations:
            return None                    # nichts gesehen → nichts zu beschreiben
        return cycle

    # ── Verlauf ──────────────────────────────────────────────────────────

    def record(self, cycle: Cycle, summary: str) -> None:
        """Fertige Verlaufsbeschreibung in den Verlauf aufnehmen."""
        cycle.summary = (summary or "").strip()
        if not cycle.summary:
            return
        self._done.append(cycle)
        if len(self._done) > self.max_cycles:
            del self._done[:-self.max_cycles]

    def history(self, limit: int | None = None) -> list[Cycle]:
        return self._done[-limit:] if limit else list(self._done)

    def silent_history(self, limit: int | None = None) -> list[Cycle]:
        cycles = [c for c in self._done if c.silent and c.summary]
        return cycles[-limit:] if limit else cycles

    def open_cycle(self) -> Cycle | None:
        return self._open

    def describe_cycle(self, cycle: Cycle) -> str:
        """Verlaufsblock für den Prompt (älteste zuerst)."""
        return "\n".join(f"- {c.summary}" for c in self.history() if c.summary)

    @staticmethod
    def gate_for_event(event_type: str) -> str | None:
        """Pipeline-Event → Gate (None = kein Gate)."""
        if event_type in ("speech_started", "speech_stopped"):
            return GATE_SPEECH
        if event_type in ("transcription_completed", "partial_transcription"):
            return GATE_TRANSCRIBE
        if event_type in ("assistant_text", "emma_viseme_timeline", "response_generation_done"):
            return GATE_RESPOND
        return None

    def gates_of(self, cycles: Iterable[Cycle]) -> set[str]:
        out: set[str] = set()
        for cycle in cycles:
            out |= cycle.gates
        return out
