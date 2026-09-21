"""Silent-Turn-Prozessor: Entscheidungen unsichtbar machen.

Bei Silent-Turns (Kamera-Zustandsbeschreibung, der User hat nichts gesagt)
entscheidet das Modell selbst, ob es antwortet. Damit die Entscheidung für den
User *unsichtbar* bleibt, wird die Ausgabe nicht gestreamt, sondern gepuffert:

* Bis ``EndOfResponse`` liegt der komplette Text im Puffer — es geht weder ein
  Assistant-Event noch ein TTS-Input in die Pipeline.
* Dann entscheidet ``decide(response_key, text)``:

  - ``"answer"``   → gepufferte Chunks normal weiterreichen (Text + TTS laufen
                     wie bei jedem Turn).
  - ``"discard"``  → nur der Lifecycle-Aufräumer läuft weiter
                     (``EndOfResponse(cleanup_only=True)``). Der Send-Loop
                     schließt die Response damit still: kein Audio, kein
                     ``response.created``, kein Eintrag im Transkript.

Damit ist ein verworfener Silent-Turn von außen nicht zu unterscheiden von
"es ist nichts passiert" — genau die Anforderung.
"""
from __future__ import annotations

import logging
from typing import Callable, Iterator

from studio.speech.core.handler_types import LLMOut, TTSIn
from studio.speech.core.messages import (
    EndOfResponse,
    LLMResponseChunk,
    TokenUsage,
    AssistantTextPart,
)
from studio.speech.core.events import PipelineEvent
from studio.speech.lm_output_processor import LMOutputProcessor

logger = logging.getLogger(__name__)

Verdict = str  # "answer" | "discard"


class SilentTurnProcessor(LMOutputProcessor):
    """LMOutputProcessor, der Silent-Turns puffert und verwirft statt zu streamen."""

    def setup(
        self,
        *args,
        is_silent: Callable[[str], bool] | None = None,
        decide: Callable[[str, str], Verdict] | None = None,
        **kwargs,
    ) -> None:
        super().setup(*args, **kwargs)
        self._is_silent = is_silent or (lambda _key: False)
        self._decide = decide or (lambda _key, _text: "answer")
        self._buffer: dict[str, list[LLMResponseChunk]] = {}

    def process(self, lm_output: LLMOut) -> Iterator[TTSIn | PipelineEvent]:
        if isinstance(lm_output, TokenUsage):
            # Kosten sind real und bleiben sichtbar (nur intern, nie im Client).
            yield from super().process(lm_output)
            return

        key = getattr(lm_output, "response_key", None)
        if not key or not self._is_silent(key):
            yield from super().process(lm_output)
            return

        if isinstance(lm_output, LLMResponseChunk):
            self._buffer.setdefault(key, []).append(lm_output)
            return

        if isinstance(lm_output, EndOfResponse):
            chunks = self._buffer.pop(key, [])
            text = self._text_of(chunks)
            verdict = "discard"
            if lm_output.error:
                # Fehlgeschlagene Generierung (z. B. Provider-Fehler) darf im
                # Silent-Turn nie hörbar werden — sonst spräche Emma eine
                # Fehlermeldung vor.
                logger.info("Silent-Turn: Generierung fehlgeschlagen (%s) — still verworfen", lm_output.error)
            else:
                try:
                    verdict = self._decide(key, text)
                except Exception as exc:                      # noqa: BLE001
                    logger.warning("Silent-Turn-Entscheidung fehlgeschlagen (%s) — verworfen", exc)
            if verdict == "answer":
                for chunk in chunks:
                    yield from super().process(chunk)
                yield from super().process(lm_output)
                logger.info("Silent-Turn: Antwort freigegeben (%d Zeichen)", len(text))
                return
            # Unsichtbar verwerfen: Buchhaltung abschließen, sonst nichts.
            self._notify_generation_done(lm_output, key, succeeded=False)
            logger.info("Silent-Turn verworfen: %r", text[:80])
            yield EndOfResponse(
                turn_id=lm_output.turn_id,
                turn_revision=lm_output.turn_revision,
                cancel_generation=lm_output.cancel_generation,
                response_key=key,
                cleanup_only=True,
            )
            self._reset_response()
            return

        yield from super().process(lm_output)

    @staticmethod
    def _text_of(chunks: list[LLMResponseChunk]) -> str:
        return "".join(
            part.text for chunk in chunks for part in chunk.parts
            if isinstance(part, AssistantTextPart)
        ).strip()

    def on_session_end(self) -> None:
        self._buffer.clear()
        super().on_session_end()
