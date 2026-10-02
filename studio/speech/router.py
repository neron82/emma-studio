"""Speech-mode router: decide per job how speech is produced.

Two modes:
  h3_native — MiniMax H3 speaks from the persona's voice sample (fast path,
              one GPU pass; weaker language fidelity)
  tts_pre   — OmniVoice synthesizes the exact script first (correct language,
              controllable pacing), H3 then animates over the full audio track

Policy: tts_pre is the default for scripted video messages (quality first);
h3_native is available as an explicit override and used when OmniVoice is
unreachable. Call turns never reach this router (audio-only).
"""
from __future__ import annotations

from .tts import health as tts_health


async def choose_speech_mode(*, requested: str | None = None) -> str:
    """Return 'tts_pre' or 'h3_native'.

    requested: explicit user choice wins if valid.
    Otherwise: tts_pre if OmniVoice is up, else h3_native.
    """
    if requested in ("tts_pre", "h3_native"):
        return requested
    return "tts_pre" if await tts_health() else "h3_native"
