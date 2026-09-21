"""Sentence-chunked TTS: synthesize a reply sentence by sentence.

The frontend plays chunks back-to-back; first audio arrives after the first
sentence instead of after the whole reply. Chunks are produced sequentially
(OmniVoice is single-voice on the 3060 Ti) but streamed to the client as
they finish via the chunk endpoints.
"""
from __future__ import annotations

import re
from pathlib import Path

from . import tts

# Split on sentence enders, keep them attached; merge very short fragments
# (e.g. "z.B." artifacts) into the following sentence.
_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")
_MIN_CHARS = 25


def split_sentences(text: str, *, min_chars: int = _MIN_CHARS) -> list[str]:
    raw = _SENTENCE_RE.split(text.strip())
    chunks: list[str] = []
    for part in raw:
        part = part.strip()
        if not part:
            continue
        if chunks and len(chunks[-1]) < min_chars:
            chunks[-1] += " " + part
        elif len(part) < min_chars and chunks:
            chunks[-1] += " " + part
        else:
            chunks.append(part)
    return chunks


async def synthesize_chunks(text: str, *, voice: str, job_dir: Path,
                            language: str | None = None,
                            on_first_chunk=None) -> list[Path]:
    """Synthesize each sentence to chunk_NN.wav. Returns ordered paths.
    Raises TTSError on first failure (caller decides fallback).
    on_first_chunk: async callback fired right after chunk 0 is written."""
    chunks = split_sentences(text)
    if not chunks:
        chunks = [text]
    paths: list[Path] = []
    job_dir.mkdir(parents=True, exist_ok=True)
    for i, sentence in enumerate(chunks):
        p = job_dir / f"chunk_{i:02d}.wav"
        await tts.synthesize(sentence, voice=voice, out_path=p, language=language)
        paths.append(p)
        if i == 0 and on_first_chunk:
            try:
                await on_first_chunk()
            except Exception:  # noqa: BLE001 — probe must never break synthesis
                pass
    return paths
