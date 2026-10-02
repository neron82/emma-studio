"""OmniVoice TTS client — OpenAI-compatible /v1/audio/speech with profile voices.

Voice resolution: pass the OmniVoice profile id (e.g. "emma_alicia") as `voice`.
The server resolves profiles internally; no clone-upload dance needed here.
"""
from __future__ import annotations

from pathlib import Path

import httpx

from .. import config


class TTSError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


async def synthesize(text: str, *, voice: str, out_path: Path,
                     language: str | None = None) -> Path:
    """Synthesize `text` to a wav file. Blocking HTTP inside async wrapper."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "omnivoice",
        "input": text,
        "voice": voice,
        "response_format": "wav",
    }
    if language:
        payload["language"] = language
    url = f"{config.OMNIVOICE_URL.rstrip('/')}/v1/audio/speech"
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(url, json=payload)
    if resp.status_code != 200:
        raise TTSError(f"OmniVoice HTTP {resp.status_code}: {resp.text[:300]}",
                       status_code=resp.status_code)
    data = resp.content
    if len(data) < 1000:  # a real wav is never this small; guards error-JSON-as-wav
        raise TTSError(f"OmniVoice returned suspiciously small audio ({len(data)} bytes)")
    out_path.write_bytes(data)
    return out_path


def synthesize_sync(text: str, *, voice: str, out_path: Path,
                    language: str | None = None) -> Path:
    """Blocking variant for worker threads."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": "omnivoice", "input": text, "voice": voice, "response_format": "wav"}
    if language:
        payload["language"] = language
    url = f"{config.OMNIVOICE_URL.rstrip('/')}/v1/audio/speech"
    with httpx.Client(timeout=300) as client:
        resp = client.post(url, json=payload)
    if resp.status_code != 200:
        raise TTSError(f"OmniVoice HTTP {resp.status_code}: {resp.text[:300]}",
                       status_code=resp.status_code)
    data = resp.content
    if len(data) < 1000:
        raise TTSError(f"OmniVoice returned suspiciously small audio ({len(data)} bytes)")
    out_path.write_bytes(data)
    return out_path


async def health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{config.OMNIVOICE_URL.rstrip('/')}/health")
        return resp.status_code == 200
    except httpx.HTTPError:
        return False
