"""Emma-specific realtime protocol events (extensions to the OpenAI dialect).

Kept next to the ported protocol code on purpose: these are the only places
where Emma's own wire format diverges, so they are easy to find and audit.

Server → client: ``emma.visemes`` (defined in studio.speech.engine, where the
lip-sync timeline is produced).
Client → server: ``emma.video_frame`` — one downscaled JPEG frame from the
user's camera, sent about once per second while the call is running. The
realtime engine turns frames into short descriptions and a rolling summary that
feeds the dialog prompt.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class EmmaVideoFrameEvent(BaseModel):
    """One camera frame from the client (base64 JPEG, optionally data:-prefixed).

    Freq: ~1/s while the microphone is on. Size: ~15-25 KB per frame after the
    browser downsizes to the configured edge, so ~0.2 Mbit/s — negligible next
    to the audio stream.
    """

    type: Literal["emma.video_frame"] = "emma.video_frame"
    event_id: str
    frame: str = Field(description="base64 JPEG data (data: prefix allowed)")
    at: Optional[float] = Field(default=None, description="client timestamp (informational)")


class EmmaStatusEvent(BaseModel):
    """Pipeline status line for the client's debug overlay.

    Emitted at the stages that are otherwise invisible from the browser: VLM
    analysis (load/describe/summary), prompt assembly (RAG + visual context),
    first LM text, TTS sentence synthesis, response completion. Best effort —
    status events are never required for the dialog to work.
    """

    type: Literal["emma.status"] = "emma.status"
    event_id: str
    stage: str = Field(description="VLM | STT | PROMPT | LM | TTS | RESP")
    detail: str = ""
    ms: Optional[float] = Field(default=None, description="duration of the reported step")
    at: Optional[float] = None
