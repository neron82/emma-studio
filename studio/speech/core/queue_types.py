"""Typed aliases for items flowing through pipeline queues.

This module centralizes queue payload unions so the rest of the codebase can
reference a small set of stable types instead of repeating large unions.
"""

from __future__ import annotations

# ruff: noqa: I001

from typing import TypeAlias

import numpy as np

from studio.speech.core.control import PipelineControlMessage
from studio.speech.core.events import PipelineEvent
from studio.speech.core.handler_types import LLMIn, LLMOut, STTOut, TTSIn, VADIn, VADOut
from studio.speech.core.messages import AudioOutput

# Use plain ``bytes`` for sentinel values on queues (``PIPELINE_END``, ``AUDIO_RESPONSE_DONE``).
# ``Queue`` is invariant; ``Literal[b"END"]`` is not accepted where ``bytes`` is required.
PipelineInternalItem: TypeAlias = PipelineControlMessage | bytes

# Audio chunks decoded from Realtime transports into VAD.
AudioInItem: TypeAlias = VADIn | PipelineControlMessage

# Audio segments flowing from VAD to STT.
VADOutItem: TypeAlias = VADOut | PipelineInternalItem

# STT output (partial + final) flowing to TranscriptionNotifier.
STTOutItem: TypeAlias = STTOut | PipelineInternalItem

# Text prompts flowing into LLM.
TextPromptItem: TypeAlias = LLMIn | PipelineInternalItem

# LLM outputs flowing into LMOutputProcessor.
LMOutItem: TypeAlias = LLMOut | PipelineInternalItem

# Inputs flowing into TTS.
TTSInItem: TypeAlias = TTSIn | PipelineEvent | PipelineInternalItem

# Ordered response events and audio flowing to the client.
AudioOutItem: TypeAlias = bytes | np.ndarray | AudioOutput | PipelineEvent | PipelineControlMessage

# VAD and transcription events that do not wait for TTS.
TextEventItem: TypeAlias = PipelineEvent | PipelineInternalItem
