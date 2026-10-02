"""STT via faster-whisper. Model loads lazily and stays resident.

Mirrors the proven chatbot configuration: large-v3-turbo on the 3060 Ti,
language pinned to German (auto-detect on short utterances hallucinates
English fragments like "you"), beam_size 5, no condition-on-previous-text.
The resident model is shared by the batch endpoints (/api/stt, video jobs),
the realtime STT handler and the viseme timeline builder.

Threading model — important: faster-whisper/CTranslate2 models are
thread-affine. Running inference on a thread other than the one that loaded
the model aborts the whole process with a bogus C++ "CUDA failed with error
out of memory" (ct2 4.8.1 / fw 1.2.1, CUDA float16; measured: load+infer on
one thread is fine, load on the main thread + infer on a worker thread dies,
and sequential inference never OOMs). All model work therefore runs on ONE
dedicated owner thread, which also serializes the concurrent callers that
exist in practice (realtime STT handler, viseme TTS handler, /api/stt).
Sequential cost is ~0.15 s per short chunk, so serialization is cheap.

Device sharing — important: the VLM (``vlm.py``, Moondream on the same device)
allocates ~23 GB in its cold start. Whisper inference must not run in that
window, otherwise the CUDA-graph capture inside CTranslate2 fails and kills
the process (see ``core/device_lock.py``). Hence every load/inference here
runs under ``DEVICE_LOCK``.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .. import config
from .core.device_lock import DEVICE_LOCK

STT_LANGUAGE = config.STT_LANGUAGE  # pinned — never auto-detect on short mic chunks

_model: Any = None
# The one thread allowed to touch the CT2 model (see module docstring).
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")


class STTError(RuntimeError):
    pass


def _load_model():
    """Load the resident model. Must run on the owner thread."""
    global _model
    if _model is None:
        if config.STT_DEVICE == "cuda":
            _preload_cuda_libs()
        from faster_whisper import WhisperModel
        with DEVICE_LOCK:                       # nicht parallel zum VLM-Kaltstart
            _model = WhisperModel(
                config.STT_MODEL,
                device=config.STT_DEVICE,
                device_index=config.STT_DEVICE_INDEX,
                compute_type="float16" if config.STT_DEVICE == "cuda" else "int8",
            )
    return _model


def get_model():
    """Ensure the resident model is loaded (on the owner thread) and return it.

    The returned object may only be *used* on the owner thread — other threads
    must call the ``transcribe*`` functions or use ``bound_model()``.
    """
    return _executor.submit(_load_model).result()


_get_model = get_model  # historical alias


class _BoundModel:
    """Handle whose ``transcribe()`` forwards to the owner thread.

    Keeps handler code (``self.model.transcribe(audio, **kwargs)``) in its
    familiar shape while the CT2 work stays on the owning thread.
    """

    __slots__ = ()

    def transcribe(self, audio: Any, **kwargs: Any):
        return transcribe_segments(audio, **kwargs)


def bound_model() -> _BoundModel:
    """Thread-safe stand-in for a WhisperModel (see ``_BoundModel``)."""
    return _BoundModel()


def _transcribe_on_owner(source: Any, kwargs: dict[str, Any]):
    # Materialize the segments: faster-whisper decodes lazily while the returned
    # generator is consumed, and that work must stay on the owner thread.
    # DEVICE_LOCK: ein paralleler VLM-Kaltstart auf demselben Gerät reißt den
    # Prozess mit — siehe core/device_lock.py.
    with DEVICE_LOCK:
        segments, info = _load_model().transcribe(source, **kwargs)
        return list(segments), info


def transcribe_segments(source: Any, **kwargs: Any) -> tuple[list, Any]:
    """Run one transcription on the owner thread; returns (segments, info)."""
    return _executor.submit(_transcribe_on_owner, source, kwargs).result()


def _preload_cuda_libs() -> None:
    """ct2 needs cublas .so.12 — the system may only ship .so.13, so load the pip copy."""
    import ctypes
    import glob
    import sys

    candidates: list[str] = []
    try:
        import nvidia.cublas.lib as _cublas_pkg  # pip nvidia-cublas-cu12
        candidates.append(str(Path(_cublas_pkg.__file__).parent / "libcublas.so.12"))
    except (ImportError, AttributeError, TypeError):
        pass
    candidates += glob.glob(
        str(Path(sys.prefix) / "lib" / "python*" / "site-packages" / "nvidia" / "cublas" / "lib" / "libcublas.so.12*")
    )
    for c in candidates[:1]:
        try:
            ctypes.CDLL(c, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


def _words_from_segments(segments) -> list[dict]:
    words: list[dict] = []
    for seg in segments:
        for w in getattr(seg, "words", None) or []:
            token = (w.word or "").strip()
            if token:
                words.append({"word": token, "start": w.start, "end": w.end})
    return words


def transcribe(audio_path: Path, *, language: str | None = STT_LANGUAGE) -> str:
    """Transcribe an audio file to text. Blocking — call from a worker thread."""
    path = Path(audio_path)
    if not path.is_file():
        raise STTError(f"audio file not found: {path}")
    segments, _info = transcribe_segments(
        str(path),
        language=language,                 # pinned 'de' — see module docstring
        beam_size=5,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    return " ".join(s.text.strip() for s in segments).strip()


def transcribe_words_array(audio: Any, *, language: str | None = STT_LANGUAGE) -> list[dict]:
    """Word timestamps for a 16 kHz float32 mono array (viseme timeline)."""
    segments, _info = transcribe_segments(
        np.asarray(audio, dtype=np.float32),
        language=language,
        beam_size=1,
        condition_on_previous_text=False,
        vad_filter=False,
        word_timestamps=True,
    )
    return _words_from_segments(segments)
