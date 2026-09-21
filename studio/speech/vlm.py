"""Visual context via Moondream 3.1 (local Photon inference).

Two capabilities, both driven from the realtime pipeline:

``describe(jpeg)``    one short English sentence about the current camera frame
                      (~150-250 ms, single-frame ``query``)
``summarize(obs)``    compresses many descriptions into one line (~110 ms,
                      text-only ``chat`` — no image, see note below)

Notes that cost real measurement to learn (keep them):

* **English prompts only.** German questions return unusable answers
  ("Was macht die Person gerade?" → "Ja, sie ist gerade.") and prepend a long
  reasoning chain. English ones are precise and short.
* **Summarize without the image.** Attaching the current frame pulls the
  summary towards *that* frame and loses the history; the text-only chat
  faithfully compresses the list.
* **Device must be explicit.** Photon defaults to ``cuda`` = device 0, which
  carries the 27B chat model; the studio cannot set CUDA_VISIBLE_DEVICES
  globally because whisper pins ``device_index=1``. Hence ``device="cuda:1"``.
* The model needs ~15.7 GB, so it is loaded lazily on the first frame and kept
  resident; every call is serialized (one model, one GPU).
"""
from __future__ import annotations

import base64
import io
import logging
import threading
import time
from typing import Any

from PIL import Image

from .. import config

logger = logging.getLogger(__name__)

_model: Any = None
_load_lock = threading.Lock()
_call_lock = threading.Lock()

# Wie viele Beobachtungen eine Verlaufsbeschreibung aufnehmen darf (Kontextbudget)
MAX_SUMMARY_INPUT = 40

# Wichtig: *Verlauf* statt Zustand. "Sie schaut lächelnd in die Kamera und hält
# einen Stift hoch" wäre ein Zustandsbild — gefragt ist, was sich nacheinander
# verändert hat ("erst neutral, dann Lächeln, dann hebt sie einen Stift").
TEMPORAL_PROMPT = (
    "Below are descriptions of the same scene, one per second, in chronological order.\n"
    "{items}\n\n"
    "Write 2-3 short English sentences describing what CHANGED, in the order it happened: "
    "how the person's appearance, expression or actions developed from the first to the last "
    "second. Merge moments that stayed the same into one clause — do NOT enumerate seconds or "
    "repeat the same state. If nothing changed at all, say that the scene stayed the same. "
    "No preamble, no bullet points."
)


class VLMError(RuntimeError):
    pass


def status() -> dict[str, Any]:
    """Für Debug/Statusanzeige — ohne das Modell zu laden."""
    return {
        "enabled": config.VLM_ENABLED,
        "loaded": _model is not None,
        "model": config.VLM_MODEL,
        "device": config.VLM_DEVICE,
    }


def _get_model():
    """Lädt das Modell beim ersten Frame (dauert ~10-15 s, ~15.7 GB VRAM)."""
    global _model
    if _model is not None:
        return _model
    with _load_lock:
        if _model is None:
            t0 = time.time()
            logger.info("VLM: lade %s auf %s", config.VLM_MODEL, config.VLM_DEVICE)
            try:
                # WICHTIG: kestrel/Photon alloziert schon beim Import auf dem
                # *aktuellen* torch-Gerät. Ohne das hier landet es auf Gerät 0,
                # wo das 27B-Chatmodell liegt (~400 MB frei) → CUDA-OOM.
                # Das Studio kann CUDA_VISIBLE_DEVICES nicht global setzen, weil
                # Whisper dort explizit device_index=1 nutzt.
                import torch

                torch.cuda.set_device(torch.device(config.VLM_DEVICE))
                import moondream as md

                _model = md.vl(local=True, model=config.VLM_MODEL, device=config.VLM_DEVICE)
            except Exception as exc:                       # noqa: BLE001
                raise VLMError(f"model load failed: {type(exc).__name__}: {exc}") from exc
            logger.info("VLM bereit in %.1fs", time.time() - t0)
    return _model


def describe(jpeg: bytes) -> str:
    """Ein Satz über den aktuellen Frame (Bild kommt als JPEG-Bytes)."""
    if not config.VLM_ENABLED:
        return ""
    img = Image.open(io.BytesIO(jpeg))
    img.load()
    img = img.convert("RGB")
    if max(img.size) > config.VLM_MAX_EDGE:
        img.thumbnail((config.VLM_MAX_EDGE, config.VLM_MAX_EDGE), Image.LANCZOS)
    with _call_lock:
        try:
            out = _get_model().query(img, config.VLM_DESCRIBE_PROMPT)
        except VLMError:
            raise
        except Exception as exc:                           # noqa: BLE001
            raise VLMError(f"describe failed: {type(exc).__name__}: {exc}") from exc
    return str((out or {}).get("answer", "")).strip()


def _chat(prompt: str) -> str:
    with _call_lock:
        try:
            # reasoning aus: die Denk-Kette kostet nur Latenz für eine Aufgabe,
            # die eine Zeile Ausgabe hat (Messung: siehe config.VLM_REASONING).
            out = _get_model().chat([{"role": "user", "content": prompt}],
                                    reasoning=config.VLM_REASONING)
        except VLMError:
            raise
        except Exception as exc:                           # noqa: BLE001
            raise VLMError(f"chat failed: {type(exc).__name__}: {exc}") from exc
    msg = out.get("message") if isinstance(out, dict) else None
    text = msg.get("content") if isinstance(msg, dict) else msg
    if isinstance(text, list):                             # content parts
        text = " ".join(str(p.get("text", "")) for p in text if isinstance(p, dict))
    return str(text or "").strip()


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items[-MAX_SUMMARY_INPUT:])


def summarize_temporal(observations: list[str]) -> str:
    """Verlaufsbeschreibung eines Fensters (ohne Bild — siehe Modul-Docstring)."""
    obs = [o.strip() for o in observations if o and o.strip()]
    if not obs:
        return ""
    if len(obs) == 1:
        return obs[0]                    # ein Frame hat keinen Verlauf
    if not config.VLM_ENABLED:
        return " ".join(obs[-2:])
    return _chat(TEMPORAL_PROMPT.format(items=_bullets(obs)))


def decode_frame(frame_b64: str) -> bytes:
    """Client-Frame (base64, optional mit data:-Präfix) → JPEG-Bytes."""
    data = frame_b64.split(",", 1)[1] if frame_b64.startswith("data:") else frame_b64
    try:
        return base64.b64decode(data, validate=False)
    except Exception as exc:                               # noqa: BLE001
        raise VLMError(f"frame decode failed: {exc}") from exc
