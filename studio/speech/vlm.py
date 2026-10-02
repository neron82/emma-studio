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
* The model needs ~15.5 GB and is kept resident; every call is serialized
  (one model, one GPU) on a single owner thread — see below.
* **The cold start must not overlap whisper.** On device 1 the load peaks at
  ~23 GB (bge-m3 + OmniVoice + whisper + VLM on a 24 GB card). Whisper inference
  running in that window takes the whole process down (measured 22.09.2026:
  CUDA-OOM inside CTranslate2's graph capture → ``std::terminate`` → SIGABRT,
  call dead). Hence every load/call here runs under
  ``core/device_lock.DEVICE_LOCK``, and video calls call :func:`warmup` at call
  start instead of loading in the middle of a turn.
* **One owner thread.** ``torch.cuda.set_device`` is thread-local and kestrel
  allocates on the *current* device while importing, so model load and inference
  must not sit on different threads (the same trap ``stt.py`` documents).
"""
from __future__ import annotations

import base64
import io
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from PIL import Image

from .. import config
from .core.device_lock import DEVICE_LOCK

logger = logging.getLogger(__name__)

_model: Any = None
_load_lock = threading.Lock()

# Alle VLM-Arbeit läuft auf EINEM Thread — wie schon bei Whisper (siehe
# ``stt.py``): ``torch.cuda.set_device`` ist thread-lokal und Kestrel alloziert
# bereits beim Import auf dem *aktuellen* Gerät. Laden und Inferenz auf
# verschiedenen Threads wäre genau die Falle, die dort schon dokumentiert ist.
# Der Executor serialisiert die Aufrufe zugleich (ein Modell, ein Gerät).
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm")
_device_set = False

_warm_lock = threading.Lock()
_warm_pending = False

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


def _ensure_device() -> None:
    """Gerät für Kestrel/torch setzen — nur auf dem VLM-Thread aufrufen."""
    global _device_set
    if not _device_set:
        import torch

        # WICHTIG: kestrel/Photon alloziert schon beim Import auf dem
        # *aktuellen* torch-Gerät. Ohne das hier landet es auf Gerät 0, wo das
        # 27B-Chatmodell liegt (~400 MB frei) → CUDA-OOM.
        # Das Studio kann CUDA_VISIBLE_DEVICES nicht global setzen, weil
        # Whisper dort explizit device_index=1 nutzt.
        torch.cuda.set_device(torch.device(config.VLM_DEVICE))
        _device_set = True


def _load_on_owner() -> Any:
    """Lädt das Modell (einmalig ~8-15 s, ~15,5 GB VRAM).

    Angestoßen entweder vorgewärmt beim Anrufstart (:func:`warmup`) oder beim
    ersten Frame. Nur auf dem VLM-Thread aufrufen (nicht per Executor — das
    liefe in den Deadlock, der Executor hat genau einen Worker).
    """
    global _model
    if _model is not None:
        return _model
    with _load_lock:
        if _model is None:
            t0 = time.time()
            logger.info("VLM: lade %s auf %s", config.VLM_MODEL, config.VLM_DEVICE)
            try:
                _ensure_device()
                import moondream as md

                # DEVICE_LOCK: der Kaltstart belegt in der Spitze ~23 GB; läuft
                # parallel eine Whisper-Inferenz, reißt CUDA-OOM den Prozess mit
                # (siehe core/device_lock.py).
                with DEVICE_LOCK:
                    _model = md.vl(local=True, model=config.VLM_MODEL, device=config.VLM_DEVICE)
            except Exception as exc:                       # noqa: BLE001
                raise VLMError(f"model load failed: {type(exc).__name__}: {exc}") from exc
            logger.info("VLM bereit in %.1fs", time.time() - t0)
    return _model


def get_model():
    """Modell laden (auf dem VLM-Thread) und zurückgeben. Blockierend."""
    return _executor.submit(_load_on_owner).result()


_get_model = get_model  # historischer Name (Diagnose/Status)


def warmup() -> None:
    """Modell im Hintergrund vorladen — idempotent, blockiert den Aufrufer nicht.

    Der Kaltstart dauert ~8-15 s und hält ``DEVICE_LOCK``, also jede
    Whisper-Inferenz. Er gehört deshalb *vor* den ersten Turn: der Client kennt
    die Modalität schon beim Anrufstart und meldet sie an
    ``POST /api/calls/start``, bevor Mikrofon und Kamera hochkommen. Ohne das
    fällt das Laden mitten in einen Turn — dort hat es am 22.09.2026 den Anruf
    abgerissen.
    """
    global _warm_pending
    if not config.VLM_ENABLED or _model is not None:
        return
    with _warm_lock:
        if _warm_pending:
            return
        _warm_pending = True
    _executor.submit(_warm_on_owner)


def _warm_on_owner() -> None:
    global _warm_pending
    t0 = time.time()
    try:
        _load_on_owner()
        logger.info("VLM vorgewärmt in %.1fs", time.time() - t0)
    except Exception as exc:                               # noqa: BLE001
        logger.warning(
            "VLM: Vorladen fehlgeschlagen (%s: %s) — Video läuft ohne Bildkontext",
            type(exc).__name__, exc,
        )
    finally:
        with _warm_lock:
            _warm_pending = False


def describe(jpeg: bytes) -> str:
    """Ein Satz über den aktuellen Frame (Bild kommt als JPEG-Bytes)."""
    if not config.VLM_ENABLED:
        return ""
    img = Image.open(io.BytesIO(jpeg))
    img.load()
    img = img.convert("RGB")
    if max(img.size) > config.VLM_MAX_EDGE:
        img.thumbnail((config.VLM_MAX_EDGE, config.VLM_MAX_EDGE), Image.LANCZOS)
    try:
        out = _executor.submit(_describe_on_owner, img).result()
    except VLMError:
        raise
    except Exception as exc:                               # noqa: BLE001
        raise VLMError(f"describe failed: {type(exc).__name__}: {exc}") from exc
    return str((out or {}).get("answer", "")).strip()


def _describe_on_owner(img: Any) -> Any:
    # DEVICE_LOCK nur für die GPU-Arbeit (eine Inferenz, ~150-250 ms) — nicht
    # fürs Bild-Vorbereiten, das läuft auf dem Aufrufer-Thread.
    with DEVICE_LOCK:
        return _load_on_owner().query(img, config.VLM_DESCRIBE_PROMPT)


def _chat(prompt: str) -> str:
    try:
        out = _executor.submit(_chat_on_owner, prompt).result()
    except VLMError:
        raise
    except Exception as exc:                               # noqa: BLE001
        raise VLMError(f"chat failed: {type(exc).__name__}: {exc}") from exc
    msg = out.get("message") if isinstance(out, dict) else None
    text = msg.get("content") if isinstance(msg, dict) else msg
    if isinstance(text, list):                             # content parts
        text = " ".join(str(p.get("text", "")) for p in text if isinstance(p, dict))
    return str(text or "").strip()


def _chat_on_owner(prompt: str) -> Any:
    with DEVICE_LOCK:
        # reasoning aus: die Denk-Kette kostet nur Latenz für eine Aufgabe,
        # die eine Zeile Ausgabe hat (Messung: siehe config.VLM_REASONING).
        return _load_on_owner().chat([{"role": "user", "content": prompt}],
                                     reasoning=config.VLM_REASONING)


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
