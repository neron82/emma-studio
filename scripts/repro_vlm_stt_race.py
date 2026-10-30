"""Manueller Regressionstest: STT und VLM dürfen sich das Gerät nicht teilen.

Reproduziert den Absturz vom 22.09.2026 (Videoanruf, 2. Turn):

    terminate called after throwing an instance of 'std::runtime_error'
      what():  CUDA failed with error operation not permitted when stream is capturing
    → SIGABRT, der ganze Studio-Prozess stirbt

Ursache: Der VLM-Kaltstart (~15,5 GB, Spitze ~23 GB) lief auf Gerät 1 parallel
zu einer Whisper-Inferenz. In diesem Fenster scheitert CTranslate2s
CUDA-Graph-Capture und die C++-Exception reißt den Prozess mit.

Erwartung:
  * mit ``core/device_lock.DEVICE_LOCK`` (Ist-Zustand): überlebt, die STT-Läufe
    stehen während des Ladens still (serialisiert).
  * ohne das Schloss (Fix zurückbauen): SIGABRT in dieser Datei.

Das ist ein *Hardware*-Test: er braucht die echten Modelle (~15,5 GB Download/
VRAM auf Gerät 1) und darf nie im Studio-Prozess laufen (er beendet sich im
Fehlerfall selbst). Aufruf:

    cd /path/to/emma-studio
    python scripts/repro_vlm_stt_race.py
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio.speech import stt, vlm  # noqa: E402


def gpu_used_mib(index: int = 1) -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
         "-i", str(index)],
        capture_output=True, text=True,
    ).stdout.strip()
    return int(out) if out.isdigit() else -1


def main() -> int:
    stt.get_model()
    audio = np.zeros(16000 * 3, dtype=np.float32)
    stt.transcribe_segments(audio, language="de", beam_size=5)
    print(f"whisper bereit — GPU1: {gpu_used_mib()} MiB", flush=True)

    stop = threading.Event()
    runs = 0

    def hammer_stt() -> None:
        nonlocal runs
        while not stop.is_set():
            stt.transcribe_segments(audio, language="de", beam_size=5)
            runs += 1

    worker = threading.Thread(target=hammer_stt, daemon=True, name="stt-hammer")
    worker.start()
    time.sleep(0.5)

    print("VLM-Kaltstart parallel zur STT …", flush=True)
    t0 = time.time()
    vlm.get_model()
    print(f"VLM geladen in {time.time() - t0:.1f}s — GPU1: {gpu_used_mib()} MiB", flush=True)

    stop.set()
    worker.join(timeout=5)
    print(f"STT-Läufe: {runs}", flush=True)
    print("ÜBERLEBT — STT und VLM sind serialisiert (kein Absturz)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
