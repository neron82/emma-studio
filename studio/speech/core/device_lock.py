"""Ein Schloss für die schwere CUDA-Arbeit im Studio-Prozess.

STT (Whisper/CTranslate2) und VLM (Moondream über Kestrel/torch) teilen sich
Gerät 1. Der VLM-Kaltstart belegt dort in der Spitze ~23 GB und versucht dabei
eine 3-GiB-Allokation, für die auf der Karte kein Platz mehr ist (gemessen:
22,8 GB belegt, 1,8 GB frei). Läuft in diesem Moment eine Whisper-Inferenz,
kippt CTranslate2s CUDA-Graph-Capture und die unbehandelte C++-Exception reißt
den *ganzen* Prozess mit — der Anruf ist tot:

    terminate called after throwing an instance of 'std::runtime_error'
      what():  CUDA failed with error operation not permitted when stream is capturing
    → SIGABRT

Belegung des Geräts im Anruf (24 GB): bge-m3 ~0,9 GB + OmniVoice ~2,3 GB +
Whisper ~2,3 GB + VLM ~15,5 GB (VLM-Kaltstart und STT sequenziell: stabil,
parallel: Absturz — am 22.09.2026 reproduziert).

Deshalb serialisiert dieses Schloss ausschließlich die GPU-Arbeit (Modell laden,
Inferenz). Es wird nie über Wartezeiten oder Netz-I/O gehalten, und die
Verschachtelung ist einbahnig: der Laden-Pfad nimmt es innerhalb des Aufrufs,
der Aufruf nie innerhalb des Ladens.
"""
from __future__ import annotations

import threading

DEVICE_LOCK = threading.RLock()

__all__ = ["DEVICE_LOCK"]
