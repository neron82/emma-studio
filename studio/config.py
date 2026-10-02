"""Emma Studio configuration.

One config system for everything: models, voices, workflows, limits.
Env vars override the SQLite-backed settings where secrets are involved.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "studio.db"
WORKFLOW_DIR = BASE_DIR / "workflows"

load_dotenv(BASE_DIR / ".env")

# --- LLM (dialog agent) — OpenAI-compatible, configurable -------------------
LLM_BASE_URL = os.getenv("STUDIO_LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("STUDIO_LLM_MODEL", "stealth/ox-alpha")
LLM_API_KEY = os.getenv("STUDIO_LLM_API_KEY", "")

# --- Memory (per-persona call summaries in memory.md) -----------------------
# Hard word limit for a persona's long-term memory; the oldest entry is dropped
# before a new one is appended once the file exceeds this.
MEMORY_MAX_WORDS = int(os.getenv("STUDIO_MEMORY_MAX_WORDS", "10000"))

# --- Speech -----------------------------------------------------------------
OMNIVOICE_URL = os.getenv("STUDIO_OMNIVOICE_URL", "http://127.0.0.1:8880")
STT_MODEL = os.getenv("STUDIO_STT_MODEL", "large-v3-turbo")  # proven chatbot config
STT_DEVICE = os.getenv("STUDIO_STT_DEVICE", "cuda")
STT_DEVICE_INDEX = int(os.getenv("STUDIO_STT_DEVICE_INDEX", "1"))  # 1 = 3060 Ti

# --- Render (ComfyUI / MiniMax H3) ------------------------------------------
COMFY_URL = os.getenv("STUDIO_COMFY_URL", "http://127.0.0.1:8188")
DEFAULT_WORKFLOW = os.getenv("STUDIO_DEFAULT_WORKFLOW", "turbo")
MAX_VIDEO_SECONDS = 15.0
FRAME_RATE = 24

# --- Search fallback ----------------------------------------------------------
SEARXNG_URL = os.getenv("STUDIO_SEARXNG_URL", "http://127.0.0.1:8081")
SEARCH_ENABLED = os.getenv("STUDIO_SEARCH_ENABLED", "1") == "1"

# --- Embeddings (RAG fact cards) ---------------------------------------------
EMBED_URL = os.getenv("STUDIO_EMBED_URL", "http://127.0.0.1:7777")  # bge-m3
EMBED_MODEL = os.getenv("STUDIO_EMBED_MODEL", "bge-m3")

# --- Limits -------------------------------------------------------------------
MAX_FACTS_PER_QUERY = 8
SEARCH_TOP_K = 5

# --- Realtime speech2speech (OpenAI-Realtime WS pipeline) --------------------
# Mounted at /v1/realtime on the studio app; pool of threaded pipeline units.
REALTIME_ENABLED = os.getenv("STUDIO_REALTIME", "1") == "1"
REALTIME_POOL_SIZE = int(os.getenv("STUDIO_REALTIME_POOL_SIZE", "1"))
REALTIME_CHAT_SIZE = int(os.getenv("STUDIO_REALTIME_CHAT_SIZE", "12"))
REALTIME_LM_TIMEOUT_S = float(os.getenv("STUDIO_REALTIME_LM_TIMEOUT", "60"))

# Schonfrist für "aktive" Call-Sessions ohne lebende Realtime-Verbindung: der
# Client startet die Session, bevor er die WS aufbaut, und ein Neustart/Absturz
# des Prozesses lässt die Session sonst für immer aktiv (Persona gesperrt).
STALE_CALL_GRACE_S = float(os.getenv("STUDIO_STALE_CALL_GRACE", "90"))

# VAD (silero via torch.hub cache)
VAD_THRESH = float(os.getenv("STUDIO_VAD_THRESH", "0.6"))
VAD_SAMPLE_RATE = int(os.getenv("STUDIO_VAD_SAMPLE_RATE", "16000"))
VAD_MIN_SILENCE_MS = int(os.getenv("STUDIO_VAD_MIN_SILENCE_MS", "64"))
VAD_MIN_SPEECH_MS = int(os.getenv("STUDIO_VAD_MIN_SPEECH_MS", "384"))
VAD_CONTINUATION_MS = int(os.getenv("STUDIO_VAD_CONTINUATION_MS", "192"))
VAD_SPEECH_PAD_MS = int(os.getenv("STUDIO_VAD_SPEECH_PAD_MS", "30"))
VAD_SPECULATIVE_REOPEN_MS = int(os.getenv("STUDIO_VAD_SPECULATIVE_REOPEN_MS", "800"))
VAD_UNANSWERED_REOPEN_MS = int(os.getenv("STUDIO_VAD_UNANSWERED_REOPEN_MS", "7000"))

# smart-turn semantic end-of-turn detection (ONNX on CPU)
SMART_TURN = os.getenv("STUDIO_SMART_TURN", "1") == "1"
SMART_TURN_MODEL_PATH = os.getenv("STUDIO_SMART_TURN_MODEL_PATH") or None
SMART_TURN_THRESHOLD = float(os.getenv("STUDIO_SMART_TURN_THRESHOLD", "0.5"))
SMART_TURN_MAX_WAIT_MS = int(os.getenv("STUDIO_SMART_TURN_MAX_WAIT_MS", "2000"))
SMART_TURN_INCOMPLETE_DELAY_MS = int(os.getenv("STUDIO_SMART_TURN_INCOMPLETE_DELAY_MS", "600"))

# Live partial transcription events (input_audio_transcription.delta)
LIVE_TRANSCRIPTION = os.getenv("STUDIO_LIVE_TRANSCRIPTION", "1") == "1"
LIVE_TRANSCRIPTION_INTERVAL_S = float(os.getenv("STUDIO_LIVE_TRANSCRIPTION_INTERVAL", "0.5"))

# Streaming TTS (OmniVoice OpenAI-compatible server)
TTS_MODEL = os.getenv("STUDIO_TTS_MODEL", "omnivoice")
TTS_VOICE_DEFAULT = os.getenv("STUDIO_TTS_VOICE", "emma_alicia")
TTS_STREAM = os.getenv("STUDIO_TTS_STREAM", "1") == "1"
TTS_BLOCKSIZE = int(os.getenv("STUDIO_TTS_BLOCKSIZE", "512"))
TTS_TIMEOUT_S = float(os.getenv("STUDIO_TTS_TIMEOUT", "300"))

# Streaming STT (shares the resident whisper singleton with /api/stt)
STT_LANGUAGE = os.getenv("STUDIO_STT_LANGUAGE", "de")
STT_BEAM_SIZE = int(os.getenv("STUDIO_STT_BEAM", "5"))

# Custom lip-sync timeline events (emma.visemes): whisper stamps on TTS audio
VISEMES = os.getenv("STUDIO_VISEMES", "1") == "1"

# --- VLM (visual context: what the user's camera shows) ----------------------
# Moondream 3.1 (9B-A2B) via Photon, resident on GPU 1 (~15.7 GB). The camera
# frame arrives over the realtime WebSocket; an analyzer thread turns frames
# into short descriptions (1/s) and keeps a rolling summary that gets injected
# into the dialog prompt at turn-complete.
VLM_ENABLED = os.getenv("STUDIO_VLM", "1") == "1"
VLM_MODEL = os.getenv("STUDIO_VLM_MODEL", "moondream3.1-9B-A2B")
# WICHTIG: Photon wählt ohne Angabe "cuda" = Gerät 0, wo das 27B-Chatmodell
# liegt (kein Platz). Das Studio kann CUDA_VISIBLE_DEVICES nicht global setzen,
# weil Whisper dort explizit device_index=1 benutzt.
VLM_DEVICE = os.getenv("STUDIO_VLM_DEVICE", "cuda:1")
VLM_MAX_EDGE = int(os.getenv("STUDIO_VLM_MAX_EDGE", "512"))
VLM_INTERVAL_S = float(os.getenv("STUDIO_VLM_INTERVAL", "1.0"))
VLM_FRAME_MAX_AGE_S = float(os.getenv("STUDIO_VLM_FRAME_MAX_AGE", "3.0"))
# Länge eines Verlaufs-Zyklus: alle 5 s wird aus den Sekunden-Beobachtungen eine
# *temporale* Beschreibung ("erst neutral, dann Lächeln, dann hält sie einen
# Stift hoch") erzeugt und gesammelt.
VLM_CYCLE_S = float(os.getenv("STUDIO_VLM_CYCLE", "5.0"))
VLM_HISTORY_MAX = int(os.getenv("STUDIO_VLM_HISTORY_MAX", "12"))   # gesammelte Zyklen
VLM_INJECT_MAX = int(os.getenv("STUDIO_VLM_INJECT_MAX", "3"))      # Zyklen im Prompt
# Reasoning des VLM für die Verlaufsbeschreibungen. Gemessen (21.09.): ~2×
# Latenz (0,14 s → 0,31 s) und im Blindvergleich 2 von 3 Fenstern nicht besser
# als ohne — Default deshalb aus. Schalter bleibt zum Experimentieren.
VLM_REASONING = os.getenv("STUDIO_VLM_REASONING", "0") == "1"
# Silent-Turn-Protokoll: entscheidet sich das Modell gegen eine Antwort, gibt es
# ausschließlich dieses Token aus; das Backend verwirft die Antwort dann
# vollständig (kein TTS, kein Audio, keine sichtbare Änderung im Client).
SILENT_TAG = os.getenv("STUDIO_SILENT_TAG", "no_answer_needed")
# Mindestanteil verworfener Silent-Turns (Persona-Default; pro Persona übersteuerbar).
VLM_SILENT_RATIO_DEFAULT = float(os.getenv("STUDIO_SILENT_RATIO", "0.8"))
# Englische Prompts sind Pflicht: deutsche Fragen liefern unbrauchbare
# Antworten ("Ja, sie ist gerade.") und schieben eine lange Reasoning-Kette vor.
VLM_DESCRIBE_PROMPT = os.getenv(
    "STUDIO_VLM_DESCRIBE_PROMPT",
    "In one short English sentence: what is the person doing, and what is their facial expression?",
)
