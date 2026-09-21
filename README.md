# Emma Studio

Studio für Avatar-Kommunikation: Telefonat, animierter Avatar-Facetime (geplant),
fotorealistische Video-Messages. Neubau — ersetzt langfristig den alten Chatbot.

## Architektur

```
studio/
├── config.py          # eine Config für alles (LLM, Speech, Render, RAG)
├── core/
│   ├── store.py       # SQLite: personas / jobs / facts (FTS5 + bge-m3 hybrid)
│   └── pipeline.py    # Job-State-Machine: draft→scripted→voiced→rendering→done
├── agent/
│   ├── dialog.py      # Persona-Prompt + Faktenkarten → LLM-Turn
│   └── search.py      # SearXNG-Fallback bei unbeantwortbaren Fragen
├── speech/
│   ├── engine.py      # Realtime-Engine: VAD→STT→LLM→TTS Thread-Pool (Lazy-Build)
│   ├── vad_handler.py # Silero-VAD + smart-turn (semantisches Ende, Barge-in)
│   ├── stt.py         # faster-whisper (lazy, resident — geteilt mit Realtime-STT)
│   ├── tts.py         # OmniVoice (/v1/audio/speech, Profile-Voices)
│   └── router.py      # h3_native vs tts_pre Entscheidung
├── render/comfy.py    # ComfyUI-Client + Workflow-Injektion (fl2va BF16, RTX 2x)
└── deliver/
    ├── app.py         # FastAPI + Web-UI (:8890)
    ├── realtime/      # OpenAI-Realtime WebSocket (/v1/realtime) + Service/Router
    └── telegram.py    # Smart_Hermione_bot Transport (--telegram Flag)
```

## Ebenen

| Ebene | Optik | Status |
|---|---|---|
| Telefonat | Audio-only | **live** — OpenAI-Realtime WS (`/v1/realtime`): VAD→STT→LLM→TTS, Barge-in, Persona+RAG |
| Facetime | VRM-Avatar im Browser | **live** (three-vrm, Viseme-Lip-Sync via `emma.visemes`) |
| Video Message | fotoreal (MiniMax H3) | Modul fertig, Anbindung offen |

## RAG — Faktenkarten, keine Session-Dumps

`facts`-Tabelle mit FTS5 + bge-m3-Embeddings (Hybrid-Ranking). Einfügen per
Web-UI oder `/fact <text>` in Telegram. Persona-scoped Facts ranken vor globalen.

## Start

```bash
./studio.sh start            # ganzer Stack: Gemma-LLM (:8080) → TTS (:8880) → Studio (:8890)
./studio.sh status           # Units + Ports + geteilte Dienste
./studio.sh stop
./studio.sh start --no-llm   # ohne lokales LLM (Cloud-Modell, oder wenn der VRAM anderswo läuft)
```

Kein Autostart: `emma-studio.service` und `omnivoice.service` sind bewusst *disabled* — der
Stack belegt VRAM nur auf Zuruf. Das LLM steuert `~/models/gemma4_26b/launch-llama-server.sh`
(GPU0, Kontext-Autosizing nach freiem VRAM, MTP-Draft); die Unit `llama-llm.service` ist
veraltet und wird nicht benutzt. Geteilte Dienste (SearXNG :8081, bge-m3 :7777) gehören nicht
zum Stack und werden nie angefasst — `status` zeigt nur, ob sie erreichbar sind.
# Telegram zusätzlich (erst wenn alter Chatbot gestoppt ist!):
# ExecStart um "--telegram" ergänzen bzw. python -m studio --telegram
```

## Secrets (.env)

```
STUDIO_LLM_API_KEY=...        # OpenRouter
STUDIO_TG_BOT_TOKEN=...       # aus ~/projects/chatbot/telegram_secrets.json
STUDIO_TG_ALLOWED_CHATS=<chat-id>
```

## Wichtige Hinweise

- **Bot-Token-Konflikt:** Smart_Hermione_bot wird vom alten Chatbot gepollt.
  Beide gleichzeitig = Update-Kampf. Erst Chatbot stoppen, dann `--telegram`.
- OmniVoice: `~/services/omnivoice/toggle-omnivoice` (Port 8880, 3060 Ti)
- ComfyUI muss laufen für Video-Messages (`systemctl --user start comfyui`)
- Workflows in `workflows/` stammen aus video-gen-bot (fl2va BF16, beta57,
  Musubi-LoRA 0.4, RTX-VSR 2× ULTRA)

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```
