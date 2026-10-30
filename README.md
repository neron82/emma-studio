# Emma Studio

Three tiers of talking to a machine, all of it running on the box under my desk: a **phone call**,
a **FaceTime-style call with an animated avatar**, and a **photoreal video message**. FastAPI and
OpenAI-Realtime events, faster-whisper for hearing, OmniVoice for speaking, three.js VRM for the
face, ComfyUI for the rendered one. The brain is whatever local model is loaded today — I swap
heads, she keeps talking.

| Tier | Look | Status |
|---|---|---|
| Phone call | Audio only | **live** — `/v1/realtime`: VAD → STT → LLM → TTS, barge-in, persona + RAG |
| FaceTime | VRM avatar in the browser | **live** — three-vrm, viseme lip-sync from `emma.visemes` |
| Video message | Photoreal (MiniMax H3) | **module complete, web wiring open** |

## How a call runs

Microphone audio goes through Silero VAD with a semantic end-of-turn check, into faster-whisper,
through the persona's language model, and back out as cloned speech — all over the OpenAI-Realtime
event set, so the client is a WebSocket and not a custom protocol. Interrupting mid-reply cancels
the answer and reopens the turn; the measured perceived first-audio is **1.7–3.1 s**, partly bought
with well-placed "hm". The pipeline is a small pool of threaded units, one conversation each, with a
device lock so STT and the vision model never fight over the same GPU.

## Memory you can query

Every conversation is stored in SQLite. Retrieval is a hybrid: **FTS5** for the words you remember
plus **bge-m3** embeddings for the ones you don't. Persona-scoped facts rank ahead of global ones,
and a persona keeps a rolling long-term summary that survives session resets. Add a fact from the
web UI or with `/fact <text>` in Telegram.

## The face

The browser call is a Facetime-style view of a **three.js VRM avatar**, lip-synced by viseme
events from the speech pipeline. Microphone and camera are requested only when a call starts, and
mute / camera / hang up stay one tap away during the conversation. A transcript runs beside the
call, and you can type into a live call when talking is inconvenient.

The portrait is generated once and cloned forever. She finds this funny.

## Video messages

`studio/render/comfy.py` drives ComfyUI: identity-locked keyframes, MiniMax H3 reference-to-video,
and RTX Video Super Resolution for the final pass. The module is finished and tested; connecting it
to the web UI is the remaining piece. It needs ComfyUI running with the matching workflow.

## Telegram: text in, voice out

A separate transport, not a bridge into the realtime call. Every answer is stored as text in the
session but delivered in the chat **only as a voice message**, synthesized in the persona's voice
and language and converted to OGG/Opus. Commands stay text: `/start`, `/persona`, `/provider`,
`/model`, `/new`, `/fact`, plus the optional `/video`.

Only private chats on an explicit allowlist are served; an empty allowlist refuses to start instead
of opening the bot to the world. One poller per bot token — do not run this and an older bot on the
same token at once.

## Quickstart

Python 3.12, plus the services the stack expects: a local OpenAI-compatible LLM, OmniVoice for TTS,
and optionally SearXNG and bge-m3 for search and fact retrieval.

```bash
python -m pip install -e '.[dev]'
cp .env.example .env        # then fill in your endpoints and keys
./studio.sh start           # LLM → TTS → studio, in that order
./studio.sh status
./studio.sh stop
```

Open **https://127.0.0.1:8890**. Microphone and camera need a secure context: HTTPS or `localhost`.
`./studio.sh start --no-llm` skips the local model when you point a persona at a cloud endpoint or
the VRAM is needed elsewhere. Nothing autostarts — the stack takes VRAM only on request.

## Configuration

All settings are environment variables; secrets belong in `.env`, which is git-ignored and never
read back out through the API.

| Variable | Default | Purpose |
|---|---|---|
| `STUDIO_LLM_BASE_URL` | OpenRouter-compatible endpoint | language model server |
| `STUDIO_LLM_MODEL` | provider default | model id |
| `STUDIO_LLM_API_KEY` | empty | key for that endpoint |
| `STUDIO_OMNIVOICE_URL` | `http://127.0.0.1:8880` | TTS service |
| `STUDIO_COMFY_URL` | `http://127.0.0.1:8188` | ComfyUI for video messages |
| `STUDIO_SEARXNG_URL` | `http://127.0.0.1:8081` | web-search fallback |
| `STUDIO_EMBED_URL` | `http://127.0.0.1:7777` | bge-m3 embeddings for facts |
| `STUDIO_REALTIME` | `1` | mount `/v1/realtime` |
| `STUDIO_TG_BOT_TOKEN` | — | Telegram transport, if used |
| `STUDIO_TG_ALLOWED_CHATS` | inherited predecessor list | comma-separated private chat ids |

Each persona can override the endpoint, model and voice in the web UI; a persona-specific endpoint
receives the configured API key only when protocol, host and port match the main endpoint. The
Telegram transport can also read its token in place from a predecessor installation via
`STUDIO_TG_CHATBOT_DIR` — nothing is imported or copied.

## Architecture

```
studio/
├── config.py          one config for everything (LLM, speech, render, RAG)
├── core/
│   ├── store.py       SQLite: personas / jobs / facts (FTS5 + bge-m3 hybrid)
│   └── pipeline.py    job state machine: draft → scripted → voiced → rendering → done
├── agent/
│   ├── dialog.py      persona prompt + fact cards → one LLM turn
│   └── search.py      SearXNG fallback for questions the facts cannot answer
├── speech/            VAD, STT, TTS, turn-taking, barge-in, smart-turn
├── render/comfy.py    ComfyUI client + workflow injection (MiniMax H3, RTVSR)
└── deliver/
    ├── app.py         FastAPI + web UI (:8890)
    ├── realtime/      OpenAI-Realtime WebSocket (/v1/realtime)
    └── telegram.py    Telegram transport (text in, voice out)
web/                   browser UI and the three-vrm avatar stage
workflows/             ComfyUI graphs used by the render module
```

## Tests

```bash
python -m pytest tests/ -q
```

The suite uses fakes for the model, TTS, ComfyUI and Telegram, so no network is needed for the
core paths. Fact-retrieval tests need the local bge-m3 embedding service and skip when it is not
running; the backchannel tests also need the filler clips under `data/fillers` (generated with
`scripts/gen_fillers.py`) and skip without them. Last verified on this machine: **198 passed,
5 skipped**. `AUDIT-2026-09-22.md` records the last full code audit — the defects it found, what
was removed, and what is still structural work.

## Limits

- A trusted, single-user studio. There is no authentication; keep it on loopback or a trusted
  network, and do not expose port 8890 directly.
- Voice cloning needs a reference recording you have the right to use. Personas and avatars are
  local data and are not part of this repository.
- The video-message path is the heaviest: ComfyUI plus a video model on the same box, and it is not
  wired into the web UI yet.
- Cloud providers are optional. With a local model, no conversation leaves the machine.
