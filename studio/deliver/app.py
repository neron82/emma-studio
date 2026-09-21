"""FastAPI app — REST + minimal web UI.

Endpoints:
  GET  /                     web UI
  GET  /api/personas         list personas
  POST /api/personas         create persona
  PATCH /api/personas/{id}   update persona
  POST /api/personas/{id}/image   upload reference image
  POST /api/personas/{id}/voice   upload voice sample
  GET  /api/jobs             recent jobs
  POST /api/jobs/video       create video-message job (runs in background)
  POST /api/jobs/turn        create call-turn job (text in, audio out)
  GET  /api/jobs/{id}        job status
  GET  /api/jobs/{id}/audio  serve TTS audio
  GET  /api/jobs/{id}/video  serve rendered mp4
  WS   /v1/realtime        realtime voice pipeline (OpenAI Realtime protocol)
  GET  /api/facts            search facts (q=)
  POST /api/facts            add fact card
  DELETE /api/facts/{id}     delete fact card
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .. import config
from ..core.store import Store
from ..core import pipeline
from ..speech import engine
from .realtime.websocket_router import register_realtime

_rt_holder: dict[str, Any] = {}

# Logging: Bibliotheken bleiben bei WARNING, Emmas eigene Logger auf INFO
# (STUDIO_LOG_LEVEL steuert es). Ohne das verschwinden genau die Zeilen, die
# erklären was der Dialog sieht — z. B. der injizierte Blickkontext.
logging.basicConfig(level=logging.WARNING)
logging.getLogger("studio").setLevel(os.getenv("STUDIO_LOG_LEVEL", "INFO").upper())


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Enter the realtime send loops; stop engine handler threads on teardown."""
    cm = _rt_holder.get("rt_lifespan")
    if cm is not None:
        async with cm():
            yield
        engine.shutdown()
    else:
        yield


app = FastAPI(title="Emma Studio", lifespan=_lifespan)
store = Store(config.DB_PATH)
app.mount("/static", StaticFiles(directory=config.BASE_DIR / "web" / "static"), name="static")

# In-memory dialog histories per persona (legacy batch call turns only)
_histories: dict[str, list[dict]] = {}

# OpenAI-Realtime pipeline pool (built lazily on the first WS connection)
_rt_pool: list = []
_rt_stop_event = threading.Event()
if config.REALTIME_ENABLED:
    _rt_holder["rt_lifespan"] = register_realtime(
        app, _rt_pool, _rt_stop_event, build_pool=engine.build_into
    )


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (config.BASE_DIR / "web" / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/facetime", response_class=HTMLResponse)
async def facetime_page() -> str:
    """Legacy route — the new shell hosts facetime at /#facetime."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/#facetime", status_code=302)


@app.get("/api/personas/portrait")
async def persona_portrait():
    """Portrait of the first persona with a ref image — facetime fallback."""
    for p in store.list_personas():
        if p.get("ref_image") and Path(p["ref_image"]).is_file():
            media = "image/png" if p["ref_image"].endswith(".png") else "image/jpeg"
            return FileResponse(p["ref_image"], media_type=media)
    raise HTTPException(404, "no persona portrait")


@app.get("/api/health/{service}")
async def service_health(service: str):
    """Lightweight service probes for the sidebar status dots."""
    import httpx
    ok = False
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            if service == "omnivoice":
                resp = await client.get(f"{config.OMNIVOICE_URL.rstrip('/')}/health")
                ok = resp.status_code == 200 and resp.json().get("ready") is True
            elif service == "comfy":
                resp = await client.get(f"{config.COMFY_URL.rstrip('/')}/system_stats")
                ok = resp.status_code == 200
    except (httpx.HTTPError, ValueError):
        ok = False
    return {"ok": ok}


@app.post("/api/fillers/pick")
async def pick_filler_endpoint(payload: dict):
    """Pick a backchannel filler for a user utterance.

    Returns {url, phrase, category, duration_s} or {url: null}.
    Called by the frontend immediately after STT completes, in parallel
    with the LLM turn.
    """
    from ..speech import backchannel
    persona = store.get_persona(payload.get("persona_id") or "") or {}
    voice = persona.get("voice_profile") or "emma_alicia"
    entry = backchannel.pick_filler(
        payload.get("text", ""), voice=voice,
        max_duration_s=float(payload.get("max_duration_s", 2.5)),
    )
    if not entry:
        return {"url": None}
    return {
        "url": f"/api/fillers/audio/{entry['file'].split('/', 1)[1]}",
        "phrase": entry["phrase"],
        "category": entry["category"],
        "duration_s": entry["duration_s"],
    }


@app.get("/api/fillers/audio/{path:path}")
async def filler_audio(path: str):
    """Serve a pre-generated filler clip (clips/<voice>/<file>.wav)."""
    if ".." in path or path.startswith("/"):
        raise HTTPException(400, "bad path")
    p = config.DATA_DIR / "fillers" / "clips" / path
    if not p.is_file():
        raise HTTPException(404, "no such clip")
    return FileResponse(p, media_type="audio/wav")


@app.post("/api/stt")
async def api_stt(audio: UploadFile = File(...)):
    """Transcribe a browser-recorded webm/ogg blob."""
    import asyncio as _asyncio
    dest = config.DATA_DIR / "stt" / f"{audio.filename or 'rec'}.webm"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(await audio.read())
    from ..speech import stt
    loop = _asyncio.get_running_loop()
    try:
        text = await loop.run_in_executor(None, stt.transcribe, dest)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e)[:200])
    return {"text": text}


# -- personas ---------------------------------------------------------------

@app.get("/api/personas")
async def list_personas():
    return store.list_personas()


@app.post("/api/personas")
async def create_persona(name: str = Form(...), personality_prompt: str = Form(""),
                         language: str = Form("German"), vrm_model: str = Form(""),
                         silent_summary: str = Form("on"), silent_ratio: float = Form(0.8)):
    p = store.create_persona(name, personality_prompt=personality_prompt,
                             language=language, vrm_model=vrm_model or None,
                             silent_summary=silent_summary, silent_ratio=silent_ratio)
    return p


@app.patch("/api/personas/{persona_id}")
async def update_persona(persona_id: str, payload: dict):
    p = store.update_persona(persona_id, **payload)
    if not p:
        raise HTTPException(404, "persona not found")
    return p


@app.post("/api/personas/{persona_id}/image")
async def upload_image(persona_id: str, file: UploadFile = File(...)):
    dest = config.DATA_DIR / "personas" / persona_id / f"ref_{file.filename}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(await file.read())
    return store.update_persona(persona_id, ref_image=str(dest))


@app.post("/api/personas/{persona_id}/voice")
async def upload_voice(persona_id: str, file: UploadFile = File(...)):
    dest = config.DATA_DIR / "personas" / persona_id / f"voice_{file.filename}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(await file.read())
    return store.update_persona(persona_id, voice_sample=str(dest))


# -- jobs -------------------------------------------------------------------

@app.post("/api/jobs/video")
async def create_video_job(persona_id: str = Form(...), topic: str = Form(...),
                           speech_mode: str = Form("")):
    job = store.create_job(persona_id, "video_message", topic)
    asyncio.create_task(pipeline.run_video_message(store, job, speech_mode=speech_mode or None))
    return job


@app.post("/api/jobs/turn")
async def create_call_turn(persona_id: str = Form(...), message: str = Form(...)):
    job = store.create_job(persona_id, "call_turn", message)
    history = _histories.setdefault(persona_id, [])
    asyncio.create_task(_run_turn_and_remember(persona_id, job, history))
    return job



@app.get("/api/history/{persona_id}")
async def get_history(persona_id: str, limit: int = 50):
    """Persistent conversation log — survives reloads."""
    return store.get_history(persona_id, limit=limit)


async def _run_turn_and_remember(persona_id: str, job: dict, history: list[dict]):
    done = await pipeline.run_call_turn(store, job, history)
    if done["state"] == "done":
        history.append({"role": "user", "content": job["user_input"]})
        history.append({"role": "assistant", "content": done["script"]})
        del history[:-12]


@app.get("/api/jobs")
async def list_jobs(persona_id: str | None = None):
    return store.list_jobs(persona_id=persona_id)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


@app.get("/api/jobs/{job_id}/audio")
async def job_audio(job_id: str):
    job = store.get_job(job_id)
    if not job or not job["speech_path"]:
        raise HTTPException(404, "no audio")
    return FileResponse(job["speech_path"], media_type="audio/wav")


@app.get("/api/jobs/{job_id}/chunks")
async def job_chunks(job_id: str):
    """List sentence-chunk audio files for sequential playback."""
    from pathlib import Path as _P
    d = config.DATA_DIR / "jobs" / job_id
    if not d.is_dir():
        return []
    files = sorted(d.glob("chunk_*.wav"))
    return [f"/api/jobs/{job_id}/chunks/{f.name}" for f in files]


@app.get("/api/jobs/{job_id}/chunks/{name}")
async def job_chunk(job_id: str, name: str):
    if not name.startswith("chunk_") or not name.endswith(".wav"):
        raise HTTPException(400, "bad chunk name")
    p = config.DATA_DIR / "jobs" / job_id / name
    if not p.is_file():
        raise HTTPException(404, "no such chunk")
    return FileResponse(p, media_type="audio/wav")


@app.get("/api/jobs/{job_id}/video")
async def job_video(job_id: str):
    job = store.get_job(job_id)
    if not job or not job["render_path"]:
        raise HTTPException(404, "no video")
    return FileResponse(job["render_path"], media_type="video/mp4")


# -- facts ------------------------------------------------------------------

@app.get("/api/facts")
async def search_facts(q: str = "", persona_id: str | None = None):
    if q:
        return store.search_facts(q, persona_id=persona_id, limit=50)
    # no query: list latest (raw scan is fine at fact-card scale)
    rows = store.conn.execute(
        "SELECT * FROM facts ORDER BY created_at DESC LIMIT 100"
    ).fetchall()
    return [dict(r) for r in rows]


@app.post("/api/facts")
async def add_fact(fact: str = Form(...), persona_id: str = Form("")):
    return store.add_fact(fact, persona_id=persona_id or None)


@app.delete("/api/facts/{fact_id}")
async def delete_fact(fact_id: str):
    if not store.delete_fact(fact_id):
        raise HTTPException(404, "fact not found")
    return {"ok": True}


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8890)


if __name__ == "__main__":
    main()
