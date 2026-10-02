"""Pipeline orchestrator — executes the job state machine.

video_message flow:
  draft -> scripted (agent writes scene+lines) -> voiced (tts_pre) or direct
  -> rendering (ComfyUI) -> done

call_turn flow:
  draft -> scripted (agent reply text) -> voiced (tts) -> done
  (no render; audio is the deliverable)
"""
from __future__ import annotations

from pathlib import Path

from ..agent import dialog as agent_dialog
from ..agent import search as web_search
from ..agent.models import resolve_endpoint
from ..core.store import Store
from ..render import comfy
from ..speech import router as speech_router
from ..speech import tts
from .. import config

DATA_DIR = config.DATA_DIR


class PipelineError(RuntimeError):
    pass


def _job_paths(job_id: str) -> dict[str, Path]:
    d = DATA_DIR / "jobs" / job_id
    return {"dir": d, "audio": d / "speech.wav", "video": d / "message.mp4"}


async def _script_video_message(store: Store, job: dict) -> dict:
    """Agent turns the user topic into a full video-generation prompt."""
    persona = store.get_persona(job["persona_id"])
    facts = store.search_facts(job["user_input"], persona_id=persona["id"])
    lang = persona.get("language") or "German"
    system = (
        f"{persona['personality_prompt'].strip()}\n\n"
        f"IMPORTANT: Write EVERYTHING in {lang}. No exceptions.\n\n"
        "You write the prompt for a reference-to-video model. The person in the "
        "video speaks a short message to the viewer. Output ONLY the final video "
        "prompt: scene description, appearance consistency with the reference "
        f"image, and the exact spoken lines in {lang}. Keep it under 400 "
        "characters so the video stays short. No headings, no commentary."
    )
    if facts:
        lines = "\n".join(f"- {f['fact']}" for f in facts)
        system += f"\n\nKnown facts about the viewer (use them, never contradict them):\n{lines}"
    import httpx
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Topic for the video message:\n{job['user_input']}"},
    ]
    endpoint = resolve_endpoint(persona)
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{endpoint.base_url.rstrip('/')}/chat/completions",
            headers=endpoint.headers(job["id"]),
            json={"model": endpoint.model, "messages": messages},
        )
    if resp.status_code != 200:
        raise PipelineError(f"script LLM HTTP {resp.status_code}: {resp.text[:200]}")
    script = resp.json()["choices"][0]["message"]["content"].strip()
    return store.set_job_state(job["id"], "scripted", script=script)


async def _voice_job(store: Store, job: dict, mode: str) -> dict:
    """Produce the audio track. tts_pre only; h3_native skips this state."""
    if mode != "tts_pre":
        return store.set_job_state(job["id"], "voiced", speech_mode="h3_native")
    persona = store.get_persona(job["persona_id"])
    voice = persona.get("voice_profile") or "emma_alicia"  # OmniVoice profile id
    paths = _job_paths(job["id"])
    try:
        await tts.synthesize(
            job["script"], voice=voice,
            out_path=paths["audio"],
            language=persona.get("language"),
        )
    except tts.TTSError as e:
        # OmniVoice broke mid-flight — fall back to native speech.
        return store.set_job_state(
            job["id"], "voiced", speech_mode="h3_native",
            error=f"tts fallback: {e}",
        )
    return store.set_job_state(job["id"], "voiced", speech_mode="tts_pre")


async def _render_job(store: Store, job: dict) -> dict:
    persona = store.get_persona(job["persona_id"])
    if not persona.get("ref_image"):
        raise PipelineError(f"persona '{persona['name']}' has no ref_image")
    paths = _job_paths(job["id"])
    frames = comfy.duration_to_frames(estimate_duration(job["script"]))
    audio_path = paths["audio"] if job["speech_mode"] == "tts_pre" else None
    out = await comfy.render(
        workflow=config.DEFAULT_WORKFLOW,
        prompt=job["script"],
        image_path=Path(persona["ref_image"]),
        audio_path=audio_path,
        length_frames=frames,
        out_path=paths["video"],
    )
    return store.set_job_state(job["id"], "done", render_path=str(out))


def estimate_duration(script: str | None) -> float:
    """Rough spoken-duration estimate: ~14 chars/second German speech."""
    if not script:
        return 8.0
    return max(3.0, min(config.MAX_VIDEO_SECONDS, len(script) / 14.0))


async def run_video_message(store: Store, job: dict, *, speech_mode: str | None = None) -> dict:
    """Execute the full pipeline; returns the final job row."""
    try:
        mode = await speech_router.choose_speech_mode(requested=speech_mode)
        job = await _script_video_message(store, job)
        job = await _voice_job(store, job, mode)
        job = await _render_job(store, job)
        return job
    except Exception as e:  # noqa: BLE001 — persist any failure on the job
        return store.set_job_state(job["id"], "failed", error=str(e)[:500])


async def run_call_turn(store: Store, job: dict, history: list[dict]) -> dict:
    """Dialog turn: user utterance -> agent reply -> TTS audio. No render."""
    import time as _time
    from .latency import _persist
    timings: dict = {}
    t_total = _time.monotonic()
    try:
        persona = store.get_persona(job["persona_id"])

        t0 = _time.monotonic()
        facts = store.search_facts(job["user_input"], persona_id=persona["id"])
        timings["facts"] = round(_time.monotonic() - t0, 3)

        # Web-search fallback: only when no fact card matched and search enabled.
        search_results = None
        used_search = False
        if not facts and config.SEARCH_ENABLED and _looks_like_question(job["user_input"]):
            t0 = _time.monotonic()
            search_results = await web_search.search(job["user_input"])
            timings["search"] = round(_time.monotonic() - t0, 3)
            used_search = bool(search_results)

        t0 = _time.monotonic()
        reply = await agent_dialog.chat(
            persona, history, job["user_input"],
            facts[: config.MAX_FACTS_PER_QUERY], search_results,
        )
        timings["llm"] = round(_time.monotonic() - t0, 3)
        job = store.set_job_state(job["id"], "scripted", script=reply)

        voice = persona.get("voice_profile") or "emma_alicia"  # OmniVoice profile id
        paths = _job_paths(job["id"])
        from ..speech.chunks import synthesize_chunks
        t0 = _time.monotonic()

        # measure first-chunk latency separately from total synthesis
        timings_box = {"done": False}

        async def _first_chunk_probe():
            if not timings_box["done"]:
                timings_box["done"] = True
                timings["tts_first"] = round(_time.monotonic() - t0, 3)

        all_paths = await synthesize_chunks(
            reply, voice=voice, job_dir=paths["dir"],
            language=persona.get("language"),
            on_first_chunk=_first_chunk_probe,
        )
        timings["tts"] = round(_time.monotonic() - t0, 3)
        # concat into one full wav for simple clients; chunks stay for streaming
        _concat_wavs(all_paths, paths["audio"])
        timings["total"] = round(_time.monotonic() - t_total, 3)
        job = store.set_job_state(
            job["id"], "done", speech_path=str(paths["audio"]), speech_mode="tts_pre",
            search_used=used_search,
        )
        _persist(store, job["id"], timings)
        return job
    except Exception as e:  # noqa: BLE001
        timings["total"] = round(_time.monotonic() - t_total, 3)
        job = store.set_job_state(job["id"], "failed", error=str(e)[:500])
        _persist(store, job["id"], timings)
        return job


def _concat_wavs(paths: list[Path], dest: Path) -> None:
    """Concatenate PCM16 wav files (same format) via ffmpeg."""
    import subprocess
    if len(paths) == 1:
        dest.write_bytes(paths[0].read_bytes())
        return
    listfile = dest.parent / "concat.txt"
    listfile.write_text("".join(f"file '{p.resolve()}'\n" for p in paths))
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
         "-c", "copy", str(dest)],
        check=True, capture_output=True,
    )
    listfile.unlink(missing_ok=True)


def _looks_like_question(text: str) -> bool:
    t = text.strip().lower()
    markers = ("?", "wer ", "was ", "wann ", "wo ", "wie ", "warum ", "welche",
               "who ", "what ", "when ", "where ", "how ", "why ")
    return any(m in t for m in markers)
