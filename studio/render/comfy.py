"""ComfyUI render module — client + workflow injection.

Owns the studio's copies of the workflows (workflows/*.json, derived from
video-gen-bot). One render at a time via a threading lock; the 3090 is
single-tenant.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

import httpx

from .. import config

RENDER_LOCK = threading.Lock()

# Node map for the bot-derived workflows (turbo.json / noturbo.json)
NODES = {
    "prompt": "58",     # MiniMaxH3ReferenceToVideo
    "combine": "20",    # VHS_VideoCombine
    "seed": "13",       # RandomNoise
    "image": "8",       # LoadImage
    "audio": "59",      # LoadAudio
    "scheduler": "16",  # BasicScheduler
}


class RenderError(RuntimeError):
    pass


def load_workflow(name: str) -> dict:
    path = config.WORKFLOW_DIR / f"{name}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RenderError(f"cannot read workflow {path}: {e}") from e


def build_workflow(*, workflow: str, prompt: str, seed: int,
                   image_name: str, audio_name: str | None,
                   length_frames: int, width: int | None = None,
                   height: int | None = None, steps: int = 20) -> dict:
    wf = load_workflow(workflow)
    wf[NODES["prompt"]]["inputs"]["prompt"] = prompt
    wf[NODES["prompt"]]["inputs"]["length"] = int(length_frames)
    if width:
        wf[NODES["prompt"]]["inputs"]["width"] = int(width)
    if height:
        wf[NODES["prompt"]]["inputs"]["height"] = int(height)
    wf[NODES["combine"]]["inputs"]["filename_prefix"] = f"studio/{workflow}_{uuid.uuid4().hex[:8]}"
    wf[NODES["seed"]]["inputs"]["noise_seed"] = int(seed)
    wf[NODES["image"]]["inputs"]["image"] = image_name
    if audio_name:
        wf[NODES["audio"]]["inputs"]["audio"] = audio_name
    if workflow == "noturbo":
        wf[NODES["scheduler"]]["inputs"]["steps"] = max(2, min(60, int(steps)))
    return wf


async def upload_file(path: Path) -> str:
    """Upload a reference asset; returns the server-side filename."""
    path = Path(path)
    if not path.is_file():
        raise RenderError(f"asset not found: {path}")
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{config.COMFY_URL.rstrip('/')}/upload/image",
            data={"overwrite": "true"},
            files={"image": (path.name, path.read_bytes())},
        )
    if resp.status_code != 200:
        raise RenderError(f"upload HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    name = data.get("name") or data.get("image")
    if not name:
        raise RenderError(f"upload returned no filename: {data}")
    return name


async def _queue_and_wait(workflow: dict, *, timeout: float = 1800.0) -> dict:
    """Queue the prompt and poll history until a video output exists."""
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{config.COMFY_URL.rstrip('/')}/prompt", json={"prompt": workflow}
        )
        if resp.status_code != 200:
            raise RenderError(f"/prompt HTTP {resp.status_code}: {resp.text[:300]}")
        prompt_id = resp.json().get("prompt_id")
        if not prompt_id:
            raise RenderError(f"/prompt returned no prompt_id: {resp.text[:200]}")

        deadline = time.monotonic() + timeout
        while True:
            hist = (await client.get(f"{config.COMFY_URL.rstrip('/')}/history/{prompt_id}")).json()
            entry = hist.get(prompt_id)
            if entry:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise RenderError(f"render failed: {status.get('message')}")
                for node in (entry.get("outputs") or {}).values():
                    for key in ("videos", "gifs"):
                        for item in node.get(key) or []:
                            if item.get("filename"):
                                return item
            if time.monotonic() >= deadline:
                raise RenderError(f"timeout after {timeout:.0f}s waiting for {prompt_id}")
            await __import__("asyncio").sleep(3)


async def download_video(item: dict, dest: Path) -> Path:
    params = {"filename": item["filename"]}
    if item.get("subfolder"):
        params["subfolder"] = item["subfolder"]
    if item.get("type"):
        params["type"] = item["type"]
    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.get(f"{config.COMFY_URL.rstrip('/')}/view", params=params)
    if resp.status_code != 200:
        raise RenderError(f"/view HTTP {resp.status_code}")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return dest


async def render(*, workflow: str, prompt: str, image_path: Path,
                 audio_path: Path | None, length_frames: int,
                 width: int | None = None, height: int | None = None,
                 steps: int = 20, out_path: Path, seed: int | None = None) -> Path:
    """Full render cycle. Serialized by RENDER_LOCK (thread-level); async
    callers should run this in an executor to avoid blocking the loop."""
    import asyncio
    import random

    if seed is None:
        seed = random.randrange(1, 2**31)

    def _blocking() -> Path:
        with RENDER_LOCK:
            image_name = _run(upload_file(image_path))
            audio_name = _run(upload_file(audio_path)) if audio_path else None
            wf = build_workflow(
                workflow=workflow, prompt=prompt, seed=seed,
                image_name=image_name, audio_name=audio_name,
                length_frames=length_frames, width=width, height=height, steps=steps,
            )
            item = _run(_queue_and_wait(wf))
            return _run(download_video(item, out_path))

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _blocking)


def _run(coro):
    """Run a coroutine inside the executor thread (which has no loop)."""
    import asyncio
    return asyncio.run(coro)


def duration_to_frames(seconds: float) -> int:
    seconds = max(1.0, min(config.MAX_VIDEO_SECONDS, float(seconds)))
    return int(round(seconds * config.FRAME_RATE))
