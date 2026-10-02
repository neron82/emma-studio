"""Entry point: run web UI and Telegram transport in one process.

Default: web only. `--telegram` adds Telegram polling on a dedicated event-loop
thread. Configuration and polling startup complete before the web app launches;
web shutdown stops and joins the Telegram thread.

`--log_transcripts` is the CLI spelling of ``STUDIO_LOG_TRANSCRIPTS=1``: the
app reads that env var when it is imported below, so both entries keep the
same privacy default (content stays out of the logs).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading

from . import config
from .core.store import Store


def main() -> int:
    parser = argparse.ArgumentParser(description="Emma Studio")
    parser.add_argument("--telegram", action="store_true", help="also run the Telegram transport")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--ssl-certfile", help="TLS certificate file (requires --ssl-keyfile)")
    parser.add_argument("--ssl-keyfile", help="TLS private key file (requires --ssl-certfile)")
    parser.add_argument(
        "--log_transcripts",
        action="store_true",
        help="write full conversation content to the log (sensitive — see transcript_logging)",
    )
    args = parser.parse_args()
    if (args.ssl_certfile is None) != (args.ssl_keyfile is None):
        parser.error("--ssl-certfile and --ssl-keyfile must be supplied together")

    # Must land before the app module is imported below; it configures logging.
    if args.log_transcripts:
        os.environ["STUDIO_LOG_TRANSCRIPTS"] = "1"

    transport = None
    if args.telegram:
        from .deliver.telegram_config import TelegramConfigError, load_settings

        try:
            settings = load_settings()
        except TelegramConfigError as exc:
            print(f"Telegram configuration error: {exc}", file=sys.stderr)
            return 2

        from .deliver.telegram import TelegramTransport

    store = Store(config.DB_PATH)  # ensure schema exists before transports start
    thread = None
    state = {}
    ready = threading.Event()
    shutdown = threading.Event()

    if args.telegram:
        transport = TelegramTransport(store, settings)

        async def _serve_telegram() -> None:
            state["loop"] = asyncio.get_running_loop()
            state["stop"] = asyncio.Event()
            if shutdown.is_set():
                state["stop"].set()
            await transport.serve(state["stop"], ready=ready)

        def _telegram_thread() -> None:
            try:
                asyncio.run(_serve_telegram())
            except Exception:
                # Network exceptions can contain token-bearing request URLs.
                # Never print their text or exception tracebacks.
                state["failed"] = True
                print("Telegram polling failed; check configuration and connectivity", file=sys.stderr)
            finally:
                ready.set()

        thread = threading.Thread(target=_telegram_thread, name="telegram")
        thread.start()

    try:
        if thread is not None:
            ready.wait()
            if state.get("failed"):
                return 2

        import uvicorn
        from .deliver.app import app

        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            ssl_certfile=args.ssl_certfile,
            ssl_keyfile=args.ssl_keyfile,
        )
    finally:
        if thread is not None:
            shutdown.set()
            loop = state.get("loop")
            if thread.is_alive() and loop is not None:
                try:
                    loop.call_soon_threadsafe(state["stop"].set)
                except RuntimeError:
                    pass  # the Telegram thread has already closed its event loop
            thread.join()
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
