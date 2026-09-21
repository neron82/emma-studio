"""Entry point: run web UI and Telegram transport in one process.

Default: web only. `--telegram` adds the Telegram polling (in a thread —
python-telegram-bot's run_polling owns its own event loop, so it gets a
dedicated thread; both layers share the same SQLite store).
"""
from __future__ import annotations

import argparse
import sys
import threading

from . import config
from .core.store import Store


def main() -> int:
    parser = argparse.ArgumentParser(description="Emma Studio")
    parser.add_argument("--telegram", action="store_true", help="also run the Telegram transport")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8890)
    args = parser.parse_args()

    store = Store(config.DB_PATH)  # ensure schema exists before transports start

    if args.telegram:
        from .deliver.telegram import TelegramTransport

        def _tg() -> None:
            TelegramTransport(store).run()

        threading.Thread(target=_tg, name="telegram", daemon=True).start()

    import uvicorn
    from .deliver.app import app

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
