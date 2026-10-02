"""Telegram credentials and private-chat access, without importing transports.

Explicit STUDIO_TG_* environment values override predecessor JSON files. An
explicit empty allowlist is an error, never a request for public access.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


class TelegramConfigError(ValueError):
    """A safe, credential-free configuration error for startup reporting."""


@dataclass(frozen=True)
class BotSettings:
    token: str = field(repr=False)
    bot_name: str
    allowed_chats: frozenset[int]

    def __post_init__(self) -> None:
        if not isinstance(self.token, str) or not self.token.strip() or any(
            char.isspace() for char in self.token
        ):
            raise TelegramConfigError("Telegram bot token is missing or invalid")
        if not isinstance(self.bot_name, str) or not re.fullmatch(
            r"[A-Za-z0-9_]+", self.bot_name
        ):
            raise TelegramConfigError("Telegram bot name is missing or invalid")
        if not self.allowed_chats or any(
            type(chat_id) is not int or chat_id <= 0 for chat_id in self.allowed_chats
        ):
            raise TelegramConfigError("Telegram requires a nonempty private-chat allowlist")
        object.__setattr__(self, "allowed_chats", frozenset(self.allowed_chats))


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, UnicodeError):
        # Never include JSON parser text, paths, or exception chains: malformed
        # secrets files can otherwise leak their contents in startup tracebacks.
        raise TelegramConfigError(f"Cannot read Telegram {path.name}") from None
    if not isinstance(data, dict):
        raise TelegramConfigError(f"Invalid Telegram {path.name}")
    return data


def _explicit_chats(value: str) -> frozenset[int]:
    parts = [part for part in re.split(r"[\s,]+", value.strip()) if part]
    if not parts or any(not part.isascii() or not part.isdecimal() for part in parts):
        raise TelegramConfigError("STUDIO_TG_ALLOWED_CHATS requires positive private chat IDs")
    chats = frozenset(int(part) for part in parts)
    if any(chat_id <= 0 for chat_id in chats):
        raise TelegramConfigError("STUDIO_TG_ALLOWED_CHATS requires positive private chat IDs")
    return chats


def _known_private_chats(settings: dict) -> frozenset[int]:
    chats: set[int] = set()
    for name in ("chat_sessions", "chat_audio_modes"):
        mapping = settings.get(name, {})
        if not isinstance(mapping, dict):
            raise TelegramConfigError("Invalid Telegram predecessor chat mappings")
        for key in mapping:
            # Telegram groups/supergroups have negative IDs. Only known private
            # IDs are inherited, and the transport also checks chat.type.
            if isinstance(key, str) and key.isascii() and key.isdecimal():
                chat_id = int(key)
                if chat_id > 0:
                    chats.add(chat_id)
    return frozenset(chats)


def load_settings() -> BotSettings:
    """Read credentials in place; never import or launch the predecessor bot."""
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    predecessor = Path(
        os.environ.get("STUDIO_TG_CHATBOT_DIR", "~/projects/chatbot")
    ).expanduser()

    if "STUDIO_TG_BOT_TOKEN" in os.environ:
        token = os.environ["STUDIO_TG_BOT_TOKEN"]
    else:
        token = _read_json(predecessor / "telegram_secrets.json").get("bot_token", "")

    legacy = {}
    if "STUDIO_TG_BOT_NAME" not in os.environ or "STUDIO_TG_ALLOWED_CHATS" not in os.environ:
        legacy = _read_json(predecessor / "telegram_settings.json")
    name = os.environ.get("STUDIO_TG_BOT_NAME", legacy.get("bot_name", ""))
    if isinstance(name, str):
        name = name.strip().removeprefix("@")

    if "STUDIO_TG_ALLOWED_CHATS" in os.environ:
        allowed = _explicit_chats(os.environ["STUDIO_TG_ALLOWED_CHATS"])
    else:
        allowed = _known_private_chats(legacy)
    return BotSettings(token=token, bot_name=name, allowed_chats=allowed)
