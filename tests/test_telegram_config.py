"""Credential precedence and fail-closed private-chat configuration."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio.deliver import telegram_config as tg


@pytest.fixture()
def predecessor(tmp_path, monkeypatch):
    for name in ("STUDIO_TG_BOT_TOKEN", "STUDIO_TG_BOT_NAME", "STUDIO_TG_ALLOWED_CHATS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STUDIO_TG_CHATBOT_DIR", str(tmp_path))
    monkeypatch.setattr(tg, "load_dotenv", lambda *args, **kwargs: None)
    return tmp_path


def _legacy(root, *, token="123456:TEST_ONLY", name="Legacy_Test_bot", sessions=None, modes=None):
    (root / "telegram_secrets.json").write_text(json.dumps({"bot_token": token}), encoding="utf-8")
    (root / "telegram_settings.json").write_text(
        json.dumps({"bot_name": name, "chat_sessions": sessions or {}, "chat_audio_modes": modes or {}}),
        encoding="utf-8",
    )


def test_inherits_private_chats_from_both_predecessor_mappings(predecessor):
    _legacy(
        predecessor,
        sessions={"101": "session", "-202": "group", "0": "invalid"},
        modes={"303": True, "101": False, "-404": False},
    )
    settings = tg.load_settings()
    assert settings.token == "123456:TEST_ONLY"
    assert settings.bot_name == "Legacy_Test_bot"
    assert settings.allowed_chats == frozenset({101, 303})


def test_explicit_environment_replaces_all_predecessor_values(predecessor, monkeypatch):
    # With complete explicit configuration, inaccessible/malformed predecessor
    # files must not prevent running this bot account.
    (predecessor / "telegram_secrets.json").write_text("not json", encoding="utf-8")
    (predecessor / "telegram_settings.json").write_text("not json", encoding="utf-8")
    monkeypatch.setenv("STUDIO_TG_BOT_TOKEN", "654321:EXPLICIT_TEST")
    monkeypatch.setenv("STUDIO_TG_BOT_NAME", "@Explicit_Test_bot")
    monkeypatch.setenv("STUDIO_TG_ALLOWED_CHATS", "707, 808 909,707")
    settings = tg.load_settings()
    assert settings.token == "654321:EXPLICIT_TEST"
    assert settings.bot_name == "Explicit_Test_bot"
    assert settings.allowed_chats == frozenset({707, 808, 909})


def test_partial_environment_uses_predecessor_for_other_fields(predecessor, monkeypatch):
    _legacy(predecessor, sessions={"101": "existing"})
    monkeypatch.setenv("STUDIO_TG_BOT_TOKEN", "654321:EXPLICIT_TEST")
    settings = tg.load_settings()
    assert settings.token == "654321:EXPLICIT_TEST"
    assert settings.bot_name == "Legacy_Test_bot"
    assert settings.allowed_chats == frozenset({101})


@pytest.mark.parametrize("allowlist", ["", " , ", "-101", "0", "101,garbage", "101,1.5", "１０１"])
def test_invalid_explicit_allowlist_never_falls_back(predecessor, monkeypatch, allowlist):
    _legacy(predecessor, sessions={"101": "existing"})
    monkeypatch.setenv("STUDIO_TG_ALLOWED_CHATS", allowlist)
    with pytest.raises(tg.TelegramConfigError):
        tg.load_settings()


@pytest.mark.parametrize("sessions,modes", [({}, {}), ({"-101": "group"}, {"-202": True})])
def test_absent_private_allowlist_fails_closed(predecessor, sessions, modes):
    _legacy(predecessor, sessions=sessions, modes=modes)
    with pytest.raises(tg.TelegramConfigError):
        tg.load_settings()


def test_explicit_empty_token_never_falls_back(predecessor, monkeypatch):
    _legacy(predecessor, sessions={"101": "existing"})
    monkeypatch.setenv("STUDIO_TG_BOT_TOKEN", "")
    with pytest.raises(tg.TelegramConfigError):
        tg.load_settings()


def test_missing_credentials_fail_closed(predecessor, monkeypatch):
    monkeypatch.setenv("STUDIO_TG_BOT_NAME", "Explicit_Test_bot")
    monkeypatch.setenv("STUDIO_TG_ALLOWED_CHATS", "101")
    with pytest.raises(tg.TelegramConfigError):
        tg.load_settings()


@pytest.mark.parametrize("filename", ["telegram_secrets.json", "telegram_settings.json"])
def test_malformed_json_does_not_expose_secret_in_error(predecessor, filename):
    _legacy(predecessor, sessions={"101": "existing"})
    secret = "987654:SHOULD_NEVER_APPEAR"
    (predecessor / filename).write_text('{"bot_token":"' + secret, encoding="utf-8")
    with pytest.raises(tg.TelegramConfigError) as failure:
        tg.load_settings()
    assert secret not in str(failure.value)
    rendered = "".join(traceback.TracebackException.from_exception(failure.value).format())
    assert secret not in rendered


def test_token_hidden_from_settings_repr(predecessor):
    secret = "987654:SHOULD_NEVER_APPEAR"
    _legacy(predecessor, token=secret, sessions={"101": "existing"})
    settings = tg.load_settings()
    assert secret not in repr(settings)


@pytest.mark.parametrize("bad_mapping", [[], None, "not a mapping"])
def test_malformed_legacy_chat_mapping_fails_closed(predecessor, bad_mapping):
    _legacy(predecessor, sessions={"101": "existing"})
    (predecessor / "telegram_settings.json").write_text(
        json.dumps({"bot_name": "Legacy_Test_bot", "chat_sessions": bad_mapping}), encoding="utf-8"
    )
    with pytest.raises(tg.TelegramConfigError):
        tg.load_settings()
