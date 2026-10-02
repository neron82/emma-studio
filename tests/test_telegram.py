"""Telegram session behavior with real SQLite/memory and isolated HTTP endpoints."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import wave

import httpx
import pytest

from studio import config
from studio.agent import memory
from studio.core.store import Store
from studio.deliver import telegram as transport
from studio.deliver.telegram_config import BotSettings


class Message:
    def __init__(self, text="hello"):
        self.text = text
        self.voice = None
        self.replies = []
        self.markups = []
        self.voices = []
        self.audio_paths = []
        self.fail_upload = False

    async def reply_text(self, text, *, reply_markup=None):
        self.replies.append(text)
        self.markups.append(reply_markup)

    async def reply_voice(self, *, voice):
        self.audio_paths.append(Path(voice.name))
        self.voices.append(voice.read())
        if self.fail_upload:
            raise RuntimeError("secret-token in https://api.telegram.org/botsecret-token/sendVoice")


def update(text="hello", *, chat_id=42, chat_type="private"):
    message = Message(text)
    return SimpleNamespace(effective_chat=SimpleNamespace(id=chat_id, type=chat_type),
                           effective_message=message, message=message)


def context(*args):
    return SimpleNamespace(args=list(args))


class Callback:
    def __init__(self, message, data, user_id):
        self.message = message
        self.data = data
        self.from_user = SimpleNamespace(id=user_id)
        self.answers = 0

    async def answer(self):
        self.answers += 1

    async def edit_message_text(self, *, text, reply_markup=None):
        self.message.replies.append(text)
        self.message.markups.append(reply_markup)


def callback(message, data, *, chat_id=42, chat_type="private", user_id=None):
    result = update(chat_id=chat_id, chat_type=chat_type)
    result.message = result.effective_message = message
    result.callback_query = Callback(message, data, chat_id if user_id is None else user_id)
    return result


def menu_button(message, contains):
    buttons = [button for row in message.markups[-1].inline_keyboard for button in row
               if contains in button.text]
    assert len(buttons) == 1
    return buttons[0]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://configured.invalid/v1")
    monkeypatch.setattr(config, "LLM_MODEL", "configured-model")
    monkeypatch.setenv("STUDIO_OPENCODE_GO_API_KEY", "private-provider-secret")
    monkeypatch.setenv("STUDIO_OPENCODE_GO_BASE_URL", "https://go.invalid/v1")
    store = Store(tmp_path / "sessions.db")
    p = store.create_persona("Emma", personality_prompt="Be Emma.", voice_profile="emma", language="German")
    other = store.create_persona("Other", personality_prompt="Be Other.", llm_model="other-model")
    store.search_facts = lambda *args, **kwargs: []
    requests = []
    failures = {"models": False, "summary": False}
    model_rows = [{"id": "chosen-model", "name": "Chosen"}, {"id": "other-model"}]
    summary = ("The user discussed their garden renovation. " + "Remember the planned garden work and preferences. " * 30).strip()

    def endpoint(request):
        body = json.loads(request.content) if request.content else None
        requests.append((request, body))
        if request.url.path.endswith("/models"):
            if failures["models"]:
                return httpx.Response(500, text="private-provider-secret secret-token")
            return httpx.Response(200, json={"data": model_rows})
        is_summary = body["messages"][0]["content"].startswith("You keep the long-term memory")
        if is_summary and failures["summary"]:
            return httpx.Response(503, text="private-provider-secret secret-token")
        return httpx.Response(200, json={"choices": [{"message": {"content": summary if is_summary else "Completed reply."}}]})

    mock_transport = httpx.MockTransport(endpoint)
    async_client, sync_client = httpx.AsyncClient, httpx.Client
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: async_client(transport=mock_transport, **kwargs))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: sync_client(transport=mock_transport, **kwargs))
    settings = BotSettings(token="secret-token", bot_name="Smart_Hermione_bot", allowed_chats=frozenset({42, 43}))

    def bot():
        instance = transport.TelegramTransport(store, settings)
        async def omit_audio(*args):
            pass
        instance._voice_reply = omit_audio
        return instance

    result = SimpleNamespace(store=store, bot=bot(), new_bot=bot, p=p, other=other,
                             requests=requests, failures=failures, summary=summary,
                             settings=settings, tmp_path=tmp_path, model_rows=model_rows)
    yield result
    store.close()


def completions(env):
    return [(request, body) for request, body in env.requests if body is not None
            and not body["messages"][0]["content"].startswith("You keep the long-term memory")]


def test_new_summarizes_and_injects_memory_without_old_session_history(env):
    async def scenario():
        first = update("My garden needs renovation.")
        await env.bot.on_text(first, context())
        old = env.bot._session(42)
        # Facetime messages for the same persona must never enter Telegram history.
        web = env.store.create_session(env.p["id"])
        env.store.add_message(env.p["id"], "user", "private web history", session_id=web["id"])
        await env.bot.cmd_new(update(), context())
        new = env.bot._session(42)
        assert new["id"] != old["id"]
        assert env.store.get_session(old["id"])["state"] == "finalized"
        assert env.store.get_session(old["id"])["summary"] == env.summary
        await env.bot.on_text(update("What did we plan?"), context())
        request, payload = completions(env)[-1]
        assert env.summary in payload["messages"][0]["content"]
        assert payload["messages"][1:] == [{"role": "user", "content": "What did we plan?"}]
        assert [m["content"] for m in env.store.session_transcript(new["id"])] == ["What did we plan?", "Completed reply."]
        # The persisted ID drives actual Go session headers after a provider switch.
        await env.bot.cmd_provider(update(), context("opencode-go"))
        await env.bot.cmd_model(update(), context("chosen-model"))
        await env.bot.on_text(update("Next turn"), context())
        assert completions(env)[-1][0].headers["x-opencode-session"] == new["id"]
    asyncio.run(scenario())


def test_provider_pending_model_selection_and_restart_are_session_scoped(env):
    async def scenario():
        await env.bot.cmd_provider(update(), context("opencode-go"))
        sid = env.bot._session(42)["id"]
        pending = update("Not sent before model selection")
        await env.bot.on_text(pending, context())
        assert "Choose an OpenCode Go model" in pending.message.replies[-1]
        assert env.store.session_transcript(sid) == []
        assert completions(env) == []
        await env.bot.cmd_model(update(), context("chosen-model"))
        restarted = env.new_bot()
        # A new DB connection, not an in-memory transport cache, must resume it.
        reopened = Store(env.store.db_path)
        reopened.search_facts = lambda *args, **kwargs: []
        restarted.store = reopened
        await restarted.on_text(update("Chosen model turn"), context())
        assert completions(env)[-1][1]["model"] == "chosen-model"
        assert completions(env)[-1][0].headers["x-opencode-session"] == sid
        assert restarted._session(42)["id"] == sid
        assert env.store.get_persona(env.p["id"])["llm_provider"] == "configured"
        await restarted.cmd_new(update(), context())
        summaries = [(request, body) for request, body in env.requests if body is not None
                     and body["messages"][0]["content"].startswith("You keep the long-term memory")]
        assert summaries[-1][1]["model"] == "chosen-model"
        assert summaries[-1][0].headers["x-opencode-session"] == sid
        assert restarted._session(42)["llm_model"] == "chosen-model"
        await restarted.cmd_provider(update(), context("configured"))
        await restarted.cmd_model(update(), context("default"))
        await restarted.on_text(update("Configured again"), context())
        assert completions(env)[-1][1]["model"] == "configured-model"
        reopened.close()
    asyncio.run(scenario())


def test_persona_switch_closes_old_and_isolates_history_and_memory(env):
    async def scenario():
        await env.bot.on_text(update("Garden plan"), context())
        old = env.bot._session(42)
        await env.bot.cmd_persona(update(), context(env.other["id"]))
        await env.bot.on_text(update("Different persona"), context())
        session = env.bot._session(42)
        assert session["persona_id"] == env.other["id"]
        assert env.store.get_session(old["id"])["state"] == "finalized"
        payload = completions(env)[-1][1]
        assert payload["model"] == "other-model"
        assert env.summary not in payload["messages"][0]["content"]
        assert "Be Other." in payload["messages"][0]["content"]
        assert payload["messages"][1:] == [{"role": "user", "content": "Different persona"}]
        assert env.summary in memory.load_memory(env.p["id"])
        assert memory.load_memory(env.other["id"]) == ""
    asyncio.run(scenario())


@pytest.mark.parametrize("handler,args", [("cmd_new", ()), ("cmd_persona", ("Other",))])
def test_summary_failure_leaves_original_session_active(env, handler, args):
    async def scenario():
        await env.bot.on_text(update("Keep this conversation"), context())
        old = env.bot._session(42)
        env.failures["summary"] = True
        failed = update()
        await getattr(env.bot, handler)(failed, context(*args))
        assert env.bot._session(42)["id"] == old["id"]
        assert env.store.get_session(old["id"])["state"] == "active"
        assert len(env.store.list_sessions(transport="telegram")) == 1
        assert "still active" in failed.message.replies[-1]
        assert "secret" not in " ".join(failed.message.replies)
        assert memory.load_memory(env.p["id"]) == ""
        env.failures["summary"] = False
        await getattr(env.bot, handler)(update(), context(*args))
        assert env.bot._session(42)["id"] != old["id"]
    asyncio.run(scenario())


def test_memory_write_failure_preserves_active_history_and_retry(env, monkeypatch):
    append_summary = memory.append_summary

    def fail_write(*args, **kwargs):
        raise OSError("private-provider-secret")

    async def scenario():
        await env.bot.on_text(update("Preserve my garden plan"), context())
        old = env.bot._session(42)
        transcript = env.store.session_transcript(old["id"])
        monkeypatch.setattr(memory, "append_summary", fail_write)
        failed = update()
        await env.bot.cmd_new(failed, context())
        assert env.bot._session(42)["id"] == old["id"]
        assert env.store.get_session(old["id"])["state"] == "active"
        assert env.store.session_transcript(old["id"]) == transcript
        assert len(env.store.list_sessions(transport="telegram")) == 1
        assert "still active" in failed.message.replies[-1]
        assert "secret" not in " ".join(failed.message.replies)
        assert memory.load_memory(env.p["id"]) == ""
        await env.bot.on_text(update("Continue the same conversation"), context())
        assert env.bot._session(42)["id"] == old["id"]
        assert completions(env)[-1][1]["messages"][1:3] == [
            {"role": "user", "content": "Preserve my garden plan"},
            {"role": "assistant", "content": "Completed reply."},
        ]
        monkeypatch.setattr(memory, "append_summary", append_summary)
        await env.bot.cmd_new(update(), context())
        assert env.bot._session(42)["id"] != old["id"]
        assert env.store.get_session(old["id"])["state"] == "finalized"
        assert env.summary in memory.load_memory(env.p["id"])
        await env.bot.on_text(update("Recall our plan"), context())
        assert env.summary in completions(env)[-1][1]["messages"][0]["content"]
        assert completions(env)[-1][1]["messages"][1:] == [{"role": "user", "content": "Recall our plan"}]
    asyncio.run(scenario())


def test_model_discovery_failure_and_unknown_id_preserve_selection(env):
    async def scenario():
        await env.bot.cmd_model(update(), context("chosen-model"))
        before = env.bot._session(42)
        unknown = update()
        await env.bot.cmd_model(unknown, context("not-advertised"))
        assert env.bot._session(42)["llm_model"] == before["llm_model"]
        env.failures["models"] = True
        for handler, args in [("cmd_provider", ("opencode-go",)), ("cmd_model", ("other-model",))]:
            failed = update()
            await getattr(env.bot, handler)(failed, context(*args))
            after = env.bot._session(42)
            assert (after["llm_provider"], after["llm_model"]) == (before["llm_provider"], before["llm_model"])
            assert "unchanged" in failed.message.replies[-1]
            assert "secret" not in " ".join(failed.message.replies)
    asyncio.run(scenario())


@pytest.mark.parametrize("chat_id,chat_type", [(999, "private"), (42, "group")])
def test_unauthorized_updates_are_silent_without_state_changes(env, chat_id, chat_type):
    async def scenario():
        message = update(chat_id=chat_id, chat_type=chat_type)
        for handler, args in [("on_text", ()), ("cmd_start", ()), ("cmd_new", ()),
                              ("cmd_persona", ("Other",)), ("cmd_provider", ("opencode-go",)),
                              ("cmd_model", ()), ("cmd_fact", ("private fact",)), ("on_voice", ())]:
            await getattr(env.bot, handler)(message, context(*args))
        assert message.message.replies == []
        assert message.message.voices == []
        assert env.store.list_sessions(transport="telegram") == []
        assert env.requests == []
    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [None, "upload", "convert", "tts"])
def test_complete_opus_delivery_and_cleanup_preserve_model_reply(env, monkeypatch, failure):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required for actual Opus conversion")
    env.bot._voice_reply = transport.TelegramTransport._voice_reply.__get__(env.bot)
    synthesis = []

    async def synthesize(text, *, voice, out_path, language):
        synthesis.append((text, voice, language))
        with wave.open(str(out_path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(24000)
            audio.writeframes(b"\x00\x00" * 24000)
        if failure == "tts":
            raise RuntimeError("secret-token")
        return out_path

    monkeypatch.setattr(transport.tts, "synthesize", synthesize)
    if failure == "convert":
        async def broken_conversion(*args, **kwargs):
            raise OSError("secret-token")
        monkeypatch.setattr(asyncio, "create_subprocess_exec", broken_conversion)

    async def scenario():
        message = update("Say something")
        message.message.fail_upload = failure == "upload"
        await env.bot.on_text(message, context())
        assert synthesis == [("Completed reply.", "emma", "German")]
        assert "Completed reply." not in message.message.replies
        transcript = env.store.session_transcript(env.bot._session(42)["id"])
        assert [m["content"] for m in transcript] == ["Say something", "Completed reply."]
        if failure in {None, "upload"}:
            assert message.message.voices[0].startswith(b"OggS")
            assert b"OpusHead" in message.message.voices[0]
            assert all(not path.exists() for path in message.message.audio_paths)
        if failure:
            assert "saved" in message.message.replies[-1]
            assert "secret-token" not in " ".join(message.message.replies)
        else:
            assert message.message.replies == []
        assert list((config.DATA_DIR / "telegram").iterdir()) == []
    asyncio.run(scenario())


def test_concurrent_turns_are_serialized_and_chats_remain_isolated(env, monkeypatch):
    original = transport.dialog.chat
    entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed(persona, history, text, facts, *, session_id, store):
        if text == "First":
            entered.set()
            await release.wait()
        return await original(persona, history, text, facts, session_id=session_id, store=store)

    monkeypatch.setattr(transport.dialog, "chat", delayed)
    async def scenario():
        first = asyncio.create_task(env.bot.on_text(update("First"), context()))
        await entered.wait()
        second = asyncio.create_task(env.bot.on_text(update("Second"), context()))
        independent = asyncio.create_task(env.bot.on_text(update("Different chat", chat_id=43), context()))
        await independent
        release.set()
        await asyncio.gather(first, second)
        assert [m["content"] for m in env.store.session_transcript(env.bot._session(42)["id"])] == ["First", "Completed reply.", "Second", "Completed reply."]
        second_payload = completions(env)[-1][1]
        assert second_payload["messages"][1:] == [{"role": "user", "content": "First"}, {"role": "assistant", "content": "Completed reply."}, {"role": "user", "content": "Second"}]
        assert [m["content"] for m in env.store.session_transcript(env.bot._session(43)["id"])] == ["Different chat", "Completed reply."]
    asyncio.run(scenario())


def test_model_picker_pages_long_utf8_ids_and_selects_discovered_model(env):
    env.model_rows[:] = [{"id": f"model-{index:02}-" + "\U0001f600" * 80,
                         "name": f"Name {index:02}"} for index in range(19)]

    async def scenario():
        listing = update()
        await env.bot.cmd_model(listing, context())
        message = listing.message
        first_choice = menu_button(message, "Name 00").callback_data
        assert not any("Name 18" in b.text for row in message.markups[-1].inline_keyboard for b in row)
        for page in range(2):
            next_page = callback(message, menu_button(message, "Next").callback_data)
            await env.bot.on_selection(next_page, context())
            assert next_page.callback_query.answers == 1
        button = menu_button(message, "Name 18")
        assert len(button.callback_data.encode("utf-8")) <= 64
        assert env.model_rows[-1]["id"] not in button.callback_data
        stale = callback(message, first_choice)
        await env.bot.on_selection(stale, context())
        assert not env.bot._session(42)["llm_override"]
        # Error replies do not contain markup; the still-active last page remains usable.
        chosen = callback(message, button.callback_data)
        await env.bot.on_selection(chosen, context())
        await env.bot.on_text(update("Use the last-page model"), context())
        assert completions(env)[-1][1]["model"] == env.model_rows[-1]["id"]
        assert env.store.get_persona(env.p["id"])["llm_model"] == ""
    asyncio.run(scenario())


@pytest.mark.parametrize("fail_stt", [False, True])
def test_voice_input_uses_durable_turn_and_removes_download(env, monkeypatch, fail_stt):
    from studio.speech import stt
    downloads = []

    class Download:
        async def download_to_drive(self, path):
            path.write_bytes(b"telegram input")
            downloads.append(path)

    class Bot:
        async def get_file(self, file_id):
            return Download()

    def transcribe(path):
        assert path.read_bytes() == b"telegram input"
        if fail_stt:
            raise RuntimeError("secret-token")
        return "Transcribed garden plan"

    monkeypatch.setattr(stt, "transcribe", transcribe)
    async def scenario():
        message = update("")
        message.message.voice = SimpleNamespace(file_id="voice-id")
        await env.bot.on_voice(message, SimpleNamespace(bot=Bot(), args=[]))
        assert all(not path.exists() for path in downloads)
        assert list((config.DATA_DIR / "telegram").iterdir()) == []
        if fail_stt:
            assert env.store.list_sessions(transport="telegram") == []
            assert "transcription failed" in message.message.replies[-1]
            assert "secret-token" not in " ".join(message.message.replies)
        else:
            transcript = env.store.session_transcript(env.bot._session(42)["id"])
            assert [m["content"] for m in transcript] == ["Transcribed garden plan", "Completed reply."]
            assert message.message.replies == []
    asyncio.run(scenario())


def test_tappable_provider_model_selection_persists_and_default_is_concrete(env):
    async def scenario():
        provider = update()
        await env.bot.cmd_provider(provider, context())
        assert "Provider: configured.invalid" in provider.message.replies[-1]
        assert "Model: configured-model" in provider.message.replies[-1]
        go = callback(provider.message, menu_button(provider.message, "OpenCode Go").callback_data)
        await env.bot.on_selection(go, context())
        assert "Model: Choose a model" in provider.message.replies[-1]
        model = callback(provider.message, menu_button(provider.message, "Chosen").callback_data)
        await env.bot.on_selection(model, context())
        assert go.callback_query.answers == model.callback_query.answers == 1
        selected = env.bot._session(42)
        assert selected["llm_provider"] == "opencode-go"
        assert selected["llm_model"] == "chosen-model"
        assert selected["llm_override"] == 1
        assert "Provider: OpenCode Go (opencode-go)" in provider.message.replies[-1]
        assert "Model: chosen-model" in provider.message.replies[-1]
        restarted = env.new_bot()
        await restarted.on_text(update("Use selection after restart"), context())
        assert completions(env)[-1][1]["model"] == "chosen-model"
        assert env.store.get_persona(env.p["id"])["llm_provider"] == "configured"

        await restarted.cmd_provider(provider, context())
        # The inheritance row also names the same host; choose the endpoint row.
        configured_data = next(b.callback_data for row in provider.message.markups[-1].inline_keyboard
                               for b in row if b.text == "configured.invalid")
        configured = callback(provider.message, configured_data)
        await restarted.on_selection(configured, context())
        default = callback(provider.message, menu_button(provider.message, "Default: configured-model").callback_data)
        await restarted.on_selection(default, context())
        await restarted.on_text(update("Use concrete default"), context())
        assert completions(env)[-1][1]["model"] == "configured-model"
        assert restarted._session(42)["llm_model"] == ""
        assert "Model: configured-model" in provider.message.replies[-1]
    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["replace", "new", "persona", "provider", "global_persona"])
def test_stale_model_menu_cannot_change_current_session(env, change):
    async def scenario():
        listing = update()
        await env.bot.cmd_model(listing, context())
        old = menu_button(listing.message, "Chosen").callback_data
        if change == "replace":
            await env.bot.cmd_model(update(), context())
        elif change == "new":
            await env.bot.cmd_new(update(), context())
        elif change == "persona":
            await env.bot.cmd_persona(update(), context("Other"))
        elif change == "provider":
            await env.bot.cmd_provider(update(), context("opencode-go"))
        else:
            env.store.update_persona(env.p["id"], llm_model="other-model")
        before = env.bot._session(42)
        rejected = callback(listing.message, old)
        await env.bot.on_selection(rejected, context())
        after = env.bot._session(42)
        assert (after["id"], after["llm_provider"], after["llm_model"], after["llm_override"]) == (
            before["id"], before["llm_provider"], before["llm_model"], before["llm_override"])
        assert rejected.callback_query.answers == 1
        assert completions(env) == []
    asyncio.run(scenario())


@pytest.mark.parametrize("chat_id,chat_type,user_id", [
    (43, "private", 43), (999, "private", 999), (42, "group", 42), (42, "private", 999),
])
def test_model_callbacks_reject_cross_chat_group_and_forged_actor(env, chat_id, chat_type, user_id):
    async def scenario():
        listing = update()
        await env.bot.cmd_model(listing, context())
        data = menu_button(listing.message, "Chosen").callback_data
        before = env.bot._session(42)
        request_count = len(env.requests)
        rejected = callback(listing.message, data, chat_id=chat_id, chat_type=chat_type, user_id=user_id)
        await env.bot.on_selection(rejected, context())
        assert env.bot._session(42) == before
        assert len(env.store.list_sessions(transport="telegram")) == 1
        assert len(env.requests) == request_count
        assert rejected.callback_query.answers == (1 if chat_id == 43 and chat_type == "private" else 0)
    asyncio.run(scenario())


def test_model_callbacks_revalidate_discovery_and_preserve_selection_on_failure(env):
    async def scenario():
        listing = update()
        await env.bot.cmd_model(listing, context())
        data = menu_button(listing.message, "Chosen").callback_data
        before = env.bot._session(42)
        env.failures["models"] = True
        await env.bot.on_selection(callback(listing.message, data), context())
        assert env.bot._session(42) == before
        assert "private-provider-secret" not in " ".join(listing.message.replies)
        env.failures["models"] = False
        env.model_rows[:] = [{"id": "other-model"}]
        await env.bot.on_selection(callback(listing.message, data), context())
        assert env.bot._session(42) == before
        await env.bot.cmd_model(listing, context())
        valid = menu_button(listing.message, "other-model").callback_data
        forged = valid.rsplit(":", 1)[0] + ":999"
        await env.bot.on_selection(callback(listing.message, forged), context())
        assert env.bot._session(42) == before
    asyncio.run(scenario())


def test_inherited_effective_model_and_local_status_follow_persona_until_explicit_choice(env, monkeypatch):
    monkeypatch.setattr(config, "LLM_BASE_URL", "http://127.0.0.1:8080/v1")
    async def scenario():
        initial = update()
        await env.bot.cmd_start(initial, context())
        assert "Provider: Local [127.0.0.1:8080]" in initial.message.replies[-1]
        assert "Model: configured-model" in initial.message.replies[-1]
        assert not env.bot._session(42)["llm_override"]
        env.store.update_persona(env.p["id"], llm_provider="opencode-go", llm_model="chosen-model")
        inherited = update()
        await env.bot.cmd_start(inherited, context())
        assert "Provider: OpenCode Go (opencode-go)" in inherited.message.replies[-1]
        assert "Model: chosen-model" in inherited.message.replies[-1]
        await env.bot.on_text(update("Use current persona endpoint"), context())
        assert completions(env)[-1][1]["model"] == "chosen-model"
        await env.bot.cmd_model(update(), context("other-model"))
        env.store.update_persona(env.p["id"], llm_model="chosen-model")
        await env.bot.cmd_new(update(), context())
        await env.bot.on_text(update("Preserve session override"), context())
        assert completions(env)[-1][1]["model"] == "other-model"
        picker = update()
        await env.bot.cmd_provider(picker, context())
        inherited_choice = callback(picker.message, menu_button(picker.message, "Use persona settings").callback_data)
        await env.bot.on_selection(inherited_choice, context())
        assert not env.bot._session(42)["llm_override"]
        env.store.update_persona(env.p["id"], llm_model="chosen-model")
        await env.bot.on_text(update("Resume persona inheritance"), context())
        assert completions(env)[-1][1]["model"] == "chosen-model"
        assert env.store.get_persona(env.p["id"])["llm_model"] == "chosen-model"
    asyncio.run(scenario())


@pytest.mark.parametrize("failure,expected", [
    ("connect", "Connection failed"), ("timeout", "timed out"),
    (401, "server credentials"), (429, "rate or usage limit"), (503, "HTTP 503"),
])
def test_provider_failure_reports_safe_effective_endpoint_and_retains_message(env, monkeypatch, failure, expected):
    async def fail_chat(*args, **kwargs):
        secret = "private-provider-secret https://secret-user:password@host/query?token=secret-token transcript"
        if failure == "connect":
            raise httpx.ConnectError(secret)
        if failure == "timeout":
            raise httpx.ReadTimeout(secret)
        raise transport.dialog.AgentError(secret, status_code=failure)
    monkeypatch.setattr(transport.dialog, "chat", fail_chat)

    async def scenario():
        message = update("Private conversation")
        await env.bot.on_text(message, context())
        text = message.message.replies[-1]
        assert expected in text
        assert "Provider: configured.invalid" in text
        assert "Model: configured-model" in text
        assert "private-provider-secret" not in text
        assert "secret-token" not in text
        assert "Private conversation" not in text
        assert [m["content"] for m in env.store.session_transcript(env.bot._session(42)["id"])] == ["Private conversation"]
    asyncio.run(scenario())


def test_model_callback_waits_for_inflight_turn_before_changing_selection(env, monkeypatch):
    original = transport.dialog.chat
    entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(transport.dialog, "chat", delayed)

    async def scenario():
        picker = update()
        await env.bot.cmd_model(picker, context())
        data = menu_button(picker.message, "Chosen").callback_data
        turn = asyncio.create_task(env.bot.on_text(update("Before selection"), context()))
        await entered.wait()
        selection = callback(picker.message, data)
        choosing = asyncio.create_task(env.bot.on_selection(selection, context()))
        await asyncio.sleep(0)
        assert selection.callback_query.answers == 1
        assert not env.bot._session(42)["llm_override"]
        release.set()
        await asyncio.gather(turn, choosing)
        assert completions(env)[-1][1]["model"] == "configured-model"
        assert env.bot._session(42)["llm_model"] == "chosen-model"
        await env.bot.on_text(update("After selection"), context())
        assert completions(env)[-1][1]["model"] == "chosen-model"
    asyncio.run(scenario())


def test_provider_picker_recovers_from_missing_inherited_go_credentials(env, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(env.tmp_path / "missing-hermes"))
    monkeypatch.setenv("STUDIO_OPENCODE_GO_API_KEY", "")
    env.store.update_persona(env.p["id"], llm_provider="opencode-go", llm_model="chosen-model")

    async def scenario():
        picker = update()
        await env.bot.cmd_provider(picker, context())
        assert "OpenCode Go" in picker.message.replies[-1]
        chosen = menu_button(picker.message, "configured.invalid")
        await env.bot.on_selection(callback(picker.message, chosen.callback_data), context())
        await env.bot.on_selection(
            callback(picker.message, menu_button(picker.message, "Chosen").callback_data),
            context(),
        )
        await env.bot.on_text(update("Use the recovered endpoint"), context())
        assert completions(env)[-1][0].url.host == "configured.invalid"
        assert env.bot._session(42)["llm_override"] == 1
        assert env.store.get_persona(env.p["id"])["llm_provider"] == "opencode-go"

    asyncio.run(scenario())


@pytest.mark.parametrize("action", ["model", "provider", "inherit"])
def test_web_persona_edit_during_callback_discovery_preserves_session_selection(env, monkeypatch, action):
    async def scenario():
        picker = update()
        if action == "model":
            await env.bot.cmd_model(picker, context())
            chosen = menu_button(picker.message, "Chosen")
        else:
            if action == "inherit":
                session = env.bot._session(42)
                env.store.update_session_model(
                    session["id"], llm_provider="opencode-go",
                    llm_base_url="", llm_model="chosen-model",
                )
            await env.bot.cmd_provider(picker, context())
            if action == "inherit":
                chosen = menu_button(picker.message, "Use persona settings")
            else:
                chosen = menu_button(picker.message, "OpenCode Go")
        before = env.bot._session(42)
        entered, release = asyncio.Event(), asyncio.Event()
        original = transport.discover_models

        async def delayed(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)

        monkeypatch.setattr(transport, "discover_models", delayed)
        pressed = callback(picker.message, chosen.callback_data)
        selecting = asyncio.create_task(env.bot.on_selection(pressed, context()))
        await entered.wait()
        env.store.update_persona(env.p["id"], llm_model="other-model")
        release.set()
        await selecting
        assert env.bot._session(42) == before
        assert "out of date" in picker.message.replies[-1]
        assert pressed.callback_query.answers == 1
        assert completions(env) == []

    asyncio.run(scenario())


def test_new_session_memory_search_is_private_and_shared(env, monkeypatch):
    previous = env.bot._session(42)
    env.store.add_message(env.p['id'], 'user', 'Remember my dog Juniper.', session_id=previous['id'])
    asyncio.run(env.bot.cmd_new(update(), context()))
    assert env.store.get_session(previous['id'])['state'] == 'finalized'
    assert env.store.search_memory(env.p['id'], 'Juniper', session_id=previous['memory_id'])
    responses = iter([
        'Let me look. {"tool":"memory_search","arguments":{"query":"Juniper"}}',
        'Your dog is named Juniper.',
    ])
    # The env fixture already wraps AsyncClient. Use a dedicated mock for this turn.
    def reply(request):
        body = json.loads(request.content)
        if len(body['messages']) > 2:
            assert any('Juniper' in m['content'] for m in body['messages'] if m['content'].startswith('<memory_tool_result>'))
        return httpx.Response(200, json={'choices': [{'message': {'content': next(responses)}}]})
    # Recover the actual class since the fixture replaces the module attribute.
    from httpx._client import AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: AsyncClient(transport=httpx.MockTransport(reply), **kwargs))
    delivered = []
    async def capture_voice(update, response, persona):
        delivered.append(response)
    env.bot._voice_reply = capture_voice
    message = update('What is my dog called?')
    asyncio.run(env.bot.on_text(message, context()))
    assert message.message.replies == []
    assert delivered == ['Your dog is named Juniper.']
    transcript = env.store.session_transcript(env.bot._session(42)['id'])
    assert [m['content'] for m in transcript] == ['What is my dog called?', 'Your dog is named Juniper.']
