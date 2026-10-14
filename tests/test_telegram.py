"""Telegram session behavior with real SQLite/memory and isolated HTTP endpoints."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace
import wave

import httpx
import pytest

from studio import config
from studio.agent import memory
from studio.core.store import Store
# Import SDK-dependent speech modules before fixtures replace HTTP client classes.
from studio.speech import stt
from studio.deliver import telegram as transport
from studio.deliver.telegram_config import BotSettings


class Message:
    def __init__(self, text="hello"):
        self.text = text
        self.voice = None
        self.photo = None
        self.caption = None
        self.document = None
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


def provider_button(message, contains):
    """A provider row of the merged /model menu (never the inheritance row)."""
    buttons = [button for row in message.markups[-1].inline_keyboard for button in row
               if contains in button.text and "Use persona settings" not in button.text]
    assert len(buttons) == 1, [b.text for row in message.markups[-1].inline_keyboard for b in row]
    return buttons[0]


async def open_model_menu(bot, message, provider=None):
    """/model is one entry point now: provider list first, then that provider's models.

    `provider=None` keeps the current selection ("Use persona settings"); otherwise the
    named provider row is clicked. Either way the caller ends up on the model list.
    """
    await bot.cmd_model(message, context())
    if provider is None:
        button = menu_button(message.message, "Use persona settings")
    else:
        button = provider_button(message.message, provider)
    await bot.on_selection(callback(message.message, button.callback_data), context())
    return message


def settle(bot, timeout=5):
    """Wait for the background summaries a switch must never block on."""
    for thread in list(bot._summaries):
        thread.join(timeout=timeout)
    return [thread for thread in bot._summaries if thread.is_alive()]


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
        settle(env.bot)                         # /new answers before the summary exists
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
        await open_model_menu(env.bot, update(), "OpenCode Go")
        await env.bot.cmd_model(update(), context("chosen-model"))
        await env.bot.on_text(update("Next turn"), context())
        assert completions(env)[-1][0].headers["x-opencode-session"] == new["id"]
    asyncio.run(scenario())


def test_provider_pending_model_selection_and_restart_are_session_scoped(env):
    async def scenario():
        await open_model_menu(env.bot, update(), "OpenCode Go")
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
        settle(restarted)
        summaries = [(request, body) for request, body in env.requests if body is not None
                     and body["messages"][0]["content"].startswith("You keep the long-term memory")]
        assert summaries[-1][1]["model"] == "chosen-model"
        assert summaries[-1][0].headers["x-opencode-session"] == sid
        assert restarted._session(42)["llm_model"] == "chosen-model"
        await open_model_menu(restarted, update(), "configured.invalid")
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
        settle(env.bot)
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
def test_failed_summary_does_not_block_or_undo_the_switch(env, handler, args):
    """A summary that fails is logged in the background, never a hung menu."""
    async def scenario():
        await env.bot.on_text(update("Keep this conversation"), context())
        old = env.bot._session(42)
        env.failures["summary"] = True
        switched = update()
        await getattr(env.bot, handler)(switched, context(*args))
        settle(env.bot)
        # The switch is applied and reported even though its summary failed ...
        assert env.bot._session(42)["id"] != old["id"]
        assert env.store.get_session(old["id"])["state"] == "finalized"
        assert transport.SUMMARY_IN_BACKGROUND in " ".join(switched.message.replies)
        assert "secret" not in " ".join(switched.message.replies)
        # ... and no half-written memory entry is left behind.
        assert memory.load_memory(env.p["id"]) == ""
        assert not (env.store.get_session(old["id"]).get("summary") or "")
        env.failures["summary"] = False
        await getattr(env.bot, handler)(switched, context(*args))
        assert env.bot._session(42)["id"] != old["id"]
    asyncio.run(scenario())


def test_memory_write_failure_keeps_the_transcript_and_no_dead_thread(env, monkeypatch):
    """A failed memory write is logged; the switch stands and no thread dies."""
    append_summary = memory.append_summary

    def fail_write(*args, **kwargs):
        raise OSError("private-provider-secret")

    died = []
    monkeypatch.setattr(threading, "excepthook", lambda args: died.append(args))

    async def scenario():
        await env.bot.on_text(update("Preserve my garden plan"), context())
        old = env.bot._session(42)
        transcript = env.store.session_transcript(old["id"])
        monkeypatch.setattr(memory, "append_summary", fail_write)
        switched = update()
        await env.bot.cmd_new(switched, context())
        settle(env.bot)
        assert died == []                                   # no traceback with provider text
        assert env.bot._session(42)["id"] != old["id"]       # the fresh session stands
        assert env.store.get_session(old["id"])["state"] == "finalized"
        assert env.store.session_transcript(old["id"]) == transcript    # raw history kept
        assert env.store.get_session(old["id"])["summary"] == env.summary
        assert memory.load_memory(env.p["id"]) == ""
        # A failed summary is not retried later; the transcript stays in its session.
        await env.bot.cmd_new(update(), context())
        settle(env.bot)
        assert memory.load_memory(env.p["id"]) == ""
        monkeypatch.setattr(memory, "append_summary", append_summary)
    asyncio.run(scenario())


def test_model_discovery_failure_and_unknown_id_preserve_selection(env):
    async def scenario():
        await env.bot.cmd_model(update(), context("chosen-model"))
        before = env.bot._session(42)
        unknown = update()
        await env.bot.cmd_model(unknown, context("not-advertised"))
        assert env.bot._session(42)["llm_model"] == before["llm_model"]
        env.failures["models"] = True
        # A failure while the provider list must poll for its models ...
        failed = update()
        await open_model_menu(env.bot, failed, "OpenCode Go")
        after = env.bot._session(42)
        assert (after["llm_provider"], after["llm_model"]) == (before["llm_provider"], before["llm_model"])
        assert "unchanged" in failed.message.replies[-1]
        assert "secret" not in " ".join(failed.message.replies)
        # ... and a failure while a model id is picked directly.
        failed = update()
        await env.bot.cmd_model(failed, context("other-model"))
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
                              ("cmd_persona", ("Other",)),
                              ("cmd_model", ()), ("cmd_fact", ("private fact",)), ("on_voice", ())]:
            await getattr(env.bot, handler)(message, context(*args))
        assert message.message.replies == []
        assert message.message.voices == []
        assert env.store.list_sessions(transport="telegram") == []
        assert env.requests == []
    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [None, "upload", "convert", "tts"])
def test_complete_opus_delivery_and_cleanup_preserve_model_reply(env, monkeypatch, failure, caplog):
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
            raise transport.tts.TTSError("secret-token", status_code=500)
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
            stage = {"tts": "synthesis", "convert": "conversion", "upload": "upload"}[failure]
            assert f"Voice {stage} failed" in message.message.replies[-1]
            assert f"stage={stage}" in caplog.text
            if failure == "tts":
                assert "http_status=500" in caplog.text
            assert "secret-token" not in caplog.text
            assert "Completed reply." not in caplog.text
            assert "secret-token" not in " ".join(message.message.replies)
        else:
            assert message.message.replies == []
        assert list((config.DATA_DIR / "telegram").iterdir()) == []
    asyncio.run(scenario())


def test_concurrent_turns_are_serialized_and_chats_remain_isolated(env, monkeypatch):
    original = transport.dialog.chat
    entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed(persona, history, text, facts, *, session_id, store, image_data_url=None):
        if text == "First":
            entered.set()
            await release.wait()
        return await original(persona, history, text, facts, session_id=session_id, store=store,
                              image_data_url=image_data_url)

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
        await open_model_menu(env.bot, listing)
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
        await env.bot.cmd_model(provider, context())
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

        await restarted.cmd_model(provider, context())
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
        await open_model_menu(env.bot, listing)
        old = menu_button(listing.message, "Chosen").callback_data
        if change == "replace":
            await env.bot.cmd_model(update(), context())
        elif change == "new":
            await env.bot.cmd_new(update(), context())
        elif change == "persona":
            await env.bot.cmd_persona(update(), context("Other"))
        elif change == "provider":
            await open_model_menu(env.bot, update(), "OpenCode Go")
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
    settle(env.bot)


@pytest.mark.parametrize("chat_id,chat_type,user_id", [
    (43, "private", 43), (999, "private", 999), (42, "group", 42), (42, "private", 999),
])
def test_model_callbacks_reject_cross_chat_group_and_forged_actor(env, chat_id, chat_type, user_id):
    async def scenario():
        listing = update()
        await open_model_menu(env.bot, listing)
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
        await open_model_menu(env.bot, listing)
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
        await open_model_menu(env.bot, listing)
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
        await env.bot.cmd_model(picker, context())
        inherited_choice = callback(picker.message, menu_button(picker.message, "Use persona settings").callback_data)
        await env.bot.on_selection(inherited_choice, context())
        assert not env.bot._session(42)["llm_override"]
        env.store.update_persona(env.p["id"], llm_model="chosen-model")
        await env.bot.on_text(update("Resume persona inheritance"), context())
        assert completions(env)[-1][1]["model"] == "chosen-model"
        assert env.store.get_persona(env.p["id"])["llm_model"] == "chosen-model"
    asyncio.run(scenario())
    settle(env.bot)


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
        await open_model_menu(env.bot, picker)
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
        await env.bot.cmd_model(picker, context())
        assert "OpenCode Go" in picker.message.replies[-1]
        chosen = provider_button(picker.message, "configured.invalid")
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
            await open_model_menu(env.bot, picker)
            chosen = menu_button(picker.message, "Chosen")
        else:
            if action == "inherit":
                session = env.bot._session(42)
                env.store.update_session_model(
                    session["id"], llm_provider="opencode-go",
                    llm_base_url="", llm_model="chosen-model",
                )
            await env.bot.cmd_model(picker, context())
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
    settle(env.bot)
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


# --- photos / images sent to the vision model ---------------------------------

class PhotoSize:
    def __init__(self, file_id):
        self.file_id = file_id


class Document:
    def __init__(self, file_id, mime_type, file_name="file"):
        self.file_id = file_id
        self.mime_type = mime_type
        self.file_name = file_name


class TgFile:
    def __init__(self, data, path="photos/file_1.jpg"):
        self.file_path = path
        self._data = data

    async def download_as_bytearray(self):
        return bytearray(self._data)


class Bot:
    def __init__(self, file):
        self._file = file
        self.requested = []

    async def get_file(self, file_id):
        self.requested.append(file_id)
        return self._file


def image_context(bot, *args):
    return SimpleNamespace(args=list(args), bot=bot)


def photo_message(caption=None, file_ids=("small", "large")):
    message = update(caption or "")
    message.message.photo = [PhotoSize(fid) for fid in file_ids]
    message.message.caption = caption
    return message


JPEG = b"\xff\xd8\xff\xe0" + b"jpeg-body-bytes" * 4


def test_photo_is_downloaded_and_sent_as_an_image_part(env):
    async def scenario():
        bot = Bot(TgFile(JPEG, "photos/file_9.jpg"))
        message = photo_message("Was ist das?")
        await env.bot.on_photo(message, image_context(bot))
        # The largest size is the one downloaded.
        assert bot.requested == ["large"]
        request, payload = completions(env)[-1]
        content = payload["messages"][-1]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "Was ist das?"}
        image = [p for p in content if p["type"] == "image_url"]
        assert len(image) == 1
        url = image[0]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")
    asyncio.run(scenario())


def test_photo_without_caption_stores_a_readable_placeholder(env):
    async def scenario():
        bot = Bot(TgFile(JPEG))
        message = photo_message(None)
        await env.bot.on_photo(message, image_context(bot))
        request, payload = completions(env)[-1]
        # No text part when there is no caption; the image still rides along.
        content = payload["messages"][-1]["content"]
        assert [p["type"] for p in content] == ["image_url"]
        transcript = env.store.session_transcript(env.bot._session(42)["id"])
        assert transcript[0]["content"] == "[Bild]"
    asyncio.run(scenario())


def test_image_document_is_accepted_and_non_image_ignored(env):
    async def scenario():
        bot = Bot(TgFile(JPEG, "documents/pic.png"))
        doc = Document("doc-1", "image/png", "pic.png")
        message = update("")
        message.message.document = doc
        message.message.caption = "Ein PNG"
        await env.bot.on_document(message, image_context(bot))
        request, payload = completions(env)[-1]
        url = payload["messages"][-1]["content"][-1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        # A non-image document is left alone (no request, no reply).
        before = len(completions(env))
        ignored = update("")
        ignored.message.document = Document("doc-2", "application/pdf", "report.pdf")
        await env.bot.on_document(ignored, image_context(bot))
        assert len(completions(env)) == before
        assert ignored.message.replies == []
    asyncio.run(scenario())


def test_photo_download_failure_reports_and_replies(env):
    async def scenario():
        class FailingBot:
            async def get_file(self, file_id):
                raise RuntimeError("secret-token in https://api.telegram.org/botsecret-token/getFile")
        message = photo_message("Bild")
        await env.bot.on_photo(message, image_context(FailingBot()))
        assert "could not be downloaded" in message.message.replies[-1]
        assert "secret-token" not in message.message.replies[-1]
        assert completions(env) == []
    asyncio.run(scenario())


# --- persona menu, session-per-character, merged provider/model menu ----------

def test_persona_command_opens_a_clickable_list_and_clicking_switches(env):
    async def scenario():
        listing = update()
        await env.bot.cmd_persona(listing, context())
        texts = [b.text for row in listing.message.markups[-1].inline_keyboard for b in row]
        assert any("Emma" in t for t in texts) and "Other" in texts   # names, no memorizing
        assert "✓ Emma" in texts                                      # the active one
        assert "Cancel" in texts
        await env.bot.on_selection(
            callback(listing.message, menu_button(listing.message, "Other").callback_data), context())
        assert env.bot._session(42)["persona_id"] == env.other["id"]
        assert "Persona: Other" in listing.message.replies[-1]
    asyncio.run(scenario())


def test_persona_flip_on_an_empty_session_reuses_one_session(env):
    """The core requirement: flipping characters with nothing said must not churn."""
    async def scenario():
        session = env.bot._session(42)
        for target in (env.other["name"], env.p["name"], env.other["name"]):
            await env.bot.cmd_persona(update(), context(target))
        sessions = env.store.list_sessions(transport="telegram")
        assert len(sessions) == 1                            # one row, re-pointed
        assert sessions[0]["id"] == session["id"]
        assert sessions[0]["state"] == "active"
        assert sessions[0]["persona_id"] == env.other["id"]
        assert sessions[0]["summary"] is None                # no empty summaries
        assert env.store.session_transcript(session["id"]) == []
        assert completions(env) == []                        # nothing was summarized
    asyncio.run(scenario())


def test_persona_switch_after_a_turn_opens_one_new_session_and_reuses_it(env):
    async def scenario():
        await env.bot.on_text(update("Garden plan"), context())
        first = env.bot._session(42)
        await env.bot.cmd_persona(update(), context(env.other["name"]))
        settle(env.bot)
        second = env.bot._session(42)
        assert second["id"] != first["id"]                   # a real switch starts fresh
        assert second["persona_id"] == env.other["id"]
        assert env.store.get_session(first["id"])["state"] == "finalized"
        assert env.store.get_session(first["id"])["summary"] == env.summary
        # The fresh session is empty, so the next flip reuses it instead of churning.
        await env.bot.cmd_persona(update(), context(env.p["name"]))
        assert env.bot._session(42)["id"] == second["id"]
        assert len(env.store.list_sessions(transport="telegram")) == 2
    asyncio.run(scenario())


def test_persona_switch_answers_before_the_summary_finishes(env, monkeypatch):
    """The reported bug: the menu must never wait for a multi-minute summary."""
    entered, release = threading.Event(), threading.Event()
    original = memory.summarize_call

    def slow(transcript, persona=None, *, session_id=None):
        entered.set()
        release.wait(timeout=10)
        return original(transcript, persona, session_id=session_id)

    async def scenario():
        await env.bot.on_text(update("Garden plan"), context())
        old = env.bot._session(42)
        monkeypatch.setattr(memory, "summarize_call", slow)
        picker = update()
        await env.bot.cmd_persona(picker, context())
        await env.bot.on_selection(
            callback(picker.message, menu_button(picker.message, "Other").callback_data), context())
        # Answered while the retired session is still being summarized ...
        assert entered.wait(timeout=5)
        assert not release.is_set()
        answer = picker.message.replies[-1]
        assert "Persona is now Other." in answer
        assert transport.SUMMARY_IN_BACKGROUND in answer
        assert env.store.get_session(old["id"])["state"] == "finalized"
        # ... and the new session takes turns right away, on the new persona.
        await env.bot.on_text(update("Different persona"), context())
        assert env.bot._session(42)["persona_id"] == env.other["id"]
        assert completions(env)[-1][1]["model"] == "other-model"
        release.set()
        settle(env.bot)
        assert env.summary in memory.load_memory(env.p["id"])
        assert env.store.get_session(old["id"])["summary"] == env.summary
    asyncio.run(scenario())
    release.set()


def test_cancel_tapped_after_a_switch_reports_the_applied_change(env):
    """A Cancel racing its own switch must not claim nothing was selected."""
    async def scenario():
        picker = update()
        await env.bot.cmd_persona(picker, context())
        choose = menu_button(picker.message, "Other").callback_data
        cancel = menu_button(picker.message, "Cancel").callback_data
        await env.bot.on_selection(callback(picker.message, choose), context())
        stale = callback(picker.message, cancel)
        await env.bot.on_selection(stale, context())
        answer = picker.message.replies[-1]
        assert "already applied" in answer
        assert "Persona: Other" in answer
        assert "unchanged" not in answer
        assert stale.callback_query.answers == 1
    asyncio.run(scenario())


def test_model_command_starts_with_the_provider_list_then_polls_models(env):
    async def scenario():
        picker = update()
        await env.bot.cmd_model(picker, context())
        assert "Choose a provider:" in picker.message.replies[-1]
        texts = [b.text for row in picker.message.markups[-1].inline_keyboard for b in row]
        assert any("OpenCode Go" in t for t in texts)
        assert "Cancel" in texts
        assert not any("Chosen" in t for t in texts)         # models only after a provider
        chosen = provider_button(picker.message, "OpenCode Go")
        await env.bot.on_selection(callback(picker.message, chosen.callback_data), context())
        assert "Choose a model" in picker.message.replies[-1]
        models = [b.text for row in picker.message.markups[-1].inline_keyboard for b in row]
        assert any("Chosen" in t for t in models)
    asyncio.run(scenario())


def test_model_menu_cancel_and_back_leave_the_selection_unchanged(env):
    async def scenario():
        picker = update()
        await env.bot.cmd_model(picker, context())
        before = env.bot._session(42)
        await env.bot.on_selection(
            callback(picker.message, menu_button(picker.message, "Cancel").callback_data), context())
        assert env.bot._session(42) == before
        assert "Cancelled" in picker.message.replies[-1]
        assert completions(env) == []

        # Provider -> models -> back to the provider list.
        await open_model_menu(env.bot, picker)
        await env.bot.on_selection(
            callback(picker.message, menu_button(picker.message, "‹ Providers").callback_data), context())
        assert "Choose a provider:" in picker.message.replies[-1]
        assert env.bot._session(42) == before
    asyncio.run(scenario())


def test_provider_command_is_merged_away_and_help_names_model(env, monkeypatch):
    # Building the Application creates a real PTB HTTP client; undo the fixture's
    # httpx stub for that one construction (it needs to pass its own transport).
    from httpx._client import AsyncClient as RealAsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", RealAsyncClient)
    app = env.bot._application()
    commands = {c for handler in app.handlers[0]
                if isinstance(handler, transport.CommandHandler)
                for c in (handler.commands or ())}
    assert {"model", "persona"} <= commands
    assert "provider" not in commands

    async def scenario():
        start = update()
        await env.bot.cmd_start(start, context())
        help_text = start.message.replies[-1]
        assert "/provider" not in help_text
        assert "/model" in help_text and "/persona" in help_text
    asyncio.run(scenario())
