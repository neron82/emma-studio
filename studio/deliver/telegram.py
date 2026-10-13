"""Private Telegram sessions: durable dialog followed by one complete voice file.

Run with ``python -m studio.deliver.telegram``. Never poll this bot concurrently
with its predecessor: Telegram permits only one getUpdates consumer per token.
"""
from __future__ import annotations

import asyncio
import base64
from functools import wraps
import ipaddress
import logging
from pathlib import Path
import tempfile
import secrets
from urllib.parse import urlsplit

import httpx
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from .. import config
from ..agent import dialog, memory
from ..agent.models import (
    ModelConfigurationError, ModelDiscoveryError, discover_models, resolve_endpoint, validate_selection,
)
from ..core.store import Store
from ..speech import tts
from .telegram_config import BotSettings, load_settings


logger = logging.getLogger(__name__)

# Image mime types by file suffix for the data: URL sent to the vision model.
# Telegram photos are JPEG; documents carry their own mime_type and take priority.
IMAGE_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
    ".bmp": "image/bmp", ".gif": "image/gif", ".tif": "image/tiff", ".tiff": "image/tiff",
    ".avif": "image/avif",
}
# Stored as the user turn's text when a photo carries no caption (keeps the
# transcript and fact search non-empty and readable).
IMAGE_TURN_PLACEHOLDER = "[Bild]"


class VoiceReplyError(RuntimeError):
    """Safe stage label; never carries an upstream exception message."""

    def __init__(self, stage: str):
        self.stage = stage
        super().__init__(stage)


class TransportError(RuntimeError):
    """Only locally authored, credential-free messages may cross this boundary."""


def serialized(handler):
    """Authorize before touching state, and serialize each chat's entire operation."""
    @wraps(handler)
    async def guarded(self, update, ctx):
        if not self._authorized(update):
            return
        query = getattr(update, "callback_query", None)
        if query:
            await query.answer()
        chat_id = update.effective_chat.id
        async with self._locks.setdefault(chat_id, asyncio.Lock()):
            try:
                await handler(self, update, ctx)
            except TransportError as exc:
                await self._reply(update, str(exc))
            except (ModelConfigurationError, ModelDiscoveryError):
                await self._reply(update, "Model selection failed. Check the provider configuration/credentials and try /provider or /model again. Your selection is unchanged.")
            except Exception:
                # Upstream exceptions can contain bot tokens, HTTP URLs, or response
                # bodies. Never echo them, or log them with traceback/transcript.
                await self._reply(update, "Operation failed. Check the Studio services and try again; existing session data is retained.")
    return guarded


def text_chunks(text: str, limit: int = 4000):
    """Bound Telegram text by UTF-16 units, including astral characters."""
    start = 0
    units = 0
    for index, char in enumerate(text):
        width = 2 if ord(char) > 0xFFFF else 1
        if units + width > limit:
            yield text[start:index]
            start, units = index, 0
        units += width
    if start < len(text):
        yield text[start:]


class TelegramTransport:
    def __init__(self, store: Store, settings: BotSettings | None = None):
        self.store = store
        self.settings = settings if settings is not None else load_settings()
        self._locks: dict[int, asyncio.Lock] = {}
        self._menus: dict[int, dict] = {}

    def _authorized(self, update: Update) -> bool:
        chat = update.effective_chat
        query = getattr(update, "callback_query", None)
        return bool(chat and chat.type == "private" and chat.id in self.settings.allowed_chats
                    and update.effective_message
                    and (not query or query.from_user.id == chat.id))

    async def _reply(self, update, text: str):
        for chunk in text_chunks(text):
            await update.effective_message.reply_text(chunk)

    def _key(self, chat_id: int) -> str:
        return f"{self.settings.bot_name}:{chat_id}"

    @staticmethod
    def _selection(persona: dict) -> dict:
        return {"llm_provider": persona.get("llm_provider") or "configured",
                "llm_base_url": persona.get("llm_base_url") or "",
                "llm_model": persona.get("llm_model") or ""}

    def _effective_persona(self, session: dict) -> dict:
        persona = self.store.get_persona(session["persona_id"])
        if not persona:
            raise TransportError("The session's persona no longer exists. Restore it in Studio before continuing.")
        persona = dict(persona)
        if session.get("llm_override"):
            persona.update(self._selection(session))
        return persona

    def _create(self, chat_id: int, persona: dict, selection: dict | None = None,
                *, override: bool = False) -> dict:
        self._menus.pop(chat_id, None)
        return self.store.create_session(
            persona["id"], transport="telegram", transport_key=self._key(chat_id),
            model_settings=selection if selection is not None else self._selection(persona),
            model_override=override,
        )

    def _session(self, chat_id: int) -> dict:
        session = self.store.get_transport_session("telegram", self._key(chat_id))
        if session:
            return session
        personas = self.store.list_personas()
        if not personas:
            raise TransportError("No persona available. Create one in the Studio web UI.")
        return self._create(chat_id, personas[0])

    @staticmethod
    def _provider_label(selection: dict) -> str:
        if selection["llm_provider"] == "opencode-go":
            return "OpenCode Go (opencode-go)"
        endpoint = resolve_endpoint(selection)
        parsed = urlsplit(endpoint.base_url)
        host = parsed.hostname or ""
        try:
            address = ipaddress.ip_address(host)
            local = address.is_private or address.is_loopback or address.is_link_local
        except ValueError:
            local = host == "localhost" or host.endswith((".localhost", ".local")) or "." not in host
        display_host = f"[{host}]" if ":" in host else host
        if parsed.port is not None:
            display_host += f":{parsed.port}"
        return f"Local [{display_host}]" if local else display_host

    @staticmethod
    def _model_label(selection: dict) -> str:
        # Go's model is already explicit; displaying it must not require credentials.
        if selection["llm_provider"] == "opencode-go":
            return selection["llm_model"] or "Choose a model"
        return resolve_endpoint(selection).model

    def _label(self, session: dict) -> str:
        persona = self._effective_persona(session)
        selection = self._selection(persona)
        model = self._model_label(selection)
        source = "session choice" if session.get("llm_override") else "persona settings"
        return f"Persona: {persona['name']}\nSession: {session['id']}\nProvider: {self._provider_label(selection)}\nModel: {model}\nSelection: {source}"

    def _view_key(self, session: dict) -> tuple:
        selected = self._selection(self._effective_persona(session))
        endpoint = resolve_endpoint(selected) if selected["llm_provider"] == "configured" else None
        base = endpoint.base_url if endpoint else selected["llm_base_url"]
        persona = self._selection(self.store.get_persona(session["persona_id"]))
        return (session["id"], session["persona_id"], selected["llm_provider"],
                base, endpoint.model if endpoint else "", selected["llm_model"],
                bool(session.get("llm_override")), tuple(persona.values()))

    def _ensure_view(self, update, key: tuple) -> dict:
        """A web edit may happen while discovery releases the chat operation."""
        chat_id = update.effective_chat.id
        current = self.store.get_transport_session("telegram", self._key(chat_id))
        if current is None or self._view_key(current) != key:
            self._menus.pop(chat_id, None)
            raise TransportError("This menu is out of date. Open /provider or /model again; selection unchanged.")
        return current

    async def _menu(self, update, session: dict, kind: str, rows: list,
                    *, discovered: dict | None = None, page: int = 0,
                    view_key: tuple | None = None):
        """Opaque, one-view choices keep arbitrary model IDs off the wire."""
        chat_id = update.effective_chat.id
        key = self._view_key(session) if view_key is None else view_key
        session = self._ensure_view(update, key)
        token = secrets.token_hex(8)
        choices = {}
        keyboard = []

        def button(label, action, value):
            index = str(len(choices))
            choices[index] = (action, value)
            return InlineKeyboardButton(label, callback_data=f"select:{token}:{index}")

        selected = self._selection(self._effective_persona(session))
        if kind == "provider":
            inherited = self._selection(self.store.get_persona(session["persona_id"]))
            model = self._model_label(inherited)
            label = f"Use persona settings ({self._provider_label(inherited)} / {model})"
            if not session.get("llm_override"):
                label = "✓ " + label
            keyboard.append([button(label, "inherit", "")])
            for provider in ("configured", "opencode-go"):
                target = selected if provider == selected["llm_provider"] else {
                    "llm_provider": provider, "llm_base_url": "", "llm_model": "",
                }
                label = self._provider_label(target)
                if session.get("llm_override") and provider == selected["llm_provider"]:
                    label = "✓ " + label
                keyboard.append([button(label, "provider", provider)])
            heading = "Choose a provider:"
        else:
            size = 8
            pages = max(1, (len(rows) + size - 1) // size)
            page = min(max(page, 0), pages - 1)
            current_model = self._model_label(selected)
            if selected["llm_provider"] == "configured":
                label = f"Default: {discovered['default_model']}"
                if not selected["llm_model"]:
                    label = "✓ " + label
                keyboard.append([button(label, "default", "")])
            for row in rows[page * size:(page + 1) * size]:
                label = row["name"] if row["name"] == row["id"] else f"{row['name']} — {row['id']}"
                if row["id"] == current_model:
                    label = "✓ " + label
                keyboard.append([button(label, "model", row["id"])])
            navigation = []
            if page:
                navigation.append(button("‹ Previous", "page", page - 1))
            if page + 1 < pages:
                navigation.append(button("Next ›", "page", page + 1))
            if navigation:
                keyboard.append(navigation)
            heading = f"Choose a model (page {page + 1}/{pages}):" if rows else "No models advertised by this provider."
        record = {"token": token, "key": key, "kind": kind,
                  "choices": choices, "rows": rows, "discovered": discovered}
        text = heading + "\n\n" + self._label(session)
        markup = InlineKeyboardMarkup(keyboard)
        query = getattr(update, "callback_query", None)
        self._ensure_view(update, key)
        if query:
            await query.edit_message_text(text=text, reply_markup=markup)
        else:
            await update.effective_message.reply_text(text, reply_markup=markup)
        self._menus[chat_id] = record

    async def _choose_provider(self, update, session: dict, provider: str):
        key = self._view_key(session)
        expected_persona = self.store.get_persona(session["persona_id"])
        current = self._selection(self._effective_persona(session))
        selected = current if provider == current["llm_provider"] else {
            "llm_provider": provider, "llm_base_url": "", "llm_model": "",
        }
        # Discovery and validation complete before changing the durable selection.
        discovered = await discover_models(provider, selected["llm_base_url"])
        if provider == "configured" or selected["llm_model"]:
            selected = validate_selection(selected)
        self._ensure_view(update, key)
        session = self.store.update_session_model(
            session["id"], **selected, expected_persona_selection=expected_persona,
        )
        if session is None:
            self._menus.pop(update.effective_chat.id, None)
            raise TransportError("The persona settings changed. Open /provider again; selection unchanged.")
        self._menus.pop(update.effective_chat.id, None)
        await self._menu(update, session, "model", discovered["models"], discovered=discovered)

    async def _inherit_provider(self, update, session: dict):
        key = self._view_key(session)
        expected_persona = self.store.get_persona(session["persona_id"])
        selected = self._selection(expected_persona)
        discovered = await discover_models(selected["llm_provider"], selected["llm_base_url"])
        if selected["llm_provider"] == "configured" or selected["llm_model"]:
            selected = validate_selection(selected)
        self._ensure_view(update, key)
        session = self.store.update_session_model(
            session["id"], **selected, model_override=False,
            expected_persona_selection=expected_persona,
        )
        if session is None:
            self._menus.pop(update.effective_chat.id, None)
            raise TransportError("The persona settings changed. Open /provider again; selection unchanged.")
        self._menus.pop(update.effective_chat.id, None)
        await self._menu(update, session, "model", discovered["models"], discovered=discovered)

    async def _choose_model(self, update, session: dict, choice: str, *, default: bool = False,
                            expected_base: str | None = None, expected_models: list | None = None):
        key = self._view_key(session)
        expected_persona = self.store.get_persona(session["persona_id"])
        selected = self._selection(self._effective_persona(session))
        discovered = await discover_models(selected["llm_provider"], selected["llm_base_url"])
        if expected_base is not None and discovered["base_url"] != expected_base:
            raise TransportError("This menu is out of date. Open /provider or /model again; selection unchanged.")
        if expected_models is not None and discovered["models"] != expected_models:
            raise TransportError("The provider's model list changed. Open /model again; selection unchanged.")
        if default and selected["llm_provider"] == "configured":
            selected["llm_model"] = ""
        elif choice in {row["id"] for row in discovered["models"]}:
            selected["llm_model"] = choice
        else:
            raise TransportError("That model is no longer available. Open /model again; selection unchanged.")
        selected = validate_selection(selected)
        self._ensure_view(update, key)
        session = self.store.update_session_model(
            session["id"], **selected, expected_persona_selection=expected_persona,
        )
        if session is None:
            self._menus.pop(update.effective_chat.id, None)
            raise TransportError("The persona settings changed. Open /model again; selection unchanged.")
        self._menus.pop(update.effective_chat.id, None)
        query = getattr(update, "callback_query", None)
        if query:
            await query.edit_message_text(text=self._label(session), reply_markup=None)
        else:
            await self._reply(update, self._label(session))

    @serialized
    async def on_selection(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        record = self._menus.get(update.effective_chat.id)
        parts = (query.data or "").split(":")
        if (not record or len(parts) != 3 or parts[0] != "select"
                or parts[1] != record["token"] or parts[2] not in record["choices"]):
            raise TransportError("This menu is out of date. Open /provider or /model again; selection unchanged.")
        session = self._session(update.effective_chat.id)
        if record["key"] != self._view_key(session):
            self._menus.pop(update.effective_chat.id, None)
            raise TransportError("This menu is out of date. Open /provider or /model again; selection unchanged.")
        action, value = record["choices"][parts[2]]
        if action == "provider":
            await self._choose_provider(update, session, value)
        elif action == "inherit":
            await self._inherit_provider(update, session)
        elif action == "page":
            selected = self._selection(self._effective_persona(session))
            discovered = await discover_models(selected["llm_provider"], selected["llm_base_url"])
            session = self._ensure_view(update, record["key"])
            if (discovered["base_url"] != record["discovered"]["base_url"]
                    or discovered["models"] != record["rows"]):
                self._menus.pop(update.effective_chat.id, None)
                raise TransportError("The provider's model list changed. Open /model again; selection unchanged.")
            await self._menu(update, session, "model", record["rows"], discovered=discovered,
                             page=value, view_key=record["key"])
        else:
            await self._choose_model(update, session, value, default=action == "default",
                                     expected_base=record["discovered"]["base_url"],
                                     expected_models=record["rows"])

    async def _close(self, session: dict):
        transcript = self.store.session_transcript(session["id"])
        summary = ""
        if transcript:
            try:
                summary = await asyncio.to_thread(
                    memory.summarize_call, transcript, self._effective_persona(session),
                    session_id=session["id"],
                )
                if not summary.strip():
                    raise ValueError("empty summary")
            except Exception:
                raise TransportError("Could not summarize this conversation. The current session is still active; check the selected model/provider, then retry /new or /persona. No new session was created.") from None
        if summary:
            try:
                await asyncio.to_thread(memory.append_summary, session["persona_id"], summary, session=session)
            except Exception:
                raise TransportError("The persona memory file could not be updated. The current session is still active; check data-directory permissions and retry /new or /persona. No new session was created.") from None
        finalized = self.store.finalize_session_now(session["id"])
        if finalized and summary:
            self.store.set_session_summary(session["id"], summary)

    @serialized
    async def cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        session = self._session(update.effective_chat.id)
        await self._reply(update, "Emma Studio. Text, a voice note or a photo → complete voice reply.\n/new — summarize and start fresh\n/persona [name or ID] — list/switch personas\n/provider — choose a provider\n/model — choose a model\n/fact <text> — remember a fact\n/video <topic> — render a video\n\n" + self._label(session))

    @serialized
    async def cmd_new(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        old = self._session(update.effective_chat.id)
        persona = self._effective_persona(old)
        selection = self._selection(persona)
        await self._close(old)
        new = self._create(update.effective_chat.id, persona, selection,
                           override=bool(old.get("llm_override")))
        await self._reply(update, "New conversation; previous memory is available.\n" + self._label(new))

    @serialized
    async def cmd_persona(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        session = self._session(update.effective_chat.id)
        choice = " ".join(ctx.args or []).strip()
        personas = self.store.list_personas()
        if not choice:
            lines = [f"{'Active: ' if p['id'] == session['persona_id'] else ''}{p['name']} — {p['id']}" for p in personas]
            await self._reply(update, "Personas (/persona <name or ID>):\n" + "\n".join(lines))
            return
        matches = [p for p in personas if p["id"] == choice or p["name"].casefold() == choice.casefold()]
        if len(matches) != 1:
            raise TransportError("Persona not found or name ambiguous. Use /persona to list exact IDs.")
        persona = matches[0]
        if persona["id"] != session["persona_id"]:
            await self._close(session)
            session = self._create(update.effective_chat.id, persona)
        await self._reply(update, self._label(session))

    @serialized
    async def cmd_provider(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        session = self._session(update.effective_chat.id)
        provider = " ".join(ctx.args or []).strip()
        if not provider:
            await self._menu(update, session, "provider", [])
            return
        if provider not in {"configured", "opencode-go"}:
            raise TransportError("Unknown provider. Use /provider to choose from the list.")
        await self._choose_provider(update, session, provider)

    @serialized
    async def cmd_model(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        session = self._session(update.effective_chat.id)
        selected = self._selection(self._effective_persona(session))
        choice = " ".join(ctx.args or []).strip()
        if not choice:
            key = self._view_key(session)
            discovered = await discover_models(selected["llm_provider"], selected["llm_base_url"])
            await self._menu(update, session, "model", discovered["models"], discovered=discovered,
                             view_key=key)
            return
        await self._choose_model(update, session, choice,
                                 default=choice == "default" and selected["llm_provider"] == "configured")

    @serialized
    async def cmd_fact(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        fact = " ".join(ctx.args or []).strip()
        if not fact:
            raise TransportError("Usage: /fact <text>")
        session = self._session(update.effective_chat.id)
        self.store.add_fact(fact, persona_id=session["persona_id"])
        await self._reply(update, "Fact saved.")

    async def _turn(self, update, text: str, image_data_url: str | None = None):
        session = self._session(update.effective_chat.id)
        persona = self._effective_persona(session)
        if persona["llm_provider"] == "opencode-go" and not persona["llm_model"]:
            raise TransportError("Choose an OpenCode Go model with /model before sending a message.")
        try:
            validate_selection(self._selection(persona))
        except ModelConfigurationError:
            raise TransportError("The selected model/provider is not configured. Check server credentials and use /provider or /model before continuing.") from None
        history = [{"role": m["role"], "content": m["content"]} for m in self.store.session_transcript(session["id"])]
        # A photo without a caption still needs a non-empty stored turn; the image
        # itself only rides on the current request (history stays text-only).
        stored = text or (IMAGE_TURN_PLACEHOLDER if image_data_url else text)
        facts = await asyncio.to_thread(self.store.search_facts, stored, persona_id=persona["id"])
        self.store.add_message(persona["id"], "user", stored, session_id=session["id"])
        try:
            response = await dialog.chat(persona, history, text, facts, session_id=session["id"],
                                         store=self.store, image_data_url=image_data_url)
            if not isinstance(response, str) or not response.strip():
                raise ValueError("empty reply")
        except Exception as exc:
            selected = self._selection(persona)
            endpoint = resolve_endpoint(selected)
            status = getattr(exc, "status_code", None) if isinstance(exc, dialog.AgentError) else None
            if isinstance(exc, httpx.TimeoutException):
                reason = "The provider timed out. Check its availability or choose another provider/model."
            elif isinstance(exc, httpx.RequestError):
                reason = "Connection failed. Ensure this endpoint is running and reachable, or choose another provider."
            elif isinstance(status, int):
                reason = f"The provider rejected the request (HTTP {status}). "
                if status in {401, 403}:
                    reason += "Check its server credentials."
                elif status == 429:
                    reason += "The provider's rate or usage limit was reached; check your allowance before retrying."
                else:
                    reason += "Check that this model is available and supports chat requests."
            else:
                reason = "The provider could not complete a valid reply. Check its model/service configuration."
            raise TransportError(f"{reason}\nProvider: {self._provider_label(selected)}\nModel: {endpoint.model}\nYour message is saved. Use /provider or /model to change the session selection, then try again.") from None
        self.store.add_message(persona["id"], "assistant", response, session_id=session["id"])
        try:
            await self._voice_reply(update, response, persona)
        except VoiceReplyError as exc:
            hints = {
                "preparation": "Check data-directory permissions and free disk space.",
                "synthesis": "Check the OmniVoice service and its available GPU memory.",
                "conversion": "Check ffmpeg and the generated audio format.",
                "upload": "Check Telegram connectivity and upload permissions.",
            }
            raise TransportError(
                f"Voice {exc.stage} failed. The complete model reply is saved in this session; "
                + hints[exc.stage]
            ) from None
        except Exception:
            raise TransportError("Voice generation or delivery failed. The complete model reply is saved in this session; check TTS/ffmpeg and Telegram connectivity.") from None

    async def _voice_reply(self, update, response: str, persona: dict):
        stage = "preparation"
        try:
            root = config.DATA_DIR / "telegram"
            root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="reply-", dir=root) as directory:
                wav = Path(directory) / "reply.wav"
                opus = Path(directory) / "reply.ogg"
                stage = "synthesis"
                await tts.synthesize(response, voice=persona.get("voice_profile") or config.TTS_VOICE_DEFAULT,
                                     language=persona.get("language"), out_path=wav)
                stage = "conversion"
                process = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(wav),
                    "-vn", "-c:a", "libopus", "-b:a", "64k", "-f", "ogg", str(opus),
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                )
                try:
                    code = await process.wait()
                finally:
                    if process.returncode is None:
                        process.kill()
                        await process.wait()
                if code or not opus.is_file() or not opus.stat().st_size:
                    raise RuntimeError("voice conversion failed")
                stage = "upload"
                with opus.open("rb") as audio:
                    await update.effective_message.reply_voice(voice=audio)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if not isinstance(status, int) or not 100 <= status <= 599:
                status = None
            # Tokens, transcripts, provider bodies and URLs can be present in
            # exception messages. Log only the stage, exception class and status.
            logger.warning("Telegram voice reply failed: stage=%s error=%s http_status=%s",
                           stage, type(exc).__name__, status)
            raise VoiceReplyError(stage) from None

    @serialized
    async def on_text(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        text = update.effective_message.text
        if text and text.strip():
            await self._turn(update, text)

    @serialized
    async def on_voice(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        voice = update.effective_message.voice
        if not voice:
            return
        from ..speech import stt
        root = config.DATA_DIR / "telegram"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="input-", dir=root) as directory:
            path = Path(directory) / "input.ogg"
            try:
                tg_file = await ctx.bot.get_file(voice.file_id)
                await tg_file.download_to_drive(path)
                text = await asyncio.to_thread(stt.transcribe, path)
            except Exception:
                raise TransportError("Voice transcription failed. Check the speech service or send text instead.") from None
        if not text or not text.strip():
            raise TransportError("No speech understood; please try again or send text.")
        await self._turn(update, text)

    async def _image_data_url(self, ctx, file_id: str, *, mime_hint: str | None = None,
                              name_hint: str | None = None) -> str:
        """Download a Telegram file and return it as an OpenAI ``data:`` URL.

        Telegram photos have no explicit mime type; their server-side path ends in
        ``.jpg``. Documents carry ``mime_type``/``file_name``. The vision server
        converts anything that is not JPEG/PNG itself.
        """
        tg_file = await ctx.bot.get_file(file_id)
        suffix = Path(getattr(tg_file, "file_path", "") or name_hint or "").suffix.lower()
        mime = (mime_hint if (mime_hint or "").startswith("image/") else None) \
            or IMAGE_MIME_BY_SUFFIX.get(suffix) or "image/jpeg"
        data = await tg_file.download_as_bytearray()
        if not data:
            raise TransportError("The image could not be downloaded; please send it again.")
        return f"data:{mime};base64,{base64.b64encode(bytes(data)).decode('ascii')}"

    @serialized
    async def on_photo(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        message = update.effective_message
        # Telegram sends several sizes; the last entry is the largest.
        if message is None or not message.photo:
            return
        photo = message.photo[-1]
        try:
            data_url = await self._image_data_url(ctx, photo.file_id)
        except TransportError:
            raise
        except Exception:
            raise TransportError("The image could not be downloaded; please send it again.") from None
        await self._turn(update, (message.caption or "").strip(), image_data_url=data_url)

    @serialized
    async def on_document(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        message = update.effective_message
        if message is None:
            return
        document = message.document
        # Only images sent "as file"; everything else (PDF, archives, ...) is left alone.
        if document is None or not (document.mime_type or "").startswith("image/"):
            return
        try:
            data_url = await self._image_data_url(ctx, document.file_id, mime_hint=document.mime_type,
                                                  name_hint=document.file_name)
        except TransportError:
            raise
        except Exception:
            raise TransportError("The image could not be downloaded; please send it again.") from None
        await self._turn(update, (message.caption or "").strip(), image_data_url=data_url)

    @serialized
    async def cmd_video(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        # The legacy render path remains optional, never imported for dialog/voice.
        from ..core import pipeline
        topic = " ".join(ctx.args or []).strip()
        if not topic:
            raise TransportError("Usage: /video <topic>")
        session = self._session(update.effective_chat.id)
        await self._reply(update, "Video rendering started.")
        job = self.store.create_job(session["persona_id"], "video_message", topic)
        done = await pipeline.run_video_message(self.store, job)
        if done["state"] != "done":
            raise TransportError("Video rendering failed. Check the render services.")
        with open(done["render_path"], "rb") as video:
            await update.effective_message.reply_video(video=video, supports_streaming=True)

    async def _error(self, update, context):
        # PTB's default error logger includes request URLs (and hence the token).
        logging.getLogger(__name__).warning("Telegram update failed; details suppressed for credential safety.")

    def _application(self):
        # Network/PTB logs can include token-bearing URLs even on polling errors.
        for name in ("httpx", "httpcore", "telegram"):
            logging.getLogger(name).setLevel(logging.CRITICAL)
        app = Application.builder().token(self.settings.token).concurrent_updates(False).build()
        for command in ("start", "new", "persona", "provider", "model", "fact", "video"):
            app.add_handler(CommandHandler(command, getattr(self, f"cmd_{command}")))
        app.add_handler(CallbackQueryHandler(self.on_selection))
        app.add_handler(MessageHandler(filters.VOICE, self.on_voice))
        app.add_handler(MessageHandler(filters.PHOTO, self.on_photo))
        app.add_handler(MessageHandler(filters.Document.IMAGE, self.on_document))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.on_text))
        app.add_error_handler(self._error)
        return app

    async def serve(self, stop_event: asyncio.Event, *, ready=None):
        """Own the complete PTB lifecycle inside an existing event loop."""
        app = self._application()
        initialized = False
        try:
            await app.initialize()
            initialized = True
            await app.start()
            await app.updater.start_polling(allowed_updates=["message", "callback_query"])
            if ready is not None:
                ready.set()
            await stop_event.wait()
        finally:
            if app.updater.running:
                await app.updater.stop()
            if app.running:
                await app.stop()
            if initialized:
                await app.shutdown()

    def run(self):
        self._application().run_polling(allowed_updates=["message", "callback_query"])


def main():
    store = None
    try:
        settings = load_settings()
        store = Store(config.DB_PATH)
        TelegramTransport(store, settings).run()
    except Exception:
        # Do not leak token-bearing bootstrap/network exceptions from run_polling.
        raise SystemExit("Telegram could not start. Check private-chat settings, credentials, network, and that no other instance is polling this bot.") from None
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    main()
