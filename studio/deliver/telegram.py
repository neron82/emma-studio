"""Telegram transport — reuses the chatbot's bot token (Smart_Hermione_bot).

IMPORTANT: this transport and the old chatbot must never poll simultaneously
(two getUpdates consumers fight over updates). Start this only when the
chatbot app is retired.

Commands:
  /video <topic>          render a video message with the active persona
  /persona                list personas; /persona <name> switches
  /fact <text>            add a fact card for the active persona
  plain text              dialog turn -> text + voice reply
  voice note              STT -> dialog turn -> voice reply
"""
from __future__ import annotations

import asyncio

from telegram import Update
from telegram.ext import (
    Application, CommandHandler, ContextTypes, MessageHandler, filters,
)

from .. import config
from ..core.store import Store
from ..core import pipeline


class TelegramTransport:
    def __init__(self, store: Store):
        self.store = store
        # active persona per chat
        self.active: dict[int, str] = {}
        self.histories: dict[int, list[dict]] = {}

    # -- helpers -----------------------------------------------------------

    def _persona(self, chat_id: int) -> dict | None:
        pid = self.active.get(chat_id)
        if pid:
            p = self.store.get_persona(pid)
            if p:
                return p
        personas = self.store.list_personas()
        return personas[0] if personas else None

    def _authorized(self, update: Update) -> bool:
        return not config.TELEGRAM_ALLOWED_CHATS or \
            update.effective_chat.id in config.TELEGRAM_ALLOWED_CHATS

    # -- commands ------------------------------------------------------------

    async def cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._authorized(update):
            return
        await update.message.reply_text(
            "Emma Studio.\n/video <thema> — Video-Message\n/persona [name] — Persona wechseln\n"
            "/fact <text> — Fakt merken\nText oder Sprachnachricht — Dialog."
        )

    async def cmd_persona(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._authorized(update):
            return
        name = " ".join(ctx.args or []).strip().lower()
        personas = self.store.list_personas()
        if not personas:
            await update.message.reply_text("Noch keine Personas — im Web-UI anlegen.")
            return
        match = next((p for p in personas if p["name"].lower() == name), None)
        if match:
            self.active[update.effective_chat.id] = match["id"]
            await update.message.reply_text(f"Aktiv: {match['name']}")
        else:
            current = self._persona(update.effective_chat.id)
            lines = [f"{'▶' if p['id']==current.get('id') else ' '} {p['name']}" for p in personas]
            await update.message.reply_text("Personas:\n" + "\n".join(lines))

    async def cmd_fact(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._authorized(update):
            return
        fact = " ".join(ctx.args or []).strip()
        if not fact:
            await update.message.reply_text("Usage: /fact <text>")
            return
        persona = self._persona(update.effective_chat.id)
        self.store.add_fact(fact, persona_id=persona["id"] if persona else None)
        await update.message.reply_text(f"Merk ich mir: {fact}")

    # -- turns ---------------------------------------------------------------

    async def on_text(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._authorized(update) or not update.message.text:
            return
        persona = self._persona(update.effective_chat.id)
        if not persona:
            await update.message.reply_text("Keine Persona vorhanden.")
            return
        status = await update.message.reply_text("…")
        job = self.store.create_job(persona["id"], "call_turn", update.message.text)
        history = self.histories.setdefault(update.effective_chat.id, [])
        done = await pipeline.run_call_turn(self.store, job, history)
        if done["state"] == "done":
            history.append({"role": "user", "content": update.message.text})
            history.append({"role": "assistant", "content": done["script"]})
            del history[:-12]
            with open(done["speech_path"], "rb") as fh:
                await status.edit_text(done["script"])
                await update.message.reply_voice(voice=fh)
        else:
            await status.edit_text(f"Fehler: {done['error']}")

    async def on_voice(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._authorized(update) or not update.message.voice:
            return
        persona = self._persona(update.effective_chat.id)
        tg_file = await ctx.bot.get_file(update.message.voice.file_id)
        dest = config.DATA_DIR / "telegram" / f"{update.message.voice.file_unique_id}.ogg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        await tg_file.download_to_drive(dest)

        from ..speech import stt
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, stt.transcribe, dest)
        if not text:
            await update.message.reply_text("(nichts verstanden)")
            return
        status = await update.message.reply_text(f"🎙 „{text}“ …")
        job = self.store.create_job(persona["id"], "call_turn", text)
        history = self.histories.setdefault(update.effective_chat.id, [])
        done = await pipeline.run_call_turn(self.store, job, history)
        if done["state"] == "done":
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": done["script"]})
            del history[:-12]
            await status.edit_text(done["script"])
            with open(done["speech_path"], "rb") as fh:
                await update.message.reply_voice(voice=fh)
        else:
            await status.edit_text(f"Fehler: {done['error']}")

    async def cmd_video(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._authorized(update):
            return
        topic = " ".join(ctx.args or []).strip()
        persona = self._persona(update.effective_chat.id)
        if not topic or not persona:
            await update.message.reply_text("Usage: /video <thema>" if not topic else "Keine Persona.")
            return
        status = await update.message.reply_text("Render gestartet… (dauert ein paar Minuten)")
        job = self.store.create_job(persona["id"], "video_message", topic)
        done = await pipeline.run_video_message(self.store, job)
        if done["state"] == "done":
            await status.delete()
            with open(done["render_path"], "rb") as fh:
                await update.message.reply_video(video=fh, supports_streaming=True)
        else:
            await status.edit_text(f"Render fehlgeschlagen: {done['error']}")

    # -- entry -----------------------------------------------------------------

    def run(self) -> None:
        if not config.TELEGRAM_BOT_TOKEN:
            raise RuntimeError("STUDIO_TG_BOT_TOKEN not set")
        app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
        app.add_handler(CommandHandler("start", self.cmd_start))
        app.add_handler(CommandHandler("persona", self.cmd_persona))
        app.add_handler(CommandHandler("fact", self.cmd_fact))
        app.add_handler(CommandHandler("video", self.cmd_video))
        app.add_handler(MessageHandler(filters.VOICE, self.on_voice))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.on_text))
        app.run_polling(allowed_updates=["message"])
