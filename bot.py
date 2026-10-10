#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram Chatbot — Hybrid (Menu buttons + AI)  |  Updated: October 2026
-----------------------------------------------------------------------
- Menu: /start pe buttons (About, Help, AI, Contact).
- AI: koi bhi text -> Google Gemini (naya google-genai SDK, free tier).
- MODEL FALLBACK: pehla model rate-limit/error de to khud agla model try hota hai.
- HEALTH SERVER: agar PORT env set ho (jaise Render free pe) to ek chhota web
  server bhi chalta hai, taaki free host bot ko zinda rakhe (keep-alive).

Zaroori env vars (.env file me):
  BOT_TOKEN       -> BotFather se mila token
  GEMINI_API_KEY  -> https://aistudio.google.com/app/apikey se FREE key
  GEMINI_MODELS   -> (optional) comma-separated fallback chain
                     default: gemini-3.6-flash,gemini-2.5-flash
"""

import os
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import defaultdict, deque

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters,
)

from google import genai
from google.genai import types

# ----------------------------- CONFIG -----------------------------
BOT_TOKEN = os.environ.get("8603530096:AAGuuDOlX7cr3Qp7-plPbnPOAOdz7edoCG8", "").strip()
GEMINI_API_KEY = os.environ.get("AQ.Ab8RN6ltrd1ir7CjPWN69p_uQOorspVSDhmneJsonzWo8cnYcg", "").strip()

# Fallback chain: pehla model try hoga; error/429 aaye to agla.
# NOTE: gemini-3.5-flash ab DEPRECATED hai (Google khud 3.6 pe route karta hai),
# isliye default me 3.6 Flash rakha hai. Chaaho to badal sakte ho.
_models_env = os.environ.get("GEMINI_MODELS", "").strip()
GEMINI_MODELS = [m.strip() for m in _models_env.split(",") if m.strip()] or [
    "gemini-3.6-flash",
    "gemini-2.5-flash",
]

SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a helpful, friendly assistant on Telegram. Answer clearly and "
    "concisely. Reply in the same language the user writes in (Hindi, Hinglish, "
    "or English). Keep answers practical.",
)

MAX_HISTORY = 10  # per chat kitne messages yaad rakhne hain

# ----------------------------- LOGGING -----------------------------
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("chatbot")

# ----------------------------- GEMINI CLIENT -----------------------------
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
HISTORY: dict[int, deque] = defaultdict(lambda: deque(maxlen=MAX_HISTORY))


async def ask_gemini(chat_id: int, user_text: str) -> str:
    """Gemini se jawab; model fallback chain ke saath."""
    if client is None:
        return ("⚠️ GEMINI_API_KEY set nahi hai. Server ke .env me apni free key "
                "daalo (aistudio.google.com/app/apikey), phir bot restart karo.")

    history = HISTORY[chat_id]
    history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_text)]))

    last_err = ""
    for model in GEMINI_MODELS:
        try:
            resp = await client.aio.models.generate_content(
                model=model,
                contents=list(history),
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
            )
            answer = (resp.text or "").strip()
            if answer:
                history.append(types.Content(role="model",
                                             parts=[types.Part.from_text(text=answer)]))
                return answer
            last_err = "khali jawab"
        except Exception as e:  # noqa: BLE001 — 429/400/network sab yahin
            last_err = str(e)
            log.warning("Model %s fail hua, agla try: %s", model, str(e)[:150])
            continue

    # Sab models fail
    HISTORY[chat_id].pop()  # failed user turn hata do
    return (f"⚠️ Abhi AI se jawab nahi mil paya (saare models try kiye: "
            f"{', '.join(GEMINI_MODELS)}). Thodi der baad try karo.\n_Reason: {last_err[:120]}_")


# ----------------------------- HEALTH SERVER (free host keep-alive) ----------
def start_health_server() -> None:
    """PORT set ho to ek chhota HTTP server chalao (Render/UptimeRobot ke liye)."""
    port = int(os.environ.get("PORT", "0") or "0")
    if port <= 0:
        return

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def log_message(self, *args):  # shant rakho
            pass

    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log.info("Health server chalu port %s pe (/ aur /health)", port)


# ----------------------------- MENU -----------------------------
def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("ℹ️ About", callback_data="about"),
         InlineKeyboardButton("❓ Help", callback_data="help")],
        [InlineKeyboardButton("🤖 AI kaise use karein", callback_data="ai_help"),
         InlineKeyboardButton("🧹 Chat clear", callback_data="clear")],
        [InlineKeyboardButton("📞 Contact", callback_data="contact")],
    ])


def back_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Menu", callback_data="menu")]])


WELCOME = ("👋 *Namaste!* Main aapka assistant bot hoon.\n\n"
           "Menu se buttons dabao, ya seedha *kuch bhi likh kar bhejo* — "
           "main AI se jawab dunga. 🎯")
ABOUT = ("🤖 *About*\n\nYe ek hybrid bot hai: fixed menu *+* AI jawab dono.\n"
         "Telegram Bot API + Google Gemini (free) se bana hai. 24/7 cloud pe chalta hai.")
HELP = ("❓ *Help / Commands*\n\n/start – bot dobara shuru\n/help – ye madad\n"
        "/about – bot ke baare me\n/ai <sawal> – AI se jawab\n/id – apna Telegram ID\n"
        "/clear – chat memory saaf\n\n💡 Koi bhi normal message bhejo — AI jawab dega.")
AI_HELP = ("🤖 *AI kaise use karein*\n\nBas kuch bhi type karke bhejo, jaise:\n"
           "• _Delhi ka mausam kaisa hai?_\n• _Ek shayari likho_\n"
           "• _Python me loop kaise banate hain?_\n\nAI aapki bhasha me jawab dega.")
CONTACT = "📞 *Contact*\n\nSawal ya problem? Apna message yahin bhej do."


# ----------------------------- HANDLERS -----------------------------
async def cmd_start(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    await u.message.reply_text(WELCOME, parse_mode="Markdown", reply_markup=main_menu())


async def cmd_help(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    await u.message.reply_text(HELP, parse_mode="Markdown", reply_markup=back_menu())


async def cmd_about(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    await u.message.reply_text(ABOUT, parse_mode="Markdown", reply_markup=back_menu())


async def cmd_id(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    usr = u.effective_user
    await u.message.reply_text(f"🆔 Aapka ID: `{usr.id}`\nNaam: {usr.first_name}",
                               parse_mode="Markdown")


async def cmd_clear(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    HISTORY.pop(u.effective_chat.id, None)
    await u.message.reply_text("🧹 Chat memory clear ho gayi.")


async def cmd_ai(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    text = " ".join(c.args) if c.args else ""
    if not text:
        await u.message.reply_text("Istemal: `/ai aapka sawal`", parse_mode="Markdown")
        return
    await _answer(u, c, text)


async def on_text(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    await _answer(u, c, u.message.text)


async def _answer(u: Update, c: ContextTypes.DEFAULT_TYPE, text: str) -> None:
    await c.bot.send_chat_action(chat_id=u.effective_chat.id, action=ChatAction.TYPING)
    ans = await ask_gemini(u.effective_chat.id, text)
    await u.message.reply_text(ans)


async def on_button(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    q = u.callback_query
    await q.answer()
    d = q.data
    if d == "menu":
        await q.edit_message_text(WELCOME, parse_mode="Markdown", reply_markup=main_menu())
    elif d == "about":
        await q.edit_message_text(ABOUT, parse_mode="Markdown", reply_markup=back_menu())
    elif d == "help":
        await q.edit_message_text(HELP, parse_mode="Markdown", reply_markup=back_menu())
    elif d == "ai_help":
        await q.edit_message_text(AI_HELP, parse_mode="Markdown", reply_markup=back_menu())
    elif d == "contact":
        await q.edit_message_text(CONTACT, parse_mode="Markdown", reply_markup=back_menu())
    elif d == "clear":
        HISTORY.pop(q.message.chat_id, None)
        await q.edit_message_text("🧹 Chat memory clear ho gayi.", reply_markup=back_menu())


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.error("Update me error: %s", context.error, exc_info=context.error)


# ----------------------------- MAIN -----------------------------
def main() -> None:
    if not BOT_TOKEN:
        raise SystemExit("❌ BOT_TOKEN set nahi hai. .env file me daalo.")

    start_health_server()  # Render/free host ke liye (PORT set ho to)

    app: Application = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("about", cmd_about))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("ai", cmd_ai))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(on_error)

    log.info("✅ Bot chalu ho raha hai... (models: %s)", ", ".join(GEMINI_MODELS))
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
