#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram Chatbot — Hybrid (Menu + AI) + Group Welcome
------------------------------------------------------
- Menu: /start pe buttons.
- AI: koi bhi text -> Google Gemini (google-genai SDK, model fallback).
       AI ke Markdown (**bold**, *italic*, `code`) ko asli Telegram formatting
       (HTML) me convert karke bhejta hai — isliye star-star dikhta nahi.
- WELCOME: naya member group me join kare to image + fancy caption.
- HEALTH SERVER: PORT set ho to (Render).

Env: BOT_TOKEN, GEMINI_API_KEY, GEMINI_MODELS (optional)
"""

import os
import re
import logging
import threading
import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import defaultdict, deque

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, ChatMemberHandler, ContextTypes, filters,
)

from google import genai
from google.genai import types

# ----------------------------- CONFIG -----------------------------
BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

_models_env = os.environ.get("GEMINI_MODELS", "").strip()
GEMINI_MODELS = [m.strip() for m in _models_env.split(",") if m.strip()] or [
    "gemini-3.6-flash",
    "gemini-2.5-flash",
]

SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a helpful, friendly assistant on Telegram. Answer clearly and "
    "concisely. Reply in the same language the user writes in (Hindi, Hinglish, "
    "or English). Keep answers practical. Use simple formatting: bold with "
    "**double asterisks** only, and bullet lists with a leading dash.",
)

MAX_HISTORY = 10
WELCOME_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "welcome.png")

WELCOME_CAPTION = (
    "𝗪𝗲𝗹𝗰𝗼𝗺𝗲 𝘁𝗼 𝗟𝗶𝗻𝗻𝗲𝘁 𝗔𝗜 𝗚𝗿𝗼𝘂𝗽, {name}.\n"
    "\n"
    "𝗛𝗲𝗿𝗲 𝗶𝘀 𝘆𝗼𝘂𝗿 𝗲𝗻𝘁𝗿𝘆 𝗽𝗮𝘀𝘀:\n"
    "\n"
    "𝗡𝗮𝗺𝗲: {name}\n"
    "𝗜𝗗: {id}\n"
    "𝗝𝗼𝗶𝗻𝗲𝗱: {date}\n"
    "\n"
    "𝗠𝗮𝗸𝗲 𝘆𝗼𝘂𝗿𝘀𝗲𝗹𝗳 𝗮𝘁 𝗵𝗼𝗺𝗲. 𝗜𝗳 𝘆𝗼𝘂 𝗵𝗮𝘃𝗲 𝗮𝗻𝘆 𝗾𝘂𝗲𝘀𝘁𝗶𝗼𝗻𝘀, "
    "𝗷𝘂𝘀𝘁 𝗱𝗿𝗼𝗽 𝗮 𝗺𝗲𝘀𝘀𝗮𝗴𝗲. 𝗦𝗲𝗲 𝘆𝗼𝘂 𝗶𝗻 𝘁𝗵𝗲 𝗰𝗵𝗮𝘁."
)

# ----------------------------- LOGGING -----------------------------
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("chatbot")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# History per (chat_id, user_id) — group me har user ka apna context rahe.
HISTORY: dict = defaultdict(lambda: deque(maxlen=MAX_HISTORY))


# ----------------------------- MARKDOWN -> TELEGRAM HTML -----------------------------
def md_to_html(text: str) -> str:
    """AI ke Markdown ko Telegram-safe HTML me badalta hai (bold/italic/code/list)."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # fenced code
    text = re.sub(r"```[a-zA-Z0-9_+\-]*\n?(.*?)```",
                  lambda m: "<pre>" + m.group(1).rstrip("\n") + "</pre>", text, flags=re.S)
    # inline code
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)
    # bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.S)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text, flags=re.S)
    # italic (single * / _)
    text = re.sub(r"(?<!\*)\*(?!\*)(?!\s)([^*\n]+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_(?!\s)([^_\n]+?)(?<!\s)_(?!\w)", r"<i>\1</i>", text)
    # headings
    text = re.sub(r"(?m)^#{1,6}\s*(.+?)\s*$", r"<b>\1</b>", text)
    # bullets (* or -) -> dot
    text = re.sub(r"(?m)^\s*[*-]\s+", "\u2022 ", text)
    return text


# ----------------------------- GEMINI -----------------------------
async def ask_gemini(key, user_text: str) -> str:
    if client is None:
        return ("⚠️ GEMINI_API_KEY set nahi hai. .env me apni free key daalo "
                "(aistudio.google.com/app/apikey).")

    history = HISTORY[key]
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
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            log.warning("Model %s fail, agla try: %s", model, str(e)[:150])
            continue

    history.pop()
    return (f"⚠️ Abhi AI se jawab nahi mil paya (models: {', '.join(GEMINI_MODELS)}). "
            f"Thodi der baad try karo.\n_Reason: {last_err[:120]}_")


# ----------------------------- HEALTH SERVER -----------------------------
def start_health_server() -> None:
    port = int(os.environ.get("PORT", "0") or "0")
    if port <= 0:
        return

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def log_message(self, *args):
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
    HISTORY.pop((u.effective_chat.id, u.effective_user.id), None)
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
    key = (u.effective_chat.id, u.effective_user.id)
    await c.bot.send_chat_action(chat_id=u.effective_chat.id, action=ChatAction.TYPING)
    ans = await ask_gemini(key, text)
    # AI ke Markdown ko asli formatting (HTML) me bhejo; fail ho to plain bhej do.
    try:
        await u.message.reply_text(md_to_html(ans), parse_mode="HTML")
    except Exception as e:
        log.warning("HTML reply fail (%s), plain bhej raha hoon", e)
        try:
            await u.message.reply_text(ans)
        except Exception as e2:
            log.warning("Reply fail: %s", e2)


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
        HISTORY.pop((q.message.chat_id, q.from_user.id), None)
        await q.edit_message_text("🧹 Chat memory clear ho gayi.", reply_markup=back_menu())


async def on_new_member(u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
    """Sirf tab jab koi NAYA member group me join kare."""
    cm = u.chat_member
    if cm is None:
        return
    old, new = cm.old_chat_member.status, cm.new_chat_member.status
    joined = old in ("left", "kicked") and new in ("member", "administrator",
                                                   "creator", "restricted")
    if not joined:
        return

    user = cm.new_chat_member.user
    caption = WELCOME_CAPTION.format(
        name=user.full_name, id=user.id,
        date=datetime.datetime.now().strftime("%d %b %Y"),
    )
    try:
        if os.path.exists(WELCOME_IMAGE):
            with open(WELCOME_IMAGE, "rb") as img:
                await c.bot.send_photo(chat_id=cm.chat.id, photo=img, caption=caption)
        else:
            await c.bot.send_message(chat_id=cm.chat.id, text=caption)
        log.info("Welcome bheja: %s (%s)", user.full_name, user.id)
    except Exception as e:
        log.warning("Welcome bhejne me error: %s", e)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.error("Update me error: %s", context.error, exc_info=context.error)


# ----------------------------- MAIN -----------------------------
def main() -> None:
    if not BOT_TOKEN:
        raise SystemExit("❌ BOT_TOKEN set nahi hai. .env file me daalo.")

    start_health_server()

    app: Application = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("about", cmd_about))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("ai", cmd_ai))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(ChatMemberHandler(on_new_member,
                                      chat_member_types=ChatMemberHandler.CHAT_MEMBER))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(on_error)

    log.info("✅ Bot chalu ho raha hai... (models: %s)", ", ".join(GEMINI_MODELS))
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
