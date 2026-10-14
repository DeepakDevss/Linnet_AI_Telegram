#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Linnet AI — long-term memory store (separate file).

Keeps a small, growing profile for every user and every chat the bot has seen:

  users:  id, name, first_name, username, language, msg count, first/last seen,
          likes/notes (what they like, what they told us), ai (which AI they
          talk about, counted), chats they appear in
  chats:  id, type, title, language, msg count, first/last seen

Everything is plain JSON on disk so it survives restarts. Written atomically.

NOTE: Render's disk is ephemeral, so this file resets on a redeploy. To keep it
across deploys, mount a Render Disk (or set MEMORY_FILE to a persistent path).
"""

import json
import os
import threading
import time

MEMORY_FILE = (os.environ.get("MEMORY_FILE") or
               os.path.join(os.path.dirname(os.path.abspath(__file__)), "linnet_memory.json"))
# 0 = UNLIMITED (no cap). Every new user / chat automatically gets its own entry.
MAX_NOTES = int(os.environ.get("MEMORY_MAX_NOTES", "0") or "0")
MAX_NOTE_LEN = int(os.environ.get("MEMORY_MAX_NOTE_LEN", "300") or "300")
MAX_CHATS_PER_USER = int(os.environ.get("MEMORY_MAX_CHATS", "0") or "0")

_LOCK = threading.Lock()
_DATA = {"users": {}, "chats": {}}
_LOADED = False

# AI names we quietly count when a user talks about them.
_AI_NAMES = {
    "chatgpt": "ChatGPT", "gpt": "ChatGPT", "openai": "ChatGPT",
    "gemini": "Gemini", "bard": "Gemini",
    "claude": "Claude", "anthropic": "Claude",
    "grok": "Grok", "deepseek": "DeepSeek",
    "copilot": "Copilot", "perplexity": "Perplexity",
    "llama": "Llama", "mistral": "Mistral", "qwen": "Qwen",
    "siri": "Siri", "alexa": "Alexa",
}


# --------------------------------------------------------------------------- io
def load() -> None:
    """Load the memory file once (safe to call many times)."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            _DATA["users"] = raw.get("users", {}) or {}
            _DATA["chats"] = raw.get("chats", {}) or {}
    except FileNotFoundError:
        pass
    except Exception:
        pass


def save() -> None:
    """Write the memory file atomically."""
    try:
        with _LOCK:
            tmp = MEMORY_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(_DATA, f, ensure_ascii=False, indent=0)
            os.replace(tmp, MEMORY_FILE)
    except Exception:
        pass


def _touch(obj: dict) -> dict:
    obj["last_seen"] = int(time.time())
    obj.setdefault("first_seen", obj["last_seen"])
    obj["msgs"] = int(obj.get("msgs", 0)) + 1
    return obj


# ----------------------------------------------------------------------- users
def remember_user(user, chat_id=None, lang=None, chat_type=None) -> None:
    """Record/refresh a user. `user` is a telegram User (or any object with the fields)."""
    if user is None or getattr(user, "is_bot", False):
        return
    uid = str(getattr(user, "id", "") or "")
    if not uid:
        return
    load()
    with _LOCK:
        u = _DATA["users"].setdefault(uid, {})
    _touch(u)
    u["name"] = getattr(user, "full_name", None) or u.get("name") or "?"
    u["first"] = getattr(user, "first_name", None) or u.get("first", "")
    if getattr(user, "username", None):
        u["username"] = user.username
    if lang:
        u["language"] = lang
    u.setdefault("likes", [])
    u.setdefault("ai", {})
    u.setdefault("chats", [])
    if chat_id is not None:
        cid = str(chat_id)
        if cid not in u["chats"]:
            u["chats"].append(cid)
            if MAX_CHATS_PER_USER:
                u["chats"] = u["chats"][-MAX_CHATS_PER_USER:]
    if chat_type:
        u["last_chat_type"] = chat_type
    save()


def note(user_id, text: str) -> bool:
    """Add a short fact about a user (deduped, capped). Returns True if stored."""
    text = (text or "").strip().replace("\n", " ")[:MAX_NOTE_LEN]
    if not text or user_id is None:
        return False
    load()
    uid = str(user_id)
    with _LOCK:
        u = _DATA["users"].setdefault(uid, {})
    likes = u.setdefault("likes", [])
    low = {x.lower() for x in likes}
    if text.lower() in low:
        return False
    likes.append(text)
    if MAX_NOTES and len(likes) > MAX_NOTES:
        del likes[0:len(likes) - MAX_NOTES]
    save()
    return True


def count_ai(user_id, text: str) -> None:
    """Count AI names the user mentions (which AI they use/talk about most)."""
    if not text or user_id is None:
        return
    low = text.lower()
    hits = {v for k, v in _AI_NAMES.items() if k in low}
    if not hits:
        return
    load()
    uid = str(user_id)
    with _LOCK:
        u = _DATA["users"].setdefault(uid, {})
    ai = u.setdefault("ai", {})
    for name in hits:
        ai[name] = int(ai.get(name, 0)) + 1
    save()


def user_summary(user_id) -> str:
    """One-line human summary of what we remember about a user ('' if nothing)."""
    load()
    u = _DATA["users"].get(str(user_id))
    if not u:
        return ""
    bits = [f"name: {u.get('name', '?')} (ID {user_id})"]
    if u.get("username"):
        bits.append(f"username: @{u['username']}")
    if u.get("language"):
        bits.append(f"usually writes in: {u['language']}")
    bits.append(f"messages seen: {u.get('msgs', 0)}")
    if u.get("first_seen"):
        bits.append("first seen: " + time.strftime("%d %b %Y", time.localtime(u["first_seen"])))
    ai = u.get("ai") or {}
    if ai:
        top = sorted(ai.items(), key=lambda kv: -kv[1])[:3]
        bits.append("AI they mention: " + ", ".join(f"{k} ({v})" for k, v in top))
    likes = u.get("likes") or []
    if likes:
        bits.append("notes about them: " + "; ".join(likes))
    return "What you remember about this person - " + " | ".join(bits) + "."


def user_basic(user_id) -> str:
    """Short public line for a normal user: name, ID, and when they joined."""
    load()
    u = _DATA["users"].get(str(user_id))
    if not u:
        return ""
    joined = (time.strftime("%d %b %Y", time.localtime(u["first_seen"]))
              if u.get("first_seen") else "?")
    return f"{u.get('name', '?')} (ID {user_id}) - you joined on {joined}."


def all_users() -> list:
    """Every remembered user, most active first (for the owner view)."""
    load()
    out = [{"id": uid, "name": u.get("name", "?"), "username": u.get("username"),
            "language": u.get("language"), "msgs": u.get("msgs", 0),
            "first_seen": u.get("first_seen"), "likes": u.get("likes") or [],
            "ai": u.get("ai") or {}}
           for uid, u in _DATA["users"].items()]
    out.sort(key=lambda x: -(x.get("msgs") or 0))
    return out


def all_chats() -> list:
    """Every remembered chat, most active first (for the owner view)."""
    load()
    out = [{"id": cid, "type": c.get("type"), "title": c.get("title"),
            "language": c.get("language"), "msgs": c.get("msgs", 0),
            "first_seen": c.get("first_seen")}
           for cid, c in _DATA["chats"].items()]
    out.sort(key=lambda x: -(x.get("msgs") or 0))
    return out


def top_ai(user_id) -> str:
    ai = (get_user(user_id).get("ai") or {})
    if not ai:
        return ""
    return max(ai.items(), key=lambda kv: kv[1])[0]


# ----------------------------------------------------------------------- chats
def remember_chat(chat, lang=None, chat_type=None) -> None:
    """Record/refresh a chat (private, group, or channel)."""
    if chat is None:
        return
    cid = str(getattr(chat, "id", "") or "")
    if not cid:
        return
    load()
    with _LOCK:
        c = _DATA["chats"].setdefault(cid, {})
    _touch(c)
    c["type"] = chat_type or getattr(chat, "type", None) or c.get("type") or "?"
    title = getattr(chat, "title", None)
    if title:
        c["title"] = title
    if getattr(chat, "username", None):
        c["username"] = chat.username
    if lang:
        c["language"] = lang
    save()


def chat_summary(chat_id) -> str:
    load()
    c = _DATA["chats"].get(str(chat_id))
    if not c:
        return ""
    bits = [f"type: {c.get('type', '?')}"]
    if c.get("title"):
        bits.append(f"title: {c['title']}")
    if c.get("language"):
        bits.append(f"usual language: {c['language']}")
    bits.append(f"messages seen here: {c.get('msgs', 0)}")
    if c.get("first_seen"):
        bits.append("first seen: " + time.strftime("%d %b %Y", time.localtime(c["first_seen"])))
    return "Chat memory - " + " | ".join(bits) + "."


# ----------------------------------------------------------------------- admin
def get_user(user_id) -> dict:
    load()
    return _DATA["users"].get(str(user_id), {}) or {}


def stats() -> dict:
    load()
    return {"users": len(_DATA["users"]), "chats": len(_DATA["chats"])}


def forget(user_id) -> bool:
    load()
    with _LOCK:
        return _DATA["users"].pop(str(user_id), None) is not None


def export() -> dict:
    load()
    return json.loads(json.dumps(_DATA))
