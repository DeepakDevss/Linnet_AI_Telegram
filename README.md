# Linnet | AI Companion

A Telegram AI bot written in Python. It holds conversations, understands media, remembers users across sessions, and can reply with text or voice.

- Bot: https://t.me/Linnet_Agents_Bot
- Repository: https://github.com/DeepakDevss/Linnet_AI_Telegram

---

## Overview

Linnet runs as a single Python process and talks to Telegram through the Bot API. Google Gemini is the main language model. Groq is used as a fallback chat provider and handles the speech-to-text side. User and chat state lives in a local JSON file managed by `memory.py`.

The bot is meant to run as a long-lived service. It should work on any platform that can run a Python process and set environment variables.

---

## What it can do

- Replies in whatever language the user writes in
- Remembers users and chats between restarts
- Tells apart who said what in group conversations
- Reads photos, documents, voice notes, and audio
- Sends voice replies using text-to-speech
- Picks a fitting emoji reaction for each message

---

## Requirements

- Python 3.10 or newer
- A Telegram bot token from BotFather
- A Google AI Studio API key
- A Groq API key

---

## Setup

Clone the repo and install the dependencies:

```bash
git clone https://github.com/DeepakDevss/Linnet_AI_Telegram.git
cd Linnet_AI_Telegram

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Create a local environment file from the template:

```bash
cp env.example.txt .env
```

Open `.env` and fill in the values you need. `env.example.txt` lists every supported variable — credentials, model selection, the text-to-speech voice, group behaviour, and memory limits.

Don't commit `.env`. In production, set the same values through your host's secret or environment settings.

---

## Running

```bash
python bot.py
```

The process starts and begins polling Telegram. The same command works under any process manager or hosting platform.

---

## Configuration

All runtime settings come from environment variables. See `env.example.txt` for the full list, the default values, and a short note on each one.

---

## Commands

Command handlers are registered in `bot.py`. Check that file for the full list of commands and exactly what each one does.

---

## Project structure

```
Linnet_AI_Telegram/
├── bot.py              Main application logic
├── memory.py           Persistent JSON store for users and chats
├── requirements.txt    Python dependencies
├── env.example.txt     Environment variable template
├── welcome.png         Image used in the group welcome message
└── README.md
```

---

## Memory

Linnet keeps user and chat records in a JSON file, and the path is set through an environment variable. Writes happen atomically, so an interrupted write won't corrupt the file. The exact record layout is defined in `memory.py`.

If your host uses an ephemeral filesystem, mount a persistent volume and point the memory file path at it. Otherwise records are lost on every restart.

---

## Data handling

- All user and chat records are stored locally in the memory file. There's no external database.
- Message content goes to Google Gemini and, when fallback kicks in, to Groq. If you run Linnet in a regulated environment, review the provider policies that apply to you.
- Credentials come from environment variables and are not part of the repository.

If you run Linnet as a public service, you are the data controller for the records it stores.

---

## Contributing

Issues and pull requests are welcome.

- Don't commit secrets or `.env`.
- If you add a config variable, document it in `env.example.txt`.
- Keep `bot.py` as a single file. That's on purpose.

---

## Related projects

- **Ongoing Links** — A Telegram community bot with a force-join gate, welcome messages, referral tracking, and AI chat.
  Repository: https://github.com/DeepakDevss/Telegram_Invited_AIBot
  Bot: https://t.me/ANIME_WORLD_8BOT

---

## Author

Deepak Devs

- GitHub: https://github.com/DeepakDevss
- X: https://x.com/Deepak_Devs
- Reddit: https://www.reddit.com/u/Deepak_Devs

---

## License

There's no license file yet. Add one before you distribute or publicly modify this project.
