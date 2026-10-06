# Nova — AI Discord Music Companion

An AI-powered Discord bot that understands natural language and controls SoundCloud music through conversation.

No slash commands. No `!play`. Just talk to it.

---

## Quick start

### 1. Prerequisites

- Python 3.11+
- Java 17+ (for Lavalink)
- A Discord application with a bot token
- An Anthropic API key

### 2. Clone and install

```bash
git clone <your-repo-url>
cd ai-discord-bot
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your DISCORD_TOKEN and LLM_API_KEY
```

### 4. Start Lavalink

Download `Lavalink.jar` from [lavalink.dev](https://github.com/lavalink-devs/Lavalink/releases).

```bash
mkdir -p lavalink
# Copy lavalink/application.yml into the same folder
cd lavalink
java -jar Lavalink.jar
```

Wait for: `Lavalink is ready to accept connections.`

### 5. Run the bot

```bash
python bot.py
```

---

## Architecture

```
Discord Message
      ↓
Message Router   ← cheap pattern check; ignores noise
      ↓
LLM (Claude)     ← understands intent, picks tool
      ↓
Tool Call        ← validated Python function
      ↓
SoundCloud → Lavalink → Discord Voice
      ↓
Tool Result
      ↓
LLM              ← writes natural response
      ↓
Discord Message
```

### Key design decisions

| Layer | Responsibility |
|---|---|
| `ai/router.py` | Decides if a message should reach the LLM at all |
| `ai/client.py` | Anthropic API + tool-use loop |
| `ai/tools.py` | Tool definitions sent to the LLM |
| `ai/prompts.py` | System prompt + personality |
| `music/manager.py` | Lavalink WS session + per-guild player registry |
| `music/player.py` | Per-guild playback state + Lavalink REST calls |
| `music/soundcloud.py` | SoundCloud search via Lavalink REST |
| `music/queue.py` | Simple track queue |
| `discord_bot/events.py` | Discord event wiring |
| `discord_bot/context.py` | Bounded per-guild conversation history |

---

## Music source

**SoundCloud only.**

The pipeline is: `SoundCloud → Lavalink → Discord Voice`

No YouTube, Spotify, Deezer, or other providers. If a SoundCloud search fails, the bot says so.

---

## Natural language examples

```
play K by Cigarettes After Sex
bro put some sad shit on
can we listen to Cigarettes After Sex
skip this
next
I don't wanna hear this
pause
hold the music
resume
what's playing?
show me the queue
put Sweater Weather next
remove the second song
clear the queue
turn it down
volume 30
louder
leave the vc
disconnect
```

---

## Discord permissions required

- Read Messages / View Channels
- Send Messages
- Read Message History
- Connect (voice)
- Speak (voice)
- Use Voice Activity

Enable **Message Content Intent** in the Discord Developer Portal → Bot settings.

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DISCORD_TOKEN` | ✅ | — | Discord bot token |
| `LLM_API_KEY` | ✅ | — | Anthropic API key |
| `LLM_MODEL` | | `claude-sonnet-4-6` | Anthropic model |
| `LAVALINK_HOST` | | `127.0.0.1` | Lavalink server host |
| `LAVALINK_PORT` | | `2333` | Lavalink server port |
| `LAVALINK_PASSWORD` | | `youshallnotpass` | Lavalink password |
| `LAVALINK_SECURE` | | `false` | Use WSS/HTTPS |
| `BOT_NAME` | | `Nova` | Bot personality name |
| `MAX_CONTEXT_MESSAGES` | | `12` | Conversation window per guild |
| `LOG_LEVEL` | | `INFO` | `DEBUG / INFO / WARNING / ERROR` |

---

## Test checklist

After setup, verify these work naturally:

- [ ] `play K by Cigarettes After Sex`
- [ ] `play Apocalypse by Cigarettes After Sex`
- [ ] `put Sweater Weather next`
- [ ] `skip`
- [ ] `skip this`
- [ ] `pause`
- [ ] `resume`
- [ ] `what's playing?`
- [ ] `show me the queue`
- [ ] `remove the second song`
- [ ] `clear the queue`
- [ ] `turn the volume down`
- [ ] `leave the vc`
- [ ] `bro put K on`
- [ ] `can we listen to cigarettes after sex`
- [ ] `I'm feeling sad, play something`
- [ ] `put something chill on`
- [ ] `nah skip this one`
- [ ] `what did you just play?`

---

## Project structure

```
ai-discord-bot/
│
├── bot.py                  ← entrypoint
├── config.py               ← environment configuration
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── ai/
│   ├── client.py           ← Anthropic API + tool-use loop
│   ├── prompts.py          ← system prompt + personality
│   ├── router.py           ← message routing (should we call the LLM?)
│   └── tools.py            ← tool definitions for the LLM
│
├── music/
│   ├── manager.py          ← Lavalink WS + guild player registry
│   ├── player.py           ← per-guild music player
│   ├── queue.py            ← track queue
│   └── soundcloud.py       ← SoundCloud search via Lavalink
│
├── discord_bot/
│   ├── events.py           ← Discord event handlers
│   └── context.py          ← per-guild conversation history
│
├── utils/
│   └── logging.py          ← structured logging setup
│
└── lavalink/
    └── application.yml     ← Lavalink config (SoundCloud only)
```

---

## Development stages

Build and verify in order:

1. **Stage 1** — Discord login, env vars, basic message receiving
2. **Stage 2** — Lavalink + SoundCloud playback (verify before adding AI)
3. **Stage 3** — Music tools (test independently)
4. **Stage 4** — LLM tool-calling integration
5. **Stage 5** — Natural conversation
6. **Stage 6** — Conversation context
7. **Stage 7** — Intent routing + cost control
