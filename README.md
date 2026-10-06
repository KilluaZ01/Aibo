# Aibo — the music friend in your voice chat

Aibo joins your voice channel, plays music, and listens for its name.
Say **"Aibo, play K by Cigarettes After Sex"** out loud, or type it. No slash commands.

It reads how you're feeling ("aibo rough day, play something") and answers that first,
then picks a song that fits.

---

## Quick start

### 1. Prerequisites

- Python 3.10+ (3.11+ recommended)
- FFmpeg (`sudo apt install ffmpeg`)
- git (voice receive installs from GitHub)
- A Discord bot token with **Message Content Intent** enabled
- A Hugging Face token (the same one the Nima/Arik/Zidan bots use)

### 2. Install

```bash
cd Aibo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`discord-ext-voice-recv` replaces `discord.py` with a pinned fork that can **hear** voice.
Install Aibo in its own venv so it doesn't change the discord.py the other bots use.

### 3. Configure

```bash
cp .env.example .env
# fill in DISCORD_TOKEN and HUGGINGFACE_TOKEN
```

### 4. Run

```bash
python bot.py
```

The first start downloads the speech model (~150 MB for `base`).

---

## Talking to Aibo

**Typed** — mention it, reply to it, say "aibo", or just use a music word:

```
aibo join
play K by Cigarettes After Sex
aibo i'm so tired, put something on
something for a rainy night
skip / pause / resume / volume 30 / what's playing / show the queue
remove the second song / leave the vc
```

**Spoken** — Aibo must be in your voice channel first (`aibo join`, or play anything). Then:

- "Aibo, play Apocalypse" — all in one breath, or
- "Aibo" … *(the music dips, so you know it heard)* … "play something chill"

The request and the answer show up in the voice channel's text chat
(or wherever you last typed to Aibo).

---

## How it works

```
Voice chat audio (per person)            Typed message
      ↓ phrase ends after ~0.7 s silence       ↓
Speech-to-text (local Whisper)                 │
      ↓ starts with "Aibo"?                    │
      └──────────────┬─────────────────────────┘
                     ↓
      Fast path (ai/intent.py)   "skip", "volume 30", "play K" → no AI call
                     ↓ otherwise
      Hugging Face LLM            reads the mood, picks an action + what to say (JSON)
                     ↓
      yt-dlp (YouTube → SoundCloud fallback) → FFmpeg → Discord voice
```

| File | Job |
|---|---|
| `voice/listener.py` | Hears each person, splits phrases, catches the wake word, ducks the music |
| `voice/stt.py` | Speech-to-text: local faster-whisper or Hugging Face Whisper |
| `ai/intent.py` | Understands clear commands without the LLM |
| `ai/client.py` | Hugging Face model chain, rate limit, JSON decisions, replies |
| `ai/prompts.py` | Aibo's personality and rules |
| `ai/mood.py` | Notices sad / tired / stressed / hyped / chill (English + Nepali) |
| `ai/router.py` | Decides whether a typed message is for Aibo |
| `music/search.py` | yt-dlp search, cookies, SoundCloud fallback |
| `music/player.py` | Per-guild playback, queue, volume, voice connection |
| `discord_bot/events.py` | Wires typed and spoken requests to the same handler |

### Privacy

Phrases without the wake word are transcribed **locally** and thrown away. Nothing is saved.
With `STT_BACKEND=hf`, every phrase is sent to Hugging Face, so prefer `local`.
Tell your friends Aibo can hear the channel while it's in VC.

---

## YouTube on Oracle ("Sign in to confirm you're not a bot")

YouTube blocks most datacenter IPs. In order of effort:

1. **Do nothing.** `SOUNDCLOUD_FALLBACK=true` (default) plays the song from SoundCloud when YouTube refuses.
2. **Cookies.** Log into a *throwaway* Google account in a browser, export `cookies.txt` for
   youtube.com (e.g. the "Get cookies.txt LOCALLY" extension), copy it to the server, and set
   `YTDLP_COOKIES=/path/to/cookies.txt`. Don't use your main account; it can get flagged.
3. **Keep yt-dlp fresh.** `pip install -U yt-dlp`. YouTube changes often and old versions break.

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DISCORD_TOKEN` | ✅ | — | Discord bot token |
| `HUGGINGFACE_TOKEN` | ✅ | — | Hugging Face token |
| `HF_MODELS` | | Llama 3.1 8B, Qwen 2.5 7B, Mistral 7B | Models tried in order |
| `HF_CALLS_PER_MIN` | | `20` | Cap on LLM calls per minute |
| `VOICE_LISTEN` | | `true` | Listen for the wake word in voice chat |
| `WAKE_WORDS` | | `aibo,ai bo,…` | Spellings that count as the wake word |
| `STT_BACKEND` | | `local` | `local` (faster-whisper) or `hf` |
| `STT_LOCAL_MODEL` | | `base` | `tiny` / `base` / `small` |
| `STT_HF_MODEL` | | `openai/whisper-large-v3-turbo` | Used when `STT_BACKEND=hf` |
| `YTDLP_COOKIES` | | — | Path to a YouTube cookies.txt |
| `SOUNDCLOUD_FALLBACK` | | `true` | Use SoundCloud when YouTube refuses |
| `SONG_COMMENTS` | | `true` | Say something when the queue moves on (max once per 20 min) |
| `BOT_NAME` | | `Aibo` | Name and wake word |
| `MAX_CONTEXT_MESSAGES` | | `12` | Conversation window per guild |
| `LOG_LEVEL` | | `INFO` | `DEBUG / INFO / WARNING / ERROR` |

## Discord permissions

View Channels, Send Messages, Read Message History, Connect, Speak, Use Voice Activity.
Enable **Message Content Intent** in the Developer Portal → Bot.
