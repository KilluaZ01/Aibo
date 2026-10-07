"""
voice/tts.py — turn Aibo's reply into speech for the voice channel.

Engines (TTS_ENGINE):
- "edge" (default): Microsoft Edge neural voices via edge-tts. Free, no key,
  sounds human. Default is en-US-BrianNeural pitched up a little (Milo). Nepali voices exist too (ne-NP-HemkalaNeural, ne-NP-SagarNeural).
- "gtts": Google Translate's voice via gTTS. Free, no key, more robotic.

Returns a path to an mp3 that FFmpeg can play, or None.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile

from config import settings

log = logging.getLogger("nova.voice.tts")

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]+", flags=re.UNICODE
)
MAX_CHARS = 300


def _short_title(card: str) -> str:
    """'🎵 **Billie Eilish - ocean eyes (Official Music Video)** — Billie Eilish' → 'ocean eyes'."""
    m = re.search(r"\*\*(.+?)\*\*", card)
    title = m.group(1) if m else card
    title = re.sub(r"[(\[][^)\]]*[)\]]", "", title)  # (Official Music Video), [Lyrics]
    title = re.sub(r"\b(official|lyrics?|video|audio|hd|4k)\b", "", title, flags=re.I)
    if " - " in title:  # "Artist - Song"
        title = title.split(" - ", 1)[1]
    return re.sub(r"\s+", " ", title).strip(" -|") or "it"


def speakable(reply: str) -> str:
    """The part of a chat reply worth saying out loud."""
    lines = [l.strip() for l in reply.splitlines() if l.strip()]
    # Prefer the conversational line over the "🎵 **Title** — Artist" card.
    talk = [l for l in lines if not l.startswith(("🎵", "➕", "🎙️"))]
    if talk:
        text = " ".join(talk)
    elif lines and lines[0].startswith(("🎵", "➕")):
        # Only the card: "Playing ocean eyes", not the whole YouTube title.
        verb = "Added" if lines[0].startswith("➕") else "Playing"
        text = f"{verb} {_short_title(lines[0])}"
    else:
        text = lines[0] if lines else ""
    text = re.sub(r"[*_`~>|]", "", text)
    text = _EMOJI.sub("", text)
    text = re.sub(r"\(#\d+ in queue\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS].rsplit(" ", 1)[0]
    return text


async def _edge(text: str, path: str) -> None:
    import edge_tts

    voice = settings.tts_voice_ne if _DEVANAGARI.search(text) else settings.tts_voice
    await edge_tts.Communicate(
        text, voice, rate=settings.tts_rate, pitch=settings.tts_pitch
    ).save(path)


def _gtts(text: str, path: str) -> None:
    from gtts import gTTS

    if _DEVANAGARI.search(text):
        gTTS(text, lang="ne").save(path)
    else:
        gTTS(text, lang="en", tld="co.in").save(path)


# Short lines repeat a lot ("Stopped.", "Volume 80."); keep their audio so the
# second time is instant instead of another ~1-2s round trip.
_CACHE: dict[str, bytes] = {}
CACHE_MAX_CHARS = 40
CACHE_MAX_ITEMS = 100


async def synthesize(text: str) -> str | None:
    if not text:
        return None
    fd, path = tempfile.mkstemp(prefix="aibo-tts-", suffix=".mp3")
    os.close(fd)
    key = f"{settings.tts_engine}|{settings.tts_voice}|{text}"
    cached = _CACHE.get(key)
    if cached:
        with open(path, "wb") as f:
            f.write(cached)
        return path
    try:
        if settings.tts_engine == "gtts":
            await asyncio.to_thread(_gtts, text, path)
        else:
            await _edge(text, path)
        if os.path.getsize(path) > 0:
            if len(text) <= CACHE_MAX_CHARS and len(_CACHE) < CACHE_MAX_ITEMS:
                with open(path, "rb") as f:
                    _CACHE[key] = f.read()
            return path
    except Exception as exc:
        log.warning("[TTS] %s failed: %s", settings.tts_engine, str(exc)[:200])
    try:
        os.remove(path)
    except OSError:
        pass
    return None
