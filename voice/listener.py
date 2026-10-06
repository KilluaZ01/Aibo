"""
voice/listener.py — hear people in voice chat and catch "Aibo, ...".

Plain discord.py can't receive voice. discord-ext-voice-recv adds it, and
the zacker150 fork also decrypts Discord's end-to-end encrypted (DAVE)
voice, which every call has used since March 2026.

Flow per guild:
    Discord → per-user PCM (48 kHz stereo, 20 ms frames)
        ↓  phrase ends after a short silence
    16 kHz mono → speech-to-text
        ↓  starts with the wake word?
    "Aibo play K"       → command "play K"
    "Aibo" ...(pause)... → music ducks, the next phrase from that person is the command

Phrases without the wake word are transcribed locally and thrown away —
nothing is logged, stored or sent anywhere.
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import re
import threading
import time
from typing import Awaitable, Callable, Optional

import discord
import numpy as np

from config import settings
from voice import stt

log = logging.getLogger("nova.voice.listener")

try:
    from discord.ext import voice_recv  # type: ignore

    VOICE_RECV_AVAILABLE = True
except ImportError:  # plain discord.py installed
    voice_recv = None
    VOICE_RECV_AVAILABLE = False

PHRASE_GAP = 0.7          # seconds of silence that ends a phrase
MAX_PHRASE = 10.0         # cut very long phrases
MIN_PHRASE = 0.35         # ignore coughs and clicks
ARMED_FOR = 7.0           # after a bare "Aibo", wait this long for the command
MIN_RMS = 0.01            # quieter than this is background noise
MAX_PENDING = 4           # drop phrases if speech-to-text falls behind

_GREETINGS = {"hey", "hi", "ok", "okay", "yo", "oi", "oye", "hello"}
_WAKE = set(settings.wake_words)
_NAME = settings.bot_name.lower()

CommandHandler = Callable[[discord.Member, str], Awaitable[None]]


def split_wake_word(text: str) -> tuple[bool, str]:
    """Return (heard_wake_word, rest_of_phrase)."""
    words = re.sub(r"[^\w\s']", " ", text.lower()).split()
    for skip in (0, 1):
        if skip and not (words and words[0] in _GREETINGS):
            continue
        rest = words[skip:]
        for n in (2, 1):
            if len(rest) < n:
                continue
            cand = " ".join(rest[:n])
            close = n == 1 and difflib.SequenceMatcher(None, cand, _NAME).ratio() >= 0.8
            if cand in _WAKE or close:
                return True, " ".join(rest[n:]).strip()
    return False, ""


def _to_16k_mono(pcm: bytes) -> np.ndarray:
    samples = np.frombuffer(pcm, dtype=np.int16)
    samples = samples[: len(samples) - len(samples) % 2].reshape(-1, 2).mean(axis=1)
    samples = samples[: len(samples) - len(samples) % 3].reshape(-1, 3).mean(axis=1)
    return (samples / 32768.0).astype(np.float32)


class _Phrase:
    __slots__ = ("member", "chunks", "start", "last")

    def __init__(self, member: discord.Member) -> None:
        self.member = member
        self.chunks: list[bytes] = []
        self.start = self.last = time.monotonic()


if VOICE_RECV_AVAILABLE:

    class _PhraseSink(voice_recv.AudioSink):
        """Collects PCM per speaker. write() runs on the voice thread."""

        def __init__(self) -> None:
            super().__init__()
            self.lock = threading.Lock()
            self.phrases: dict[int, _Phrase] = {}

        def wants_opus(self) -> bool:
            return False

        def write(self, user, data) -> None:
            if user is None or getattr(user, "bot", False) or not data.pcm:
                return
            with self.lock:
                phrase = self.phrases.get(user.id)
                if phrase is None:
                    phrase = self.phrases[user.id] = _Phrase(user)
                phrase.chunks.append(data.pcm)
                phrase.last = time.monotonic()

        def take_finished(self) -> list[_Phrase]:
            now = time.monotonic()
            done = []
            with self.lock:
                for uid, p in list(self.phrases.items()):
                    if now - p.last >= PHRASE_GAP or p.last - p.start >= MAX_PHRASE:
                        done.append(self.phrases.pop(uid))
            return done

        def cleanup(self) -> None:
            with self.lock:
                self.phrases.clear()


class VoiceListener:
    """Wake-word listener for one guild's voice connection."""

    def __init__(
        self,
        on_command: CommandHandler,
        duck: Callable[[bool], None],
    ) -> None:
        self._on_command = on_command
        self._duck = duck
        self._sink = None
        self._task: Optional[asyncio.Task] = None
        self._armed: dict[int, float] = {}   # user id → armed until (monotonic)
        self._pending = 0
        self._stt_lock = asyncio.Lock()

    def attach(self, voice_client: discord.VoiceClient) -> bool:
        if not VOICE_RECV_AVAILABLE or not hasattr(voice_client, "listen"):
            return False
        self._sink = _PhraseSink()
        voice_client.listen(self._sink)
        self._task = asyncio.create_task(self._poll())
        log.info("[Voice] Listening for '%s' in %s", settings.bot_name, voice_client.channel)
        return True

    def detach(self, voice_client: Optional[discord.VoiceClient]) -> None:
        if self._task:
            self._task.cancel()
            self._task = None
        if voice_client and getattr(voice_client, "is_listening", lambda: False)():
            voice_client.stop_listening()
        self._sink = None
        if self._armed:
            self._armed.clear()
            self._duck(False)

    async def _poll(self) -> None:
        try:
            while True:
                await asyncio.sleep(0.15)
                self._expire_armed()
                if not self._sink:
                    continue
                for phrase in self._sink.take_finished():
                    if self._pending >= MAX_PENDING:
                        continue
                    self._pending += 1
                    asyncio.create_task(self._handle(phrase))
        except asyncio.CancelledError:
            pass

    def _expire_armed(self) -> None:
        now = time.monotonic()
        expired = [uid for uid, until in self._armed.items() if until < now]
        for uid in expired:
            del self._armed[uid]
        if expired and not self._armed:
            self._duck(False)

    async def _handle(self, phrase: _Phrase) -> None:
        try:
            pcm = b"".join(phrase.chunks)
            seconds = len(pcm) / (48000 * 2 * 2)
            if seconds < MIN_PHRASE:
                return
            audio = _to_16k_mono(pcm)
            if float(np.sqrt(np.mean(audio ** 2))) < MIN_RMS:
                return

            async with self._stt_lock:
                text = await stt.transcribe(audio)
            if not text:
                return

            member = phrase.member
            armed = self._armed.pop(member.id, None) is not None
            heard, rest = split_wake_word(text)

            if heard and not rest:
                # Bare "Aibo" — duck the music so they know we heard, wait for the ask.
                log.info("[Voice] Wake word from %s", member)
                self._armed[member.id] = time.monotonic() + ARMED_FOR
                self._duck(True)
                return

            command = rest if heard else (text if armed else "")
            if not command:
                return

            log.info("[Voice] %s: %s", member, command)
            self._duck(True)
            try:
                await self._on_command(member, command)
            finally:
                if not self._armed:
                    self._duck(False)
        except Exception:
            log.error("[Voice] Failed to handle phrase", exc_info=True)
        finally:
            self._pending -= 1
