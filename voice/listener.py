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
import os
import re
import threading
import time
import wave
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

PHRASE_GAP = 0.55         # seconds of silence that ends a phrase
MAX_PHRASE = 10.0         # cut very long phrases
MIN_PHRASE = 0.35         # ignore coughs and clicks
ARMED_FOR = 7.0           # after a bare wake word, wait this long for the command
SPEECH_HOLD = 1.0         # keep the music down this long after someone stops talking
FOLLOW_UP = 20.0          # after the bot answers someone, they can keep talking without the name
MIN_RMS = 0.01            # quieter than this is background noise
MAX_PENDING = 4           # drop phrases if speech-to-text falls behind
STATUS_EVERY = 10.0       # seconds between "[Voice] status" log lines

_GREETINGS = {"hey", "hi", "ok", "okay", "yo", "oi", "oye", "hello"}
_WAKE = set(settings.wake_words)
_NAME = settings.bot_name.lower()

# Words Google might make of "Aibo" (rainbow, elbow, ibo…). Only trusted when
# a music command follows, so "combo was crazy" doesn't count.
_SOUNDS_LIKE_NAME = re.compile(r"[a-z]{0,5}(bo|bow|boo|bou|bu|vo)")

# Phrases that are clearly a music request on their own.
_MUSIC_START = re.compile(
    r"(play|queue|put on|put|add|skip|next|change|stop|pause|resume|unpause|volume|"
    r"louder|quieter|increase|decrease|lower|raise|turn (it |the music )?(up|down)|"
    r"what'?s playing|leave|join)\b"
)
# Without the wake word, a phrase that mentions music anywhere goes to the
# LLM, which decides whether it was meant for the bot ("bro that's not what I
# want, stop") or just friends talking ("let's play valorant" → ignored).
_MUSIC_WORDS = re.compile(
    r"\b(play|playing|song|songs|music|track|stop|skip|next|change|pause|resume|"
    r"volume|louder|quieter|loud|increase|decrease|queue|"
    r"bajau|bajaideu|bajaa|gana|gaana|geet)\b"
)

# (member, request, every speech-to-text guess, whether they said the bot's name)
# Ordinary words Google has produced for "Milo" in front of a command.
_NEAR_MISSES = {
    "no", "low", "hello", "below", "willow", "mellow", "mil", "mi", "me", "my",
    "will you", "mutton",
}

CommandHandler = Callable[[discord.Member, str, list[str], bool], Awaitable[None]]


def said_name(text: str) -> bool:
    """True when the phrase starts with the bot's name ("Milo, …", "hey Milo …")."""
    words = re.sub(r"[^\w\s']", " ", text.lower()).split()
    if words and words[0] in _GREETINGS:
        words = words[1:]
    for n in (2, 1):
        if len(words) >= n:
            cand = " ".join(words[:n])
            if cand in _WAKE or (
                n == 1 and difflib.SequenceMatcher(None, cand, _NAME).ratio() >= 0.8
            ):
                return True
    return False


def split_wake_word(text: str) -> tuple[bool, str]:
    """Return (heard_wake_word, rest_of_phrase).

    - "Aibo <anything>"        → wake, any request (a bare "Aibo" arms the listener)
    - "hey <music command>"    → wake ("hey play ocean eyes", "hey skip")
    - "<music command>"        → wake for the safe ones ("play ocean eyes", "skip")
    """
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
        # Sound-alike of the name, but only in front of a music command.
        if _NAME == "aibo" and rest and _SOUNDS_LIKE_NAME.fullmatch(rest[0]):
            after = " ".join(rest[1:])
            if _MUSIC_START.match(after):
                return True, after
        # Common words speech-to-text turns the name into ("no play…" = "Milo
        # play…"). Too ordinary to count alone, so only with a command after.
        for n in (2, 1):
            if len(rest) > n and " ".join(rest[:n]) in _NEAR_MISSES:
                after = " ".join(rest[n:])
                if _MUSIC_START.match(after):
                    return True, after

    if not words or settings.voice_name_only:
        # Name-only mode: nothing counts unless it starts with the bot's name.
        return False, ""
    phrase = " ".join(words)
    # "hey play ocean eyes", "ok skip"
    if words[0] in _GREETINGS:
        after = " ".join(words[1:])
        if _MUSIC_START.match(after):
            return True, after
    if settings.voice_direct_commands and _MUSIC_WORDS.search(phrase):
        return True, phrase
    return False, ""


def _to_16k_mono(pcm: bytes) -> np.ndarray:
    samples = np.frombuffer(pcm, dtype=np.int16)
    samples = samples[: len(samples) - len(samples) % 2].reshape(-1, 2).mean(axis=1)
    samples = samples[: len(samples) - len(samples) % 3].reshape(-1, 3).mean(axis=1)
    return (samples / 32768.0).astype(np.float32)


class _Phrase:
    __slots__ = ("user_id", "member", "chunks", "start", "last")

    def __init__(self, user_id: int, member: Optional[discord.Member]) -> None:
        self.user_id = user_id
        self.member = member          # None until we can resolve who it is
        self.chunks: list[bytes] = []
        self.start = self.last = time.monotonic()


if VOICE_RECV_AVAILABLE:

    class _PhraseSink(voice_recv.AudioSink):
        """Collects PCM per speaker. write() runs on the voice thread."""

        def __init__(self) -> None:
            super().__init__()
            self.lock = threading.Lock()
            self.phrases: dict[int, _Phrase] = {}
            # Counters for the periodic status log
            self.packets = 0
            self.unknown = 0
            self.empty = 0
            self.last_speech = 0.0

        def wants_opus(self) -> bool:
            return False

        def write(self, user, data) -> None:
            if getattr(user, "bot", False):
                return
            if not data.pcm:
                # Packet arrived but couldn't be decrypted/decoded.
                self.empty += 1
                return
            user_id = getattr(user, "id", None)
            if user_id is None:
                # Discord hasn't told us whose stream this is yet; the SSRC
                # map usually knows. The member is looked up later.
                ssrc = getattr(getattr(data, "packet", None), "ssrc", None)
                vc = self.voice_client
                if ssrc is not None and vc is not None:
                    user_id = vc._get_id_from_ssrc(ssrc)
            with self.lock:
                self.packets += 1
                if user_id is None:
                    self.unknown += 1
                    return
                phrase = self.phrases.get(user_id)
                if phrase is None:
                    phrase = self.phrases[user_id] = _Phrase(user_id, user)
                phrase.chunks.append(data.pcm)
                phrase.last = self.last_speech = time.monotonic()

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
        self._follow_up: dict[int, float] = {}  # user id → in conversation until
        self._busy = 0          # requests being handled right now
        self._ducked = False
        self._pending = 0
        self._stt_lock = asyncio.Lock()
        self._vc: Optional[discord.VoiceClient] = None
        # Status counters, logged every STATUS_EVERY seconds while listening
        self._phrases = 0
        self._too_quiet = 0
        self._transcribed = 0
        self._last_packets = 0
        self._last_diag: dict[str, int] = {}

    def attach(self, voice_client: discord.VoiceClient) -> bool:
        if not VOICE_RECV_AVAILABLE or not hasattr(voice_client, "listen"):
            return False
        self._vc = voice_client
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
        self._armed.clear()
        if self._ducked:
            self._ducked = False
            self._duck(False)

    def _log_status(self) -> None:
        sink = self._sink
        if not sink:
            return
        vc = self._vc
        dave = getattr(vc, "dave_ready", "n/a")
        new = sink.packets - self._last_packets
        self._last_packets = sink.packets
        playing = bool(vc and vc.is_playing())
        listening = bool(vc and getattr(vc, "is_listening", lambda: False)())
        log.info(
            "[Voice] status: +%d packets in last %ds (total %d, unknown speaker %d, "
            "undecodable %d) phrases=%d too quiet=%d transcribed=%d | bot playing=%s "
            "listening=%s encryption ready=%s pending=%d",
            new, int(STATUS_EVERY), sink.packets, sink.unknown, sink.empty, self._phrases,
            self._too_quiet, self._transcribed, playing, listening, dave, self._pending,
        )
        # The voice library's own decrypt/decode counters, as changes since last time.
        try:
            diag = vc.get_recv_diagnostics() if vc else {}
        except Exception as exc:
            diag = {}
            log.debug("[Voice] diagnostics unavailable: %s", exc)
        keys = [
            k for k, v in diag.items()
            if isinstance(v, int) and not isinstance(v, bool)
            and (k.startswith(("dave_inner", "opus_decode", "jitter", "pcm_frames", "dave_plaintext")))
        ]
        changes = {k: diag[k] - self._last_diag.get(k, 0) for k in keys}
        changes = {k: v for k, v in changes.items() if v}
        self._last_diag = {k: diag[k] for k in keys}
        if changes:
            log.info("[Voice] decoder: %s", changes)
        if sink.packets == 0:
            log.info(
                "[Voice] No audio received yet. Talk in the channel; if this stays 0, "
                "voice receive isn't getting audio from Discord."
            )

    async def _poll(self) -> None:
        last_status = time.monotonic()
        try:
            while True:
                await asyncio.sleep(0.15)
                if time.monotonic() - last_status >= STATUS_EVERY:
                    last_status = time.monotonic()
                    self._log_status()
                self._expire_armed()
                self._update_duck()
                if not self._sink:
                    continue
                for phrase in self._sink.take_finished():
                    if self._pending >= MAX_PENDING:
                        log.warning(
                            "[Voice] Speech-to-text is behind (%d waiting), dropped a phrase",
                            self._pending,
                        )
                        continue
                    self._pending += 1
                    asyncio.create_task(self._handle(phrase))
        except asyncio.CancelledError:
            pass

    def _save_debug_audio(self, audio: np.ndarray, guesses: list[str]) -> None:
        """DEBUG_SAVE_AUDIO=true: keep the phrase as a wav to check what we really got."""
        try:
            os.makedirs("logs/audio", exist_ok=True)
            playing = bool(self._vc and self._vc.is_playing())
            name = time.strftime("%H%M%S") + f"_{'music' if playing else 'quiet'}_{self._transcribed}.wav"
            path = os.path.join("logs/audio", name)
            pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(pcm)
            log.debug("[Voice] saved %s (google: %r)", path, guesses[:1])
        except Exception as exc:
            log.debug("[Voice] couldn't save debug audio: %s", exc)

    def _alone_with(self, member: discord.Member) -> bool:
        channel = getattr(self._vc, "channel", None)
        if channel is None:
            return False
        humans = [m for m in channel.members if not m.bot]
        return len(humans) == 1 and humans[0].id == member.id

    async def _resolve_member(self, user_id: int) -> Optional[discord.Member]:
        guild = getattr(getattr(self._vc, "channel", None), "guild", None)
        if guild is None:
            return None
        member = guild.get_member(user_id)
        if member is None:
            try:
                member = await guild.fetch_member(user_id)
            except discord.DiscordException:
                return None
        return None if member.bot else member

    def _update_duck(self) -> None:
        """Music down whenever someone is talking (or Milo is handling a request).

        On speakers, people's mics pick up the music; turning it down while
        they speak keeps it out of what speech-to-text hears.
        """
        speaking = bool(self._sink) and time.monotonic() - self._sink.last_speech < SPEECH_HOLD
        want = speaking or bool(self._armed) or self._busy > 0
        if want != self._ducked:
            self._ducked = want
            self._duck(want)

    def _expire_armed(self) -> None:
        now = time.monotonic()
        expired = [uid for uid, until in self._armed.items() if until < now]
        for uid in expired:
            del self._armed[uid]

    async def _handle(self, phrase: _Phrase) -> None:
        try:
            self._phrases += 1
            pcm = b"".join(phrase.chunks)
            seconds = len(pcm) / (48000 * 2 * 2)
            if seconds < MIN_PHRASE:
                return
            audio = _to_16k_mono(pcm)
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < MIN_RMS:
                self._too_quiet += 1
                log.debug("[Voice] %.1fs phrase too quiet (rms %.4f)", seconds, rms)
                return

            async with self._stt_lock:
                stt_started = time.monotonic()
                guesses = await stt.transcribe_all(audio)
                stt_took = time.monotonic() - stt_started
            self._transcribed += 1
            # Only at DEBUG: what people say without the wake word is nobody's business.
            log.debug(
                "[Voice] heard %.1fs (rms %.4f, speech-to-text %.1fs): %r",
                seconds, rms, stt_took, guesses,
            )
            if settings.debug_save_audio:
                self._save_debug_audio(audio, guesses)
            if not guesses:
                return
            # The top guess often mangles the name; a runner-up may have it.
            text = guesses[0]
            heard, rest = split_wake_word(text)
            for alt in guesses[1:]:
                if heard:
                    break
                heard, rest = split_wake_word(alt)
                if heard:
                    text = alt
            # Put the guess we went with first; the LLM sees all of them.
            guesses = [text] + [g for g in guesses if g != text]

            member = phrase.member or await self._resolve_member(phrase.user_id)
            if member is None:
                log.warning("[Voice] Couldn't work out who user %s is", phrase.user_id)
                return
            armed = self._armed.pop(member.id, None) is not None
            # Mid-conversation: no need to say the name again.
            if not settings.voice_name_only:
                armed = armed or self._follow_up.get(member.id, 0) > time.monotonic()

            if heard and not rest:
                # Bare "Aibo" — duck the music so they know we heard, wait for the ask.
                log.info("[Voice] Wake word from %s", member)
                self._armed[member.id] = time.monotonic() + ARMED_FOR
                return

            # Alone in the call with the bot: everything said is for it.
            alone = not settings.voice_name_only and self._alone_with(member)
            command = rest if heard else (text if armed or alone else "")
            if not command:
                return

            log.info("[Voice] %s: %s", member, command)
            self._busy += 1
            try:
                named = any(said_name(g) for g in guesses)
                await self._on_command(member, command, guesses, named)
                self._follow_up[member.id] = time.monotonic() + FOLLOW_UP
            finally:
                self._busy -= 1
        except Exception:
            log.error("[Voice] Failed to handle phrase", exc_info=True)
        finally:
            self._pending -= 1
