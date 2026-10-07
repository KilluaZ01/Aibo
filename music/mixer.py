"""
music/mixer.py — lay Aibo's voice over the music.

A Discord voice connection plays one AudioSource at a time. MixedSource
wraps the song and, when Aibo talks, adds the speech on top (with the music
pulled down) instead of stopping the song.
"""

from __future__ import annotations

import threading
from typing import Optional

import discord
import numpy as np

FRAME_BYTES = 3840          # 20 ms of 48 kHz stereo s16le
VOICE_OVER_MUSIC = 0.35     # music level while Aibo is talking


class MixedSource(discord.AudioSource):
    def __init__(self, music: discord.PCMVolumeTransformer) -> None:
        self.music = music
        self._voice: Optional[discord.AudioSource] = None
        self._lock = threading.Lock()

    def say(self, voice: discord.AudioSource) -> None:
        """Start speaking over the music (replaces anything still being said)."""
        with self._lock:
            old, self._voice = self._voice, voice
        if old:
            old.cleanup()

    def is_opus(self) -> bool:
        return False

    def read(self) -> bytes:
        music = self.music.read()
        with self._lock:
            voice = self._voice
        if voice is None:
            return music

        speech = voice.read()
        if not speech:
            with self._lock:
                if self._voice is voice:
                    self._voice = None
            voice.cleanup()
            return music
        if not music:
            # Song ended mid-sentence: finish the sentence, then end.
            return speech

        m = np.frombuffer(music.ljust(FRAME_BYTES, b"\0"), dtype=np.int16).astype(np.int32)
        v = np.frombuffer(speech.ljust(FRAME_BYTES, b"\0"), dtype=np.int16).astype(np.int32)
        mixed = (m * VOICE_OVER_MUSIC + v).clip(-32768, 32767).astype(np.int16)
        return mixed.tobytes()

    def cleanup(self) -> None:
        self.music.cleanup()
        with self._lock:
            voice, self._voice = self._voice, None
        if voice:
            voice.cleanup()
