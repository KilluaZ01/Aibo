"""
music/player.py — GuildPlayer

Controls FFmpeg + discord.py voice playback for one guild.

Flow:
    play_song()
        ↓
    search_track()      — yt-dlp search, returns Track with metadata
        ↓
    _play_track()
        ↓
    resolve_stream()    — yt-dlp re-fetches fresh audio URL
        ↓
    FFmpegPCMAudio      — streams to Discord voice
        ↓
    after_callback      — when track ends, plays next in queue
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import discord

from config import settings
from music.mixer import MixedSource
from music.queue import GuildQueue, Track
from music.search import search_track, resolve_stream
from voice.listener import VOICE_RECV_AVAILABLE, voice_recv
from voice import tts

VOICE_WAIT = 4.0  # max seconds a new song waits for Milo to finish talking
DUCK_FACTOR = 0.15  # music volume multiplier while someone is talking (keeps it out of mics)

log = logging.getLogger("nova.music.player")

FFMPEG_OPTIONS = {
    "before_options": (
        "-reconnect 1 " "-reconnect_streamed 1 " "-reconnect_delay_max 5"
    ),
    "options": "-vn",
}


class GuildPlayer:
    """Controls music playback for a single Discord guild via FFmpeg."""

    def __init__(
        self,
        guild: discord.Guild,
        bot: discord.Client,
        listener_factory=None,
        on_track_start=None,
    ) -> None:
        self.guild = guild
        self._bot = bot
        # listener_factory(player) -> VoiceListener; on_track_start(player, track) is awaited
        self._listener_factory = listener_factory
        self._on_track_start = on_track_start
        self._listener = None
        self._ducked = False

        self.queue = GuildQueue()
        self.current_track: Optional[Track] = None
        self._voice_client: Optional[discord.VoiceClient] = None
        self._volume: float = 0.5  # 0.0 – 1.0 (lower = less music leaking into mics)
        self._paused: bool = False
        self._music: Optional[discord.PCMVolumeTransformer] = None
        self._mixer: Optional[MixedSource] = None

    # ------------------------------------------------------------------
    # Voice connection
    # ------------------------------------------------------------------

    async def ensure_voice(self, channel: discord.VoiceChannel) -> bool:
        """Connect or move to the given voice channel."""
        try:
            if self._voice_client and self._voice_client.is_connected():
                if self._voice_client.channel.id != channel.id:
                    await self._voice_client.move_to(channel)
                return True

            if settings.voice_listen and VOICE_RECV_AVAILABLE:
                # Not deafened: Aibo has to hear the channel to catch its wake word.
                self._voice_client = await channel.connect(
                    cls=voice_recv.VoiceRecvClient, self_deaf=False
                )
            else:
                self._voice_client = await channel.connect(self_deaf=True)
            log.info(
                "[Music] Connected to voice channel '%s' in guild %d",
                channel.name,
                self.guild.id,
            )
            if self._listener_factory and settings.voice_listen:
                self._listener = self._listener_factory(self)
                if not self._listener.attach(self._voice_client):
                    log.warning(
                        "[Voice] Voice receive unavailable — install the "
                        "discord-ext-voice-recv fork (see README). Text commands still work."
                    )
                    self._listener = None
            return True

        except discord.DiscordException as exc:
            log.error("[Music] Failed to connect to voice: %s", exc)
            return False

    async def _disconnect_voice(self) -> None:
        if self._listener:
            self._listener.detach(self._voice_client)
            self._listener = None
        if self._voice_client and self._voice_client.is_connected():
            await self._voice_client.disconnect()
        self._voice_client = None

    @property
    def voice_channel(self):
        if self._voice_client and self._voice_client.is_connected():
            return self._voice_client.channel
        return None

    def duck(self, on: bool) -> None:
        """Lower the music while someone is talking to Aibo, restore after."""
        self._ducked = on
        self._apply_volume()

    def _apply_volume(self) -> None:
        if self._music:
            factor = DUCK_FACTOR if self._ducked else 1.0
            self._music.volume = self._volume * factor

    def _stop_audio(self) -> None:
        """Stop what's playing but keep listening.

        VoiceRecvClient.stop() also stops *receiving* audio, which would make
        Milo deaf after the first skip — use stop_playing() when it exists.
        """
        vc = self._voice_client
        if vc:
            getattr(vc, "stop_playing", vc.stop)()

    def _music_active(self) -> bool:
        """A song is loaded (playing or paused) — Aibo talking doesn't count."""
        return self.current_track is not None and self._mixer is not None

    async def speak(self, text: str) -> bool:
        """Say something in the voice channel, over the music if a song is on."""
        vc = self._voice_client
        if not settings.tts_reply or not vc or not vc.is_connected():
            return False
        line = tts.speakable(text)
        path = await tts.synthesize(line)
        if not path:
            return False
        source = _SpeechSource(path)
        if self._music_active() and not self._paused and vc.source is self._mixer:
            self._mixer.say(source)
        elif not vc.is_playing() and not vc.is_paused():
            vc.play(source)
        else:
            # Song is paused — talking would unpause the player; stay quiet.
            source.cleanup()
            return False
        log.info("[Voice] Saying: %s", line)
        return True

    @property
    def voice_channel_id(self) -> Optional[int]:
        if self._voice_client and self._voice_client.is_connected():
            return self._voice_client.channel.id
        return None

    # ------------------------------------------------------------------
    # Internal playback
    # ------------------------------------------------------------------

    async def _play_track(self, track: Track) -> dict:
        """Resolve stream and start FFmpeg playback."""
        if not self._voice_client or not self._voice_client.is_connected():
            return {"success": False, "error": "not_connected"}

        stream_url = await resolve_stream(track)
        if not stream_url:
            return {"success": False, "error": "stream_failed"}

        try:
            source = discord.FFmpegPCMAudio(stream_url, **FFMPEG_OPTIONS)
            factor = DUCK_FACTOR if self._ducked else 1.0
            music = discord.PCMVolumeTransformer(source, volume=self._volume * factor)
            mixer = MixedSource(music)
        except Exception as exc:
            log.error("[Music] FFmpeg error: %s", exc)
            return {"success": False, "error": "ffmpeg_error"}

        def after_playing(error: Optional[Exception]) -> None:
            if error:
                log.error("[Music] Playback error: %s", error)
            # Schedule next track from the event loop
            asyncio.run_coroutine_threadsafe(
                self._on_track_end(),
                self._bot.loop,
            )

        if self._voice_client.is_playing() or self._voice_client.is_paused():
            # Only Milo's voice can be on here (songs go through the queue).
            # He usually announces the song while it's being found; let him
            # finish the sentence, then start the song.
            for _ in range(int(VOICE_WAIT / 0.1)):
                if not self._voice_client.is_playing():
                    break
                await asyncio.sleep(0.1)
            self._stop_audio()
        self._music, self._mixer = music, mixer
        self._voice_client.play(mixer, after=after_playing)
        self.current_track = track
        self._paused = False

        log.info("[Music] Playing: %s — %s", track.title, track.artist)

        return {
            "success": True,
            "action": "playing",
            "title": track.title,
            "artist": track.artist,
            "duration": track.duration_ms,
            "url": track.identifier,
        }

    async def _on_track_end(self) -> None:
        """Called when a track finishes. Plays next in queue if available."""
        self.current_track = None
        self._paused = False
        self._music = self._mixer = None

        next_track = self.queue.pop_next()
        if next_track:
            log.info("[Music] Auto-playing next: %s", next_track.title)
            result = await self._play_track(next_track)
            if result.get("success") and self._on_track_start:
                try:
                    await self._on_track_start(self, next_track)
                except Exception:
                    log.error("[Music] on_track_start hook failed", exc_info=True)
        else:
            log.info("[Music] Queue empty — playback finished")

    # ------------------------------------------------------------------
    # Public music tools
    # ------------------------------------------------------------------

    async def play_song(
        self,
        query: str,
        requester: Optional[str] = None,
        voice_channel: Optional[discord.VoiceChannel] = None,
    ) -> dict:
        if not self._voice_client or not self._voice_client.is_connected():
            if not voice_channel:
                return {"success": False, "error": "not_in_voice"}
            connected = await self.ensure_voice(voice_channel)
            if not connected:
                return {"success": False, "error": "voice_connect_failed"}

        track = await search_track(query, requester)
        if not track:
            return {"success": False, "error": "no_results", "query": query}

        # If a song is on, queue it
        if self._music_active():
            pos = self.queue.add(track)
            return {
                "success": True,
                "action": "queued",
                "title": track.title,
                "artist": track.artist,
                "duration": track.duration_ms,
                "queue_position": pos,
            }

        return await self._play_track(track)

    async def queue_song(
        self,
        query: str,
        requester: Optional[str] = None,
        voice_channel: Optional[discord.VoiceChannel] = None,
    ) -> dict:
        if not self._voice_client or not self._voice_client.is_connected():
            if not voice_channel:
                return {"success": False, "error": "not_in_voice"}
            connected = await self.ensure_voice(voice_channel)
            if not connected:
                return {"success": False, "error": "voice_connect_failed"}

        track = await search_track(query, requester)
        if not track:
            return {"success": False, "error": "no_results", "query": query}

        # If no song is on, start immediately
        if not self._music_active():
            return await self._play_track(track)

        pos = self.queue.add(track)
        return {
            "success": True,
            "action": "queued",
            "title": track.title,
            "artist": track.artist,
            "duration": track.duration_ms,
            "queue_position": pos,
        }

    async def play_now(
        self,
        query: str,
        requester: Optional[str] = None,
        voice_channel: Optional[discord.VoiceChannel] = None,
    ) -> dict:
        """Replace whatever is on with this song ("wrong song, I said X")."""
        if not self._music_active():
            return await self.play_song(query, requester, voice_channel)
        track = await search_track(query, requester)
        if not track:
            return {"success": False, "error": "no_results", "query": query}
        self.queue.add_front(track)
        self._stop_audio()  # after_playing → _on_track_end plays the front track
        return {
            "success": True,
            "action": "playing",
            "title": track.title,
            "artist": track.artist,
            "duration": track.duration_ms,
        }

    async def skip_song(self) -> dict:
        if not self._voice_client or not self._music_active():
            return {"success": False, "error": "nothing_playing"}

        skipped = self.current_track
        self._stop_audio()  # triggers after_playing → _on_track_end

        return {
            "success": True,
            "action": "skipped",
            "skipped": {
                "title": skipped.title if skipped else "Unknown",
                "artist": skipped.artist if skipped else "Unknown",
            },
        }

    async def pause_music(self) -> dict:
        if not self._voice_client or not self._music_active() or self._paused:
            return {"success": False, "error": "nothing_playing"}
        self._voice_client.pause()
        self._paused = True
        return {"success": True}

    async def resume_music(self) -> dict:
        if not self._voice_client or not self._voice_client.is_paused():
            return {"success": False, "error": "not_paused"}
        self._voice_client.resume()
        self._paused = False
        return {"success": True}

    async def stop_music(self) -> dict:
        self.queue.clear()
        self.current_track = None
        self._paused = False
        if self._voice_client and (
            self._voice_client.is_playing() or self._voice_client.is_paused()
        ):
            self._stop_audio()
        return {"success": True}

    def now_playing(self) -> dict:
        if not self.current_track:
            return {"success": True, "playing": False}
        return {
            "success": True,
            "playing": True,
            "title": self.current_track.title,
            "artist": self.current_track.artist,
            "duration": self.current_track.duration_ms,
            "requester": self.current_track.requester,
            "paused": self._paused,
        }

    def show_queue(self) -> dict:
        return {
            "success": True,
            "queue": self.queue.as_list(),
            "length": len(self.queue),
        }

    async def join_voice(self, voice_channel: Optional[discord.VoiceChannel]) -> dict:
        if not voice_channel:
            return {"success": False, "error": "not_in_voice"}
        if not await self.ensure_voice(voice_channel):
            return {"success": False, "error": "voice_connect_failed"}
        return {"success": True, "listening": self._listener is not None}

    def remove_from_queue(self, index: int) -> dict:
        if index == -1:  # "the last one"
            index = len(self.queue)
        track = self.queue.remove(index)
        if not track:
            return {"success": False, "error": "invalid_index", "index": index}
        return {
            "success": True,
            "removed": {"title": track.title, "artist": track.artist},
        }

    def clear_queue(self) -> dict:
        count = self.queue.clear()
        return {"success": True, "cleared": count}

    async def set_volume(self, level: int) -> dict:
        level = max(0, min(100, level))
        self._volume = level / 100.0

        self._apply_volume()
        return {"success": True, "volume": level}

    async def leave_voice(self) -> dict:
        await self.stop_music()
        await self._disconnect_voice()
        self.current_track = None
        return {"success": True}

    def state_snapshot(self) -> dict:
        snap: dict = {
            "playing": (
                self.current_track is not None
                and self._voice_client is not None
                and self._voice_client.is_playing()
            )
        }
        if self.current_track:
            snap.update(
                {
                    "title": self.current_track.title,
                    "artist": self.current_track.artist,
                    "duration": self.current_track.duration_ms,
                    "paused": self._paused,
                }
            )
        snap["queue"] = self.queue.as_list()
        snap["volume"] = int(self._volume * 100)
        return snap


class _SpeechSource(discord.FFmpegPCMAudio):
    """An mp3 from voice/tts.py that deletes itself when done."""

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self._path = path

    def cleanup(self) -> None:
        super().cleanup()
        try:
            os.remove(self._path)
        except OSError:
            pass
