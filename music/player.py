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
from typing import Optional

import discord

from music.queue import GuildQueue, Track
from music.search import search_track, resolve_stream

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
    ) -> None:
        self.guild = guild
        self._bot = bot

        self.queue = GuildQueue()
        self.current_track: Optional[Track] = None
        self._voice_client: Optional[discord.VoiceClient] = None
        self._volume: float = 0.8  # 0.0 – 1.0
        self._paused: bool = False

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

            self._voice_client = await channel.connect(self_deaf=True)
            log.info(
                "[Music] Connected to voice channel '%s' in guild %d",
                channel.name,
                self.guild.id,
            )
            return True

        except discord.DiscordException as exc:
            log.error("[Music] Failed to connect to voice: %s", exc)
            return False

    async def _disconnect_voice(self) -> None:
        if self._voice_client and self._voice_client.is_connected():
            await self._voice_client.disconnect()
        self._voice_client = None

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
            source = discord.PCMVolumeTransformer(source, volume=self._volume)
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

        self._voice_client.play(source, after=after_playing)
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

        next_track = self.queue.pop_next()
        if next_track:
            log.info("[Music] Auto-playing next: %s", next_track.title)
            await self._play_track(next_track)
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

        # If already playing, queue it
        if self._voice_client.is_playing() and not self._paused:
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

        # If nothing playing, start immediately
        if not self._voice_client.is_playing():
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

    async def skip_song(self) -> dict:
        if not self._voice_client or not self._voice_client.is_playing():
            return {"success": False, "error": "nothing_playing"}

        skipped = self.current_track
        self._voice_client.stop()  # triggers after_playing → _on_track_end

        return {
            "success": True,
            "action": "skipped",
            "skipped": {
                "title": skipped.title if skipped else "Unknown",
                "artist": skipped.artist if skipped else "Unknown",
            },
        }

    async def pause_music(self) -> dict:
        if not self._voice_client or not self._voice_client.is_playing():
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
        if self._voice_client and self._voice_client.is_playing():
            self._voice_client.stop()
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

    def remove_from_queue(self, index: int) -> dict:
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

        if (
            self._voice_client
            and self._voice_client.source
            and isinstance(self._voice_client.source, discord.PCMVolumeTransformer)
        ):
            self._voice_client.source.volume = self._volume

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
