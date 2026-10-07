"""
music/manager.py — MusicManager

Manages per-guild GuildPlayer instances.
Lavalink is no longer used — voice is handled directly by discord.py.
"""

from __future__ import annotations

import logging
from typing import Optional

import discord

from music.player import GuildPlayer

log = logging.getLogger("nova.music.manager")


class MusicManager:
    def __init__(self, bot: discord.Client, listener_factory=None, on_track_start=None) -> None:
        self._bot = bot
        self._players: dict[int, GuildPlayer] = {}
        self._listener_factory = listener_factory
        self._on_track_start = on_track_start

    def set_hooks(self, listener_factory=None, on_track_start=None) -> None:
        """Voice listener + song-change hooks; used by players created after this."""
        self._listener_factory = listener_factory
        self._on_track_start = on_track_start

    async def start(self) -> None:
        """No-op — no Lavalink connection needed."""
        log.info("[Music] MusicManager ready (yt-dlp + FFmpeg mode)")

    async def close(self) -> None:
        """Disconnect all voice clients on shutdown."""
        for player in self._players.values():
            try:
                await player.leave_voice()
            except Exception:
                pass

    def get_or_create_player(self, guild: discord.Guild) -> GuildPlayer:
        if guild.id not in self._players:
            player = GuildPlayer(
                guild,
                self._bot,
                listener_factory=self._listener_factory,
                on_track_start=self._on_track_start,
            )
            self._players[guild.id] = player
            log.info(
                "[Music] Created player for guild %d (%s)",
                guild.id,
                guild.name,
            )
        return self._players[guild.id]

    def get_player(self, guild_id: int) -> Optional[GuildPlayer]:
        return self._players.get(guild_id)

    async def execute_tool(
        self,
        tool_name: str,
        tool_input: dict,
        guild: discord.Guild,
        voice_channel: Optional[discord.VoiceChannel],
        requester: Optional[str],
    ) -> dict:
        player = self.get_or_create_player(guild)

        if tool_name == "play_song":
            query = str(tool_input.get("query", "")).strip()
            if not query:
                return {"success": False, "error": "empty_query"}
            return await player.play_song(query, requester, voice_channel)

        elif tool_name == "play_now":
            query = str(tool_input.get("query", "")).strip()
            if not query:
                return {"success": False, "error": "empty_query"}
            return await player.play_now(query, requester, voice_channel)

        elif tool_name == "queue_song":
            query = str(tool_input.get("query", "")).strip()
            if not query:
                return {"success": False, "error": "empty_query"}
            return await player.queue_song(query, requester, voice_channel)

        elif tool_name == "skip_song":
            return await player.skip_song()

        elif tool_name == "pause_music":
            return await player.pause_music()

        elif tool_name == "resume_music":
            return await player.resume_music()

        elif tool_name == "stop_music":
            return await player.stop_music()

        elif tool_name == "now_playing":
            return player.now_playing()

        elif tool_name == "show_queue":
            return player.show_queue()

        elif tool_name == "remove_from_queue":
            try:
                index = int(tool_input.get("index", 0))
            except (TypeError, ValueError):
                return {"success": False, "error": "invalid_index"}
            return player.remove_from_queue(index)

        elif tool_name == "clear_queue":
            return player.clear_queue()

        elif tool_name == "set_volume":
            try:
                level = int(tool_input.get("level", 80))
            except (TypeError, ValueError):
                return {"success": False, "error": "invalid_volume"}
            return await player.set_volume(level)

        elif tool_name == "join_voice":
            return await player.join_voice(voice_channel)

        elif tool_name == "leave_voice":
            return await player.leave_voice()

        else:
            log.warning("Unknown tool requested: %s", tool_name)
            return {"success": False, "error": "unknown_tool", "tool": tool_name}
