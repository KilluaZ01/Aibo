"""
discord_bot/events.py — Discord event handler registration.

Typed messages and spoken voice commands go through the same
handle_request(), so "aibo play K" works the same either way.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

import discord

from config import settings
from ai.router import should_process, strip_mention, record_bot_spoke
from ai.client import process_message, comment_on_track
from ai.mood import MoodTracker
from discord_bot.context import ConversationContext
from music.manager import MusicManager
from voice.listener import VoiceListener
from voice import stt

log = logging.getLogger("nova.discord.events")

# Talking about a song when the queue moves on is nice once in a while,
# annoying every three minutes.
SONG_COMMENT_COOLDOWN = timedelta(minutes=20)


def register_events(
    bot: discord.Client,
    music_manager: MusicManager,
    context: ConversationContext,
) -> None:
    moods = MoodTracker()
    # guild_id → text channel Aibo should talk in (last place someone talked to it)
    reply_channels: dict[int, discord.abc.Messageable] = {}
    last_comment: dict[int, datetime] = {}

    def reply_channel_for(guild: discord.Guild, fallback=None):
        player = music_manager.get_player(guild.id)
        return reply_channels.get(guild.id) or fallback or (player and player.voice_channel)

    async def send(channel, text: str) -> None:
        if not channel or not text:
            return
        await channel.send(text[:1900], allowed_mentions=discord.AllowedMentions.none())
        record_bot_spoke(channel.id)

    # One request at a time per guild, in the order they were said: otherwise a
    # slow "play X" can finish after a quick "stop" and start the music again.
    guild_locks: dict[int, asyncio.Lock] = {}

    async def handle_request(
        guild: discord.Guild,
        author: discord.Member,
        text: str,
        channel,
        spoken: bool,
        guesses: Optional[list[str]] = None,
        named: bool = False,
    ) -> None:
        lock = guild_locks.setdefault(guild.id, asyncio.Lock())
        async with lock:
            await _handle_request(guild, author, text, channel, spoken, guesses, named)

    async def _handle_request(
        guild: discord.Guild,
        author: discord.Member,
        text: str,
        channel,
        spoken: bool,
        guesses: Optional[list[str]] = None,
        named: bool = False,
    ) -> None:
        voice_channel: Optional[discord.VoiceChannel] = None
        if isinstance(author, discord.Member) and author.voice:
            voice_channel = author.voice.channel

        async def tool_executor(tool_name: str, tool_input: dict) -> dict:
            return await music_manager.execute_tool(
                tool_name=tool_name,
                tool_input=tool_input,
                guild=guild,
                voice_channel=voice_channel,
                requester=author.display_name,
            )

        moods.observe(guild.id, author.display_name, text)
        how = "said in voice chat" if spoken else "wrote"
        context.add_user(guild.id, f"{author.display_name} ({how}): {text}")

        player = music_manager.get_player(guild.id)
        music_state = player.state_snapshot() if player else None

        announced: list[str] = []

        async def announce(line: str) -> None:
            """Spoken requests: say the reply while the song is still loading."""
            # Mark it said up front: the final reply is ready before the voice
            # line finishes generating, and must not say it a second time.
            announced.append(line)
            p = music_manager.get_player(guild.id)
            if not (p and await p.speak(line)):
                announced.remove(line)

        try:
            response = await process_message(
                conversation=context.get(guild.id),
                user_text=text,
                music_state=music_state,
                mood_note=moods.describe(guild.id),
                tool_executor=tool_executor,
                guesses=guesses,
                announce=announce if spoken else None,
                # Spoken: only touch the music when they used a clear command
                # word ("Milo play…", "Milo volume…"). Even after the name, a
                # stray word ("Milo… my") must not become a song.
                require_command_words=spoken,
            )
        except Exception as exc:
            log.error("LLM processing error: %s", exc, exc_info=True)
            response = "Something went wrong on my end. Try again."

        if response:
            context.add_assistant(guild.id, response)
            # Asked out loud → answer out loud too (the text stays as a record),
            # unless that line was already said while the song loaded.
            already_said = any(line in response for line in announced)
            if spoken and not already_said:
                player = music_manager.get_player(guild.id)
                if player:
                    await player.speak(response)
            if spoken:
                response = f"🎙️ *{author.display_name}: \"{text}\"*\n{response}"
            await send(channel, response)

    def make_listener(player) -> VoiceListener:
        guild = player.guild

        async def on_voice_command(
            member: discord.Member, text: str, guesses: list[str], named: bool
        ) -> None:
            channel = reply_channel_for(guild, fallback=player.voice_channel)
            await handle_request(
                guild, member, text, channel, spoken=True, guesses=guesses, named=named
            )

        return VoiceListener(on_command=on_voice_command, duck=player.duck)

    async def on_track_start(player, track) -> None:
        if not settings.song_comments:
            return
        gid = player.guild.id
        now = datetime.utcnow()
        if now - last_comment.get(gid, datetime.min) < SONG_COMMENT_COOLDOWN:
            return
        last_comment[gid] = now
        line = await comment_on_track(track.title, track.artist, moods.describe(gid))
        if line:
            await send(reply_channel_for(player.guild), f"🎵 {line}")

    music_manager.set_hooks(listener_factory=make_listener, on_track_start=on_track_start)

    @bot.event
    async def on_ready() -> None:
        log.info("[Discord] Logged in as %s (ID: %d)", bot.user.name, bot.user.id)
        await bot.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name=f'"{settings.bot_name}, play…"',
            )
        )
        await music_manager.start()
        if settings.voice_listen:
            await stt.warm_up()

    @bot.event
    async def on_message(message: discord.Message) -> None:
        if not message.guild:
            return

        if not should_process(message, bot.user):
            return

        user_text = strip_mention(message.content, bot.user).strip()
        if not user_text:
            return

        reply_channels[message.guild.id] = message.channel
        async with message.channel.typing():
            await handle_request(
                message.guild, message.author, user_text, message.channel, spoken=False
            )

    @bot.event
    async def on_voice_state_update(
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        # Auto-disconnect when everyone leaves the voice channel
        if member.id == bot.user.id:
            return

        if not member.guild:
            return

        player = music_manager.get_player(member.guild.id)
        if not player or not player._voice_client:
            return

        vc = player._voice_client
        if not vc.is_connected():
            return

        # If the bot is alone in the channel, leave
        channel = vc.channel
        non_bot_members = [m for m in channel.members if not m.bot]
        if not non_bot_members:
            log.info("[Music] Everyone left — disconnecting from '%s'", channel.name)
            await player.leave_voice()
