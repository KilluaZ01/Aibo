"""
discord_bot/events.py — Discord event handler registration.
"""

from __future__ import annotations

import logging
from typing import Optional

import discord

from config import settings
from ai.router import should_process, strip_mention, record_bot_spoke
from ai.client import process_message
from discord_bot.context import ConversationContext
from music.manager import MusicManager

log = logging.getLogger("nova.discord.events")


def register_events(
    bot: discord.Client,
    music_manager: MusicManager,
    context: ConversationContext,
) -> None:

    @bot.event
    async def on_ready() -> None:
        log.info("[Discord] Logged in as %s (ID: %d)", bot.user.name, bot.user.id)
        await bot.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name="your requests",
            )
        )
        await music_manager.start()

    @bot.event
    async def on_message(message: discord.Message) -> None:
        if not message.guild:
            return

        if not should_process(message, bot.user):
            return

        guild = message.guild
        channel = message.channel

        user_text = strip_mention(message.content, bot.user).strip()
        if not user_text:
            return

        voice_channel: Optional[discord.VoiceChannel] = None
        if isinstance(message.author, discord.Member) and message.author.voice:
            voice_channel = message.author.voice.channel

        async def tool_executor(tool_name: str, tool_input: dict) -> dict:
            return await music_manager.execute_tool(
                tool_name=tool_name,
                tool_input=tool_input,
                guild=guild,
                voice_channel=voice_channel,
                requester=str(message.author),
            )

        context.add_user(guild.id, user_text)
        conversation = context.get(guild.id)

        player = music_manager.get_player(guild.id)
        music_state = player.state_snapshot() if player else None

        async with channel.typing():
            try:
                response = await process_message(
                    conversation=conversation,
                    music_state=music_state,
                    tool_executor=tool_executor,
                )
            except Exception as exc:
                log.error("LLM processing error: %s", exc, exc_info=True)
                response = "Something went wrong on my end. Try again."

        if response:
            context.add_assistant(guild.id, response)
            record_bot_spoke(channel.id)
            await channel.send(response)

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
