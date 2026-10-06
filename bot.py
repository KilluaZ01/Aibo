"""
bot.py — Aibo Discord bot entrypoint.
"""

import asyncio
import logging

import discord

from utils.logging import setup_logging

setup_logging()

from config import settings
from music.manager import MusicManager
from discord_bot.events import register_events
from discord_bot.context import ConversationContext

log = logging.getLogger("nova.bot")


def create_bot() -> discord.Client:
    intents = discord.Intents.default()
    intents.message_content = True
    intents.voice_states = True
    intents.guilds = True
    return discord.Client(intents=intents)


async def main() -> None:
    log.info("Starting %s...", settings.bot_name)

    bot = create_bot()
    music_manager = MusicManager(bot)
    context = ConversationContext()

    register_events(bot, music_manager, context)

    try:
        await bot.start(settings.discord_token)
    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        await music_manager.close()
        if not bot.is_closed():
            await bot.close()
        log.info("Goodbye.")


if __name__ == "__main__":
    asyncio.run(main())
