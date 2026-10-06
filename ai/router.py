"""
ai/router.py — lightweight pre-LLM routing layer.

Determines whether a Discord message should be forwarded to the LLM.
Only messages that pass the filter reach the (expensive) AI call.

Rules (any match → process):
1. Bot is directly @mentioned.
2. Message is a reply to the bot's own message.
3. Message contains a strong music intent keyword.
4. Message is in an active conversation thread with the bot
   (i.e. bot has spoken recently in this channel).
"""

from __future__ import annotations

import re
import logging
from collections import defaultdict
from datetime import datetime, timedelta

import discord

log = logging.getLogger("nova.ai.router")

# If the bot spoke in a channel within this window, treat follow-ups as directed at it
_ACTIVE_WINDOW = timedelta(minutes=3)

# Strong music-intent patterns (case-insensitive)
_MUSIC_PATTERNS = re.compile(
    r"\b("
    r"play|queue|skip|pause|resume|stop|volume|vol|"
    r"next|previous|shuffle|repeat|loop|"
    r"now playing|what.?s playing|what.?s on|"
    r"show queue|clear queue|remove.+from queue|"
    r"turn (it |the music )?(up|down)|louder|quieter|mute|unmute|"
    r"leave vc|leave the vc|leave voice|disconnect|"
    r"put .+ on|add .+ to queue"
    r")\b",
    re.IGNORECASE,
)

# Per-channel: last time the bot sent a message { channel_id: datetime }
_bot_last_spoke: dict[int, datetime] = defaultdict(lambda: datetime.min)


def record_bot_spoke(channel_id: int) -> None:
    """Call this every time the bot sends a message in a channel."""
    _bot_last_spoke[channel_id] = datetime.utcnow()


def should_process(message: discord.Message, bot_user: discord.ClientUser) -> bool:
    """
    Return True if this message should be sent to the LLM.
    """
    # Never process own messages
    if message.author == bot_user:
        return False

    # Never process bots (including webhooks)
    if message.author.bot:
        return False

    # 1. Direct @mention
    if bot_user in message.mentions:
        log.debug("Router: mentioned — processing")
        return True

    # 2. Reply to the bot
    ref = message.reference
    if ref and ref.resolved and isinstance(ref.resolved, discord.Message):
        if ref.resolved.author == bot_user:
            log.debug("Router: reply to bot — processing")
            return True

    # 3. Strong music intent
    if _MUSIC_PATTERNS.search(message.content):
        log.debug("Router: music intent — processing")
        return True

    # 4. Active conversation window
    last = _bot_last_spoke.get(message.channel.id, datetime.min)
    if datetime.utcnow() - last < _ACTIVE_WINDOW:
        log.debug("Router: active conversation window — processing")
        return True

    log.debug("Router: ignoring message from %s", message.author)
    return False


def strip_mention(content: str, bot_user: discord.ClientUser) -> str:
    """Remove the bot's @mention from the start of a message."""
    mention_variants = [
        f"<@{bot_user.id}>",
        f"<@!{bot_user.id}>",
    ]
    result = content.strip()
    for mention in mention_variants:
        if result.startswith(mention):
            result = result[len(mention):].strip()
    return result
