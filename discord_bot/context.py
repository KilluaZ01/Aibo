"""
discord_bot/context.py — bounded per-guild conversation history.

Each guild gets a sliding window of the last N message exchanges.
This gives the LLM enough context to understand pronoun references
("skip it", "put Apocalypse after it") without sending unbounded history.
"""

from __future__ import annotations

from collections import defaultdict, deque
from config import settings


class ConversationContext:
    """Stores bounded {role, content} message lists per guild."""

    def __init__(self) -> None:
        # guild_id → deque of {role: str, content: str}
        self._history: dict[int, deque[dict]] = defaultdict(
            lambda: deque(maxlen=settings.max_context_messages)
        )

    def add_user(self, guild_id: int, content: str) -> None:
        self._history[guild_id].append({"role": "user", "content": content})

    def add_assistant(self, guild_id: int, content: str) -> None:
        self._history[guild_id].append({"role": "assistant", "content": content})

    def get(self, guild_id: int) -> list[dict]:
        """Return current conversation window as a list (most recent last)."""
        return list(self._history[guild_id])

    def clear(self, guild_id: int) -> None:
        self._history[guild_id].clear()
