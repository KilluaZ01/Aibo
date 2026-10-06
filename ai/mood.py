"""
ai/mood.py — lightweight read of how people in a guild are feeling.

Cheap keyword pass (English + romanised Nepali) on every message Aibo hears,
typed or spoken. The result is handed to the LLM as a hint so it can pick
songs and words that fit the room — the model still reads the actual text.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from datetime import datetime, timedelta

_MOOD_WORDS: dict[str, tuple[str, ...]] = {
    "sad": (
        "sad", "depressed", "crying", "cry", "lonely", "alone", "heartbroken",
        "broke up", "breakup", "miss her", "miss him", "hurts", "rough day",
        "bad day", "dukha", "udas", "runu", "eklo", "chot",
    ),
    "tired": (
        "tired", "exhausted", "sleepy", "drained", "burnt out", "burned out",
        "long day", "thakyo", "thakeko", "nidra", "nindra",
    ),
    "stressed": (
        "stressed", "stress", "anxious", "anxiety", "exam", "deadline",
        "overthinking", "tension", "pressure", "worried",
    ),
    "angry": (
        "angry", "mad", "pissed", "furious", "annoyed", "hate this",
        "rissa", "risayo", "tilted",
    ),
    "happy": (
        "happy", "excited", "hyped", "lets go", "let's go", "won", "we won",
        "great day", "good day", "love this", "khusi", "ramro", "majja",
        "maja", "party",
    ),
    "chill": (
        "chill", "relax", "relaxing", "calm", "vibe", "vibing", "lofi",
        "late night", "rain",
    ),
}

_PATTERNS = {
    mood: re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.I)
    for mood, words in _MOOD_WORDS.items()
}

_MOOD_WINDOW = timedelta(minutes=30)


def detect_mood(text: str) -> str | None:
    """Return the strongest mood keyword family in the text, if any."""
    best, hits = None, 0
    for mood, pattern in _PATTERNS.items():
        n = len(pattern.findall(text))
        if n > hits:
            best, hits = mood, n
    return best


class MoodTracker:
    """Remembers the last few mood signals per guild."""

    def __init__(self) -> None:
        # guild_id → deque of (when, who, mood, snippet)
        self._signals: dict[int, deque] = defaultdict(lambda: deque(maxlen=8))

    def observe(self, guild_id: int, who: str, text: str) -> str | None:
        mood = detect_mood(text)
        if mood:
            self._signals[guild_id].append((datetime.utcnow(), who, mood, text[:80]))
        return mood

    def describe(self, guild_id: int) -> str:
        """A short note for the system prompt, or '' when nothing stands out."""
        cutoff = datetime.utcnow() - _MOOD_WINDOW
        recent = [s for s in self._signals[guild_id] if s[0] >= cutoff]
        if not recent:
            return ""
        latest: dict[str, tuple] = {}
        for when, who, mood, snippet in recent:
            latest[who] = (when, mood, snippet)
        lines = ["[How people seem lately]"]
        for who, (when, mood, snippet) in latest.items():
            mins = int((datetime.utcnow() - when).total_seconds() // 60)
            ago = "just now" if mins < 1 else f"{mins} min ago"
            lines.append(f'- {who} sounded {mood} ({ago}): "{snippet}"')
        return "\n".join(lines)
