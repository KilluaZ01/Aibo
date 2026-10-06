"""
music/queue.py — track queue for a single guild.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Track:
    title: str
    artist: str
    identifier: str  # YouTube page URL or direct URL
    webpage_url: str  # always the YouTube watch URL — used for stream re-resolution
    duration_ms: int
    requester: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "title": self.title,
            "artist": self.artist,
            "identifier": self.identifier,
            "duration": self.duration_ms,
            "requester": self.requester,
        }

    def duration_fmt(self) -> str:
        s = self.duration_ms // 1000
        return f"{s // 60}:{s % 60:02d}"


class GuildQueue:
    def __init__(self) -> None:
        self._tracks: list[Track] = []

    def add(self, track: Track) -> int:
        self._tracks.append(track)
        return len(self._tracks)

    def pop_next(self) -> Optional[Track]:
        return self._tracks.pop(0) if self._tracks else None

    def remove(self, index: int) -> Optional[Track]:
        real = index - 1
        if 0 <= real < len(self._tracks):
            return self._tracks.pop(real)
        return None

    def clear(self) -> int:
        count = len(self._tracks)
        self._tracks.clear()
        return count

    def __len__(self) -> int:
        return len(self._tracks)

    def __bool__(self) -> bool:
        return bool(self._tracks)

    def peek(self) -> Optional[Track]:
        return self._tracks[0] if self._tracks else None

    def as_list(self) -> list[dict]:
        return [t.as_dict() for t in self._tracks]
