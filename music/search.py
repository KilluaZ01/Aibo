"""
music/search.py — YouTube search and stream resolution via yt-dlp.

Search uses ytsearch1:<query> to find the top YouTube result.
Stream resolution re-fetches a fresh audio URL immediately before
playback because YouTube stream URLs expire quickly.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional
from urllib.parse import urlparse

import yt_dlp

from music.queue import Track

log = logging.getLogger("nova.music.search")

_YDL_SEARCH_OPTS = {
    "format": "bestaudio/best",
    "noplaylist": True,
    "quiet": True,
    "no_warnings": True,
    "default_search": "ytsearch",
    "source_address": "0.0.0.0",
    "extract_flat": False,
}

_YDL_STREAM_OPTS = {
    "format": "bestaudio/best",
    "noplaylist": True,
    "quiet": True,
    "no_warnings": True,
    "source_address": "0.0.0.0",
    "extract_flat": False,
}


def _is_url(query: str) -> bool:
    try:
        result = urlparse(query)
        return result.scheme in ("http", "https")
    except Exception:
        return False


def _search_sync(query: str) -> Optional[dict]:
    """Synchronous yt-dlp search — run in thread."""
    search_query = query if _is_url(query) else f"ytsearch1:{query}"
    with yt_dlp.YoutubeDL(_YDL_SEARCH_OPTS) as ydl:
        info = ydl.extract_info(search_query, download=False)
        if not info:
            return None
        # ytsearch1 returns a dict with 'entries'
        if "entries" in info:
            entries = [e for e in info["entries"] if e]
            return entries[0] if entries else None
        return info


def _resolve_stream_sync(webpage_url: str) -> Optional[str]:
    """Re-fetch a fresh direct audio stream URL — run in thread."""
    with yt_dlp.YoutubeDL(_YDL_STREAM_OPTS) as ydl:
        info = ydl.extract_info(webpage_url, download=False)
        if not info:
            return None
        url = info.get("url")
        if url:
            return url
        # Some formats nest the URL in formats list
        formats = info.get("formats", [])
        for fmt in reversed(formats):
            if fmt.get("url") and fmt.get("acodec") != "none":
                return fmt["url"]
        return None


async def search_track(
    query: str,
    requester: Optional[str] = None,
) -> Optional[Track]:
    """
    Search YouTube and return a Track with metadata.
    Does NOT resolve the stream URL yet — that happens in resolve_stream().
    """
    log.info("[Music] YouTube search: %s", query)
    try:
        info = await asyncio.to_thread(_search_sync, query)
    except yt_dlp.utils.DownloadError as exc:
        log.error("[Music] yt-dlp search error: %s", exc)
        return None
    except Exception as exc:
        log.error("[Music] Unexpected search error: %s", exc, exc_info=True)
        return None

    if not info:
        log.info("[Music] No results for: %s", query)
        return None

    webpage_url = info.get("webpage_url") or info.get("url", "")
    title = info.get("title", "Unknown Title")
    artist = info.get("uploader") or info.get("channel") or "Unknown Artist"
    duration_ms = int(info.get("duration", 0) or 0) * 1000

    track = Track(
        title=title,
        artist=artist,
        identifier=webpage_url,
        webpage_url=webpage_url,
        duration_ms=duration_ms,
        requester=requester,
    )

    log.info("[Music] Resolved: %s — %s", track.title, track.artist)
    return track


async def resolve_stream(track: Track) -> Optional[str]:
    """
    Re-fetch a fresh direct audio stream URL immediately before playback.
    Called right before FFmpeg starts — never cache this URL.
    """
    log.info("[Music] Resolving fresh stream for: %s", track.title)
    try:
        url = await asyncio.to_thread(_resolve_stream_sync, track.webpage_url)
    except yt_dlp.utils.DownloadError as exc:
        log.error("[Music] Stream resolution error: %s", exc)
        return None
    except Exception as exc:
        log.error("[Music] Unexpected stream error: %s", exc, exc_info=True)
        return None

    if not url:
        log.error("[Music] No stream URL found for: %s", track.webpage_url)
        return None

    log.info("[Music] Stream URL obtained for: %s", track.title)
    return url
