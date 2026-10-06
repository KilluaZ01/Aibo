"""
ai/client.py — Aibo's brain on Hugging Face Inference.

Uses the same HUGGINGFACE_TOKEN and model fallback chain as the
Nima/Arik/Zidan bots. Small HF models don't do native tool calling
reliably, so the model answers with one JSON object that names a music
action plus what to say; clear commands skip the LLM entirely (ai/intent.py).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable

from huggingface_hub import AsyncInferenceClient

from config import settings
from ai.intent import parse_command
from ai.prompts import (
    ACTION_RULES,
    COMMENT_RULES,
    PERSONALITY,
    PLAYED_RULES,
    build_music_context,
)

log = logging.getLogger("nova.ai.client")

_client = AsyncInferenceClient(token=settings.hf_token)

# action → (tool name in MusicManager.execute_tool)
ACTION_TO_TOOL = {
    "play": "play_song",
    "queue": "queue_song",
    "skip": "skip_song",
    "pause": "pause_music",
    "resume": "resume_music",
    "stop": "stop_music",
    "leave": "leave_voice",
    "join": "join_voice",
    "now_playing": "now_playing",
    "show_queue": "show_queue",
    "clear_queue": "clear_queue",
    "volume": "set_volume",
    "remove": "remove_from_queue",
}

ToolExecutor = Callable[[str, dict], Awaitable[Any]]

# ----------------------------------------------------------------------
# HF call with the same per-minute cap the other bots use
# ----------------------------------------------------------------------

_call_times: list[datetime] = []


def _quota_ok() -> bool:
    cutoff = datetime.now() - timedelta(minutes=1)
    while _call_times and _call_times[0] < cutoff:
        _call_times.pop(0)
    if len(_call_times) >= settings.hf_calls_per_min:
        return False
    _call_times.append(datetime.now())
    return True


async def call_llm(messages: list[dict], max_tokens: int = 160, temperature: float = 0.8) -> str | None:
    if not _quota_ok():
        log.warning("[AI] HF_CALLS_PER_MIN reached — skipping LLM call")
        return None
    for model in settings.hf_models:
        try:
            resp = await _client.chat_completion(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception as exc:
            log.warning("[AI] %s failed: %s", model, str(exc)[:200])
    return None


def _clean_line(text: str | None, limit: int = 300) -> str | None:
    if not text:
        return None
    text = text.strip().strip('"').strip()
    # Models sometimes prefix their own name.
    text = re.sub(rf"^{re.escape(settings.bot_name)}\s*:\s*", "", text, flags=re.I)
    text = text.split("\n")[0].strip()
    return text[:limit] or None


def _parse_decision(text: str) -> dict:
    """Pull the first JSON object out of the model's answer."""
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    # Not JSON — treat it as plain talk.
    return {"action": "none", "reply": text}


# ----------------------------------------------------------------------
# Result formatting
# ----------------------------------------------------------------------

_ERRORS = {
    "not_in_voice": "Hop in a voice channel first and I'll come to you.",
    "voice_connect_failed": "I couldn't get into the voice channel. Check I have Connect + Speak there.",
    "no_results": "Couldn't find that one. Try other words?",
    "stream_failed": "Found it but couldn't stream it. Try another version or song?",
    "ffmpeg_error": "My audio player tripped. Try again in a sec.",
    "nothing_playing": "Nothing's playing right now.",
    "not_paused": "It's not paused.",
    "invalid_index": "There's no song at that spot in the queue.",
    "empty_query": "Play what though?",
}


def _track_line(result: dict) -> str:
    title = result.get("title", "?")
    artist = result.get("artist", "?")
    if result.get("action") == "queued":
        return f"➕ **{title}** — {artist} (#{result.get('queue_position', '?')} in queue)"
    return f"🎵 **{title}** — {artist}"


def _canned(action: str, result: dict) -> str:
    if not result.get("success"):
        return _ERRORS.get(result.get("error", ""), "That didn't work. Try again?")
    if action == "skip":
        return "Skipped."
    if action == "pause":
        return "Paused."
    if action == "resume":
        return "Back on."
    if action == "stop":
        return "Stopped and cleared the queue."
    if action == "leave":
        return "Peace ✌️"
    if action == "join":
        if result.get("listening"):
            return f"I'm in. Say \"{settings.bot_name}\" and tell me what to play 🎧"
        return "I'm in. Type what you want to hear."
    if action == "clear_queue":
        return f"Queue cleared ({result.get('cleared', 0)} songs)."
    if action == "volume":
        return f"Volume {result.get('volume')}."
    if action == "remove":
        r = result.get("removed", {})
        return f"Removed **{r.get('title', '?')}**."
    if action == "now_playing":
        if not result.get("playing"):
            return "Nothing's playing right now."
        state = " (paused)" if result.get("paused") else ""
        return f"🎵 **{result.get('title')}** — {result.get('artist')}{state}"
    if action == "show_queue":
        queue = result.get("queue", [])
        if not queue:
            return "Queue's empty."
        lines = [f"{i}. {t.get('title')} — {t.get('artist')}" for i, t in enumerate(queue[:10], 1)]
        if len(queue) > 10:
            lines.append(f"…and {len(queue) - 10} more")
        return "\n".join(lines)
    return "Done."


async def _run(action: str, data: dict, executor: ToolExecutor) -> dict:
    tool = ACTION_TO_TOOL.get(action)
    if not tool:
        return {"success": False, "error": "unknown_tool"}
    tool_input = {}
    if action in ("play", "queue"):
        tool_input["query"] = str(data.get("query") or "").strip()
    elif action == "volume":
        tool_input["level"] = data.get("level", 80)
    elif action == "remove":
        tool_input["index"] = data.get("index", 0)
    log.info("[AI] Action: %s %s", action, tool_input)
    try:
        return await executor(tool, tool_input)
    except Exception as exc:
        log.error("Action %s raised", action, exc_info=True)
        return {"success": False, "error": str(exc)}


# ----------------------------------------------------------------------
# Public entry points
# ----------------------------------------------------------------------


def _system(music_state: dict | None, mood_note: str, extra: str) -> str:
    parts = [PERSONALITY, build_music_context(music_state)]
    if mood_note:
        parts.append(mood_note)
    parts.append(extra)
    return "\n\n".join(parts)


async def process_message(
    conversation: list[dict],
    user_text: str,
    music_state: dict | None,
    mood_note: str,
    tool_executor: ToolExecutor,
) -> str:
    """Turn one request (typed or spoken) into an action + what Aibo says."""
    volume = (music_state or {}).get("volume", 80)

    # Fast path: obvious commands need no LLM to understand.
    cmd = parse_command(user_text, volume)
    if cmd:
        action = cmd["action"]
        result = await _run(action, cmd, tool_executor)
        if action in ("play", "queue") and result.get("success"):
            line = await _line_after_action(user_text, result, music_state, mood_note)
            return f"{_track_line(result)}\n{line}" if line else _track_line(result)
        return _canned(action, result)

    # Everything else: let the model read the feeling and decide.
    messages = [{"role": "system", "content": _system(music_state, mood_note, ACTION_RULES)}]
    messages += conversation
    raw = await call_llm(messages, max_tokens=200, temperature=0.7)
    if not raw:
        return "My brain's rate-limited for a sec. Try again in a minute?"

    decision = _parse_decision(raw)
    action = str(decision.get("action") or "none").lower()
    reply = _clean_line(decision.get("reply"))

    if action == "ignore":
        return ""
    if action == "none" or action not in ACTION_TO_TOOL:
        return reply or "I'm here."

    result = await _run(action, decision, tool_executor)
    if not result.get("success"):
        return _canned(action, result)
    if action in ("play", "queue"):
        return f"{_track_line(result)}\n{reply}" if reply else _track_line(result)
    if action in ("now_playing", "show_queue"):
        # Real data beats whatever the model guessed.
        return _canned(action, result)
    return reply or _canned(action, result)


async def _line_after_action(
    user_text: str, result: dict, music_state: dict | None, mood_note: str
) -> str | None:
    messages = [
        {"role": "system", "content": _system(music_state, mood_note, PLAYED_RULES)},
        {
            "role": "user",
            "content": (
                f'They said: "{user_text}"\n'
                f"Result: {result.get('action')} {result.get('title')} by {result.get('artist')}"
            ),
        },
    ]
    return _clean_line(await call_llm(messages, max_tokens=60, temperature=0.9), limit=200)


async def comment_on_track(track_title: str, track_artist: str, mood_note: str) -> str | None:
    """One short line when the queue moves on to a new song."""
    messages = [
        {"role": "system", "content": f"{PERSONALITY}\n\n{mood_note}\n\n{COMMENT_RULES}"},
        {"role": "user", "content": f"Now starting: {track_title} — {track_artist}"},
    ]
    return _clean_line(await call_llm(messages, max_tokens=50, temperature=0.9), limit=200)
