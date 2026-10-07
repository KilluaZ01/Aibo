"""
ai/client.py — Milo's brain.

NVIDIA's OpenAI-compatible API first (NVIDIA_API_KEY), Hugging Face
Inference as a fallback (HUGGINGFACE_TOKEN). The model answers with one
JSON object that names a music action plus what to say; clear commands
skip the LLM entirely (ai/intent.py).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable

import aiohttp
from huggingface_hub import AsyncInferenceClient

from config import settings
from ai.intent import parse_command
from ai.prompts import (
    ACTION_RULES,
    COMMENT_RULES,
    CHAT_RULES,
    PERSONALITY,
    build_music_context,
)

log = logging.getLogger("nova.ai.client")

# Without a timeout one stuck provider hangs that request forever; on timeout
# the next model in the chain gets a turn.
LLM_TIMEOUT = 15  # seconds
_client = AsyncInferenceClient(token=settings.hf_token or None, timeout=LLM_TIMEOUT)
_THINK = re.compile(r"<think>.*?</think>", re.S)


async def _nvidia_chat(model: str, messages: list[dict], max_tokens: int, temperature: float) -> str:
    """One call to NVIDIA's OpenAI-compatible /chat/completions."""
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Nemotron "thinks" before answering by default: great for puzzles,
        # seconds too slow for a voice chat.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Authorization": f"Bearer {settings.nvidia_api_key}"}
    timeout = aiohttp.ClientTimeout(total=LLM_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            f"{settings.nvidia_base_url}/chat/completions", json=body, headers=headers
        ) as resp:
            data = await resp.json(content_type=None)
            if resp.status != 200:
                detail = (data or {}).get("error") or (data or {}).get("detail") or data
                raise RuntimeError(f"NVIDIA {resp.status}: {str(detail)[:200]}")
    content = data["choices"][0]["message"].get("content") or ""
    return _THINK.sub("", content).strip()

# action → (tool name in MusicManager.execute_tool)
ACTION_TO_TOOL = {
    "play": "play_song",
    "play_now": "play_now",
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


async def call_llm(
    messages: list[dict],
    max_tokens: int = 160,
    temperature: float = 0.8,
    models: tuple[str, ...] | None = None,
) -> str | None:
    if not _quota_ok():
        log.warning("[AI] HF_CALLS_PER_MIN reached — skipping LLM call")
        return None
    global _last_failure
    if settings.nvidia_api_key:
        for model in settings.nvidia_models:
            started = time.monotonic()
            try:
                text = await _nvidia_chat(model, messages, max_tokens, temperature)
                log.info("[AI] %s answered in %.1fs", model, time.monotonic() - started)
                if text:
                    return text
            except Exception as exc:
                _last_failure = f"{type(exc).__name__}: {str(exc)[:200]}"
                log.warning("[AI] %s failed: %s", model, _last_failure)
    if not settings.hf_token:
        return None
    for model in models or settings.hf_models:
        started = time.monotonic()
        try:
            resp = await _client.chat_completion(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = (resp.choices[0].message.content or "").strip()
            log.info("[AI] %s answered in %.1fs", model, time.monotonic() - started)
            if text:
                return text
        except Exception as exc:
            text_exc = str(exc)
            if "402" in text_exc:
                _last_failure = "Hugging Face credits are used up (402 Payment Required)"
            elif "not a chat model" in text_exc or "model_not_supported" in text_exc:
                _last_failure = f"{model} isn't available for chat on Hugging Face"
            else:
                _last_failure = text_exc[:200]
            log.warning("[AI] %s failed: %s", model, _last_failure)
    return None


# Said when every model failed (usually: Hugging Face credits used up).
_NO_AI_LINE = "My AI brain is offline right now. Skip, stop, pause and volume still work."

# Remember why the last call failed so the log says it plainly.
_last_failure = ""


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
    return {"intent": "chat", "reply": text}


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
    "not_paused": "It's already playing. Say the song name with play to change it.",
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
    if action == "play_now":
        return _track_line(result)
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
    if action in ("play", "play_now", "queue"):
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


# Words that must be in a spoken phrase (without the bot's name) before an
# action may change the music. Looking/asking (now_playing, show_queue) is free.
_COMMAND_WORDS = {
    "play": r"play|put on|put|queue|add|bajau|bajaideu|bajaa|baja|lagau|lagaideu",
    "volume": r"volume|louder|quieter|loud|turn (it |the music )?(up|down)|increase|decrease|lower|raise|mute",
    "skip": r"skip|next|change",
    # "tap"/"top": speech-to-text often clips the s off "stop" over music
    "stop": r"stop|tap|top",
    "pause": r"pause|hold",
    "resume": r"resume|continue|unpause|play",
    "join": r"join|come",
    "leave": r"leave|disconnect|get out",
    "remove": r"remove|delete",
    "clear_queue": r"clear|empty",
}
_COMMAND_WORDS["play_now"] = _COMMAND_WORDS["queue"] = _COMMAND_WORDS["play"]


def _has_command_words(action: str, texts: list[str]) -> bool:
    pattern = _COMMAND_WORDS.get(action)
    if pattern is None:
        return True  # chat, now_playing, show_queue…
    regex = re.compile(rf"\b({pattern})\b", re.I)
    return any(regex.search(t) for t in texts)


async def _chat_only(
    conversation: list[dict],
    user_text: str,
    music_state: dict | None,
    mood_note: str,
    guesses: list[str] | None,
) -> str:
    """Re-ask for a plain chat reply when the first answer was a music action."""
    messages = [{"role": "system", "content": _system(music_state, mood_note, CHAT_RULES)}]
    messages += conversation[:-1]
    messages.append({"role": "user", "content": user_text})
    return _clean_line(await call_llm(messages, max_tokens=80, temperature=0.8)) or "I'm here."


_PLAY_START = re.compile(r"^(?:play|put on|queue|bajau)\s+(?P<rest>.+)$", re.I)


# intent from the model → action we run
_INTENT_ALIASES = {
    "set_volume": "volume",
    "volume_up": "volume",
    "volume_down": "volume",
}


def _decision_to_action(decision: dict, current_volume: int) -> tuple[str, dict]:
    intent = str(decision.get("intent") or decision.get("action") or "chat").strip().lower()
    data = dict(decision)
    try:
        amount = int(decision.get("amount") or decision.get("level") or 0)
    except (TypeError, ValueError):
        amount = 0
    if intent == "volume_up":
        data["level"] = min(100, current_volume + (amount or 20))
    elif intent == "volume_down":
        data["level"] = max(0, current_volume - (amount or 20))
    elif intent == "set_volume":
        data["level"] = max(0, min(100, amount))
    return _INTENT_ALIASES.get(intent, intent), data


def _request_message(user_text: str, guesses: list[str] | None) -> dict:
    lines = [f'Request: "{user_text}"']
    if guesses and len(guesses) > 1:
        shown = " | ".join(guesses[:5])
        lines.append(f"(said out loud — speech recognition guesses: {shown})")
    elif guesses:
        lines.append("(said out loud — may contain speech recognition mistakes)")
    lines.append("Answer with the JSON object only.")
    return {"role": "user", "content": "\n".join(lines)}


async def process_message(
    conversation: list[dict],
    user_text: str,
    music_state: dict | None,
    mood_note: str,
    tool_executor: ToolExecutor,
    guesses: list[str] | None = None,
    announce: Callable[[str], Awaitable[Any]] | None = None,
    require_command_words: bool = False,
    allow_llm: bool = True,
) -> str:
    """Turn one request (typed or spoken) into an action + what Milo says.

    Commands (play X, skip, stop, pause, volume up…) never reach the LLM. Only
    what the parser doesn't understand does, and only when allow_llm is set
    (the bot was addressed by name, @mention or reply).
    """
    volume = (music_state or {}).get("volume", 80)

    # Fast path: clear commands ("skip", "pause", "volume up", "play X") need no LLM.
    cmd = parse_command(user_text, volume)
    if cmd and require_command_words and not _has_command_words(cmd["action"], [user_text]):
        cmd = None  # casual "wait" / "bye" in conversation shouldn't touch the music
    if cmd:
        action = cmd["action"]
        result = await _run(action, cmd, tool_executor)
        if action in ("play", "queue") and result.get("success"):
            return _track_line(result)
        return _canned(action, result)

    if not allow_llm:
        return ""

    # Everything else: the model classifies the intent, cleans up the song
    # name and writes the reply. The last conversation entry is this request,
    # so it's replaced with a version that carries the speech guesses.
    messages = [{"role": "system", "content": _system(music_state, mood_note, ACTION_RULES)}]
    messages += conversation[:-1]
    messages.append(_request_message(user_text, guesses))
    raw = await call_llm(
        messages, max_tokens=200, temperature=0.3, models=settings.hf_intent_models or None
    )
    if not raw:
        if require_command_words:
            # Overheard chat while the AI is down: say nothing rather than
            # announcing the outage after every sentence in the call.
            return ""
        return _NO_AI_LINE

    decision = _parse_decision(raw)
    action, data = _decision_to_action(decision, volume)

    # Small models sometimes misfile a plain "play X" (e.g. as volume). If the
    # request literally starts with play, it's a song request.
    m = _PLAY_START.match(user_text.strip())
    if m and action not in ("play", "play_now", "queue") and m.group("rest").strip():
        data["query"] = str(decision.get("query") or "").strip() or m.group("rest").strip()
        log.info("[AI] %r starts with play but came back as %s — playing %r",
                 user_text, action, data["query"])
        action = "play"
    reply = _clean_line(decision.get("reply"))
    log.info("[AI] Understood %r as %s %s", user_text, action, data.get("query") or data.get("level") or "")

    if action == "ignore":
        return ""
    if require_command_words and not _has_command_words(action, [user_text] + (guesses or [])):
        # The AI wants to change the music, but nobody said "Milo" or a clear
        # command word ("hello hello" → Adele, "tap tap" → Tory Lanez). Just talk.
        log.info("[AI] %s needs a command word or the name — treating as chat", action)
        return await _chat_only(conversation, user_text, music_state, mood_note, guesses)
    if action in ("chat", "none") or action not in ACTION_TO_TOOL:
        return reply or "I'm here."

    if announce and reply and action in ("play", "play_now", "queue"):
        # Say "Iris, coming up" while YouTube is still searching, instead of
        # after the song has started. The song waits for the sentence to end.
        asyncio.create_task(announce(reply))

    result = await _run(action, data, tool_executor)
    if not result.get("success"):
        return _canned(action, result)
    if action in ("play", "play_now", "queue"):
        return f"{_track_line(result)}\n{reply}" if reply else _track_line(result)
    if action in ("now_playing", "show_queue"):
        # Real data beats whatever the model guessed.
        return _canned(action, result)
    return reply or _canned(action, result)


async def comment_on_track(track_title: str, track_artist: str, mood_note: str) -> str | None:
    """One short line when the queue moves on to a new song."""
    messages = [
        {"role": "system", "content": f"{PERSONALITY}\n\n{mood_note}\n\n{COMMENT_RULES}"},
        {"role": "user", "content": f"Now starting: {track_title} — {track_artist}"},
    ]
    return _clean_line(await call_llm(messages, max_tokens=50, temperature=0.9), limit=200)
