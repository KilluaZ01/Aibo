"""
ai/intent.py — deterministic fast path for clear music commands.

"skip", "pause", "volume 30", "play K by Cigarettes After Sex" don't need an
LLM to understand. Handling them here keeps voice commands snappy and saves
the free Hugging Face quota for the requests that actually need judgement
("play something for a rainy night", "I'm sad").

parse_command() returns an action dict or None when the LLM should decide.
"""

from __future__ import annotations

import re

_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "last": -1,
}

# A "play X" where X is a mood or a vague ask goes to the LLM to pick a song.
_VAGUE = re.compile(
    r"\b(something|anything|some|a song|songs|music|vibe|vibes|mood|"
    r"like|similar|kind of|kinda|sad|happy|chill|hype|romantic|you pick|your choice)\b",
    re.I,
)

_FILLER = re.compile(
    r"^(hey |yo |ok |okay |bro |bruh |pls |please |can you |could you |will you )+", re.I
)


def _clean(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[!?.,]+$", "", text)
    return _FILLER.sub("", text).strip()


def parse_command(text: str, current_volume: int = 80) -> dict | None:
    t = _clean(text)
    if not t:
        return None

    if re.fullmatch(r"(skip|next|next song|skip (it|this|this one|this song)|nah skip( this| it)?)", t):
        return {"action": "skip"}
    if re.fullmatch(r"(pause|pause (it|the music|this)|hold (on|the music)|wait)", t):
        return {"action": "pause"}
    if re.fullmatch(r"(resume|continue|unpause|play again|keep playing|resume (it|the music))", t):
        return {"action": "resume"}
    if re.fullmatch(r"(stop|stop (it|the music|playing|everything))", t):
        return {"action": "stop"}
    if re.fullmatch(r"(join|join (the )?(vc|voice|call)|come (here|in|to (the )?(vc|call))|get in (the )?(vc|call))", t):
        return {"action": "join"}
    if re.fullmatch(r"(leave|leave (the )?(vc|voice|call)|disconnect|get out|bye|go away)", t):
        return {"action": "leave"}
    if re.fullmatch(r"(what'?s playing|what is playing|now playing|np|what song is this|which song is this)", t):
        return {"action": "now_playing"}
    if re.fullmatch(r"(queue|show (me )?the queue|what'?s next|what'?s in the queue|show queue)", t):
        return {"action": "show_queue"}
    if re.fullmatch(r"(clear|clear (the )?queue|empty the queue)", t):
        return {"action": "clear_queue"}

    m = re.fullmatch(r"(?:set )?(?:the )?vol(?:ume)?(?: to)? (\d{1,3})%?", t)
    if m:
        return {"action": "volume", "level": int(m.group(1))}
    if re.fullmatch(r"(louder|turn it up|volume up|turn up)", t):
        return {"action": "volume", "level": min(100, current_volume + 20)}
    if re.fullmatch(r"(quieter|softer|turn it down|volume down|turn down|lower)", t):
        return {"action": "volume", "level": max(0, current_volume - 20)}
    if re.fullmatch(r"(mute)", t):
        return {"action": "volume", "level": 0}

    m = re.fullmatch(r"remove (?:the )?(?:song )?(?:number )?(\d+|\w+)(?: song| one)?(?: from (?:the )?queue)?", t)
    if m:
        word = m.group(1)
        idx = int(word) if word.isdigit() else _ORDINALS.get(word)
        if idx:
            return {"action": "remove", "index": idx}

    m = re.fullmatch(r"(play|queue|add|put)\s+(.+?)(\s+next|\s+on|\s+to (the )?queue)?", t)
    if m:
        verb, query = m.group(1), m.group(2).strip()
        if query and not _VAGUE.search(query) and len(query) >= 2:
            action = "play" if verb == "play" else "queue"
            return {"action": action, "query": query}

    return None
