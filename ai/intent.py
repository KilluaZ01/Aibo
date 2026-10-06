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

    if re.fullmatch(
        # "ski"/"sk"/"kip": speech-to-text often clips "skip"
        r"(skip|ski|sk|skeep|kip|next|next song|next one|skip (it|this|this one|this song|song)|nah skip( this| it)?"
        r"|change|change (it|this|this song|the song|song)|another (one|song)|play (another|something else))",
        t,
    ):
        return {"action": "skip"}
    if re.fullmatch(r"(pause|pause (it|the music|this)|hold (on|the music)|wait)", t):
        return {"action": "pause"}
    if re.fullmatch(r"(resume|continue|unpause|play again|keep playing|resume (it|the music))", t):
        return {"action": "resume"}
    if re.fullmatch(r"(stop|stop (it|the music|the song|this song|music|playing|everything))", t):
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
    # "turn the volume up to 100", "make the volume 20%", "set it to 50", "make it 30"
    m = re.fullmatch(
        r"(?:turn|set|make|put|change|bring)\s+(?:the\s+)?(?:volume|music|it|sound)"
        r"(?:\s+(?:up|down))?(?:\s+to)?\s+(\d{1,3})\s*(?:%|percent)?",
        t,
    )
    if m:
        return {"action": "volume", "level": min(100, int(m.group(1)))}
    # Short phrase with "volume <n>" somewhere, e.g. a misheard name in front:
    # "low volume 0%" (= "Milo, volume 0%").
    m = re.search(r"\bvol(?:ume)?\s+(?:to\s+)?(\d{1,3})\s*(?:%|percent)?$", t)
    if m and len(t.split()) <= 5:
        return {"action": "volume", "level": min(100, int(m.group(1)))}
    _VOL = r"(?:it|the volume|volume|the music|music|the sound|sound)"
    if re.fullmatch(
        rf"(louder|volume up|turn up|turn {_VOL} up|increase {_VOL}|raise {_VOL}|"
        rf"pump it up|make it louder|(?:a bit|little) louder)( a bit| a little| more)?",
        t,
    ):
        return {"action": "volume", "level": min(100, current_volume + 20)}
    if re.fullmatch(
        rf"(quieter|softer|lower|volume down|turn down|turn {_VOL} down|decrease {_VOL}|"
        rf"lower {_VOL}|reduce {_VOL}|make it quieter)( a bit| a little| more)?",
        t,
    ):
        return {"action": "volume", "level": max(0, current_volume - 20)}
    if re.fullmatch(r"(mute)", t):
        return {"action": "volume", "level": 0}

    m = re.fullmatch(r"remove (?:the )?(?:song )?(?:number )?(\d+|\w+)(?: song| one)?(?: from (?:the )?queue)?", t)
    if m:
        word = m.group(1)
        idx = int(word) if word.isdigit() else _ORDINALS.get(word)
        if idx:
            return {"action": "remove", "index": idx}

    # Song requests ("play oben eyes") are left to the LLM on purpose: it
    # fixes misheard names and picks the right song before we search.
    return None
