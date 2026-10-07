"""
ai/intent.py — deterministic fast path for clear music commands.

"skip", "pause", "volume up", "play K by Cigarettes After Sex" don't need an
LLM to understand. Handling them here keeps voice commands snappy, keeps them
working when the AI is down, and saves tokens for the requests that actually
need judgement ("describe this song", "play something for a rainy night").

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
        r"|change|change (it|this|this song|the song|song)|another (one|song)|play (another|something else)( one| song)?)",
        t,
    ):
        return {"action": "skip"}
    if re.fullmatch(r"(pause|pause (it|the music|this)|hold (on|the music)|wait)", t):
        return {"action": "pause"}
    if re.fullmatch(
        r"(resume|continue|unpause|play|play again|play it|play (the )?music|keep playing|resume (it|the music))", t
    ):
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

    # Clipped or padded ("stop the", "pause it for now", "skip that one please"):
    # a short phrase that starts with the command, with no second request after it.
    m = re.fullmatch(
        r"(stop|pause|skip)((?: (?:it|the|this|that|music|song|one|now|for|a|sec|please|pls|bro|yaar))*)", t
    )
    if m:
        return {"action": m.group(1)}

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
    if re.fullmatch(r"(volume|vol) (up|increase|higher|louder)( a bit| a little| more)?", t):
        return {"action": "volume", "level": min(100, current_volume + 20)}
    if re.fullmatch(r"(volume|vol) (down|decrease|lower|quieter)( a bit| a little| more)?", t):
        return {"action": "volume", "level": max(0, current_volume - 20)}
    if re.fullmatch(r"(full volume|max volume|volume (full|max)|full blast)", t):
        return {"action": "volume", "level": 100}
    if re.fullmatch(r"(mute)", t):
        return {"action": "volume", "level": 0}

    m = re.fullmatch(r"remove (?:the )?(?:song )?(?:number )?(\d+|\w+)(?: song| one)?(?: from (?:the )?queue)?", t)
    if m:
        word = m.group(1)
        idx = int(word) if word.isdigit() else _ORDINALS.get(word)
        if idx:
            return {"action": "remove", "index": idx}

    return _parse_play(t)


# "play something sad", "play some chill music": no song named, the LLM picks one.
_VAGUE = re.compile(
    r"^(something|some |anything|a song|any song|songs? (for|that)|music (for|that)|"
    r"whatever|what ?ever|you choose|your choice|my favou?rite)", re.I
)


def _parse_play(t: str) -> dict | None:
    """'play X' / 'play X next' / 'play X now' / 'X bajau' → search for X as said."""
    action = "play"
    m = re.fullmatch(r"(?:play|put on|bajau|lagau)\s+(.+?)\s+(?:now|right now|instead)", t)
    if m:
        action = "play_now"
    else:
        m = re.fullmatch(r"(?:play|put on)\s+(.+?)\s+(?:next|after this(?: one)?)", t) or \
            re.fullmatch(r"(?:queue|add)\s+(.+?)(?:\s+(?:to|in)\s+(?:the\s+)?queue|\s+next)?", t)
        if m:
            action = "queue"
        else:
            m = re.fullmatch(r"(?:play|put on|bajau|lagau)\s+(.+)", t) or \
                re.fullmatch(r"(.+?)\s+(?:bajau|bajaideu|bajaa|lagau|lagaideu)", t)
    if not m:
        return None
    query = re.sub(r"^(the song |song |me |us )", "", m.group(1).strip()).strip()
    if not query or _VAGUE.match(query):
        return None
    return {"action": action, "query": query}
