"""
ai/tools.py — Anthropic tool definitions.

These are passed to the LLM API so the model knows what tools exist
and what arguments they accept. The actual implementation lives in
music/player.py — the LLM only decides *which* tool to call.
"""

MUSIC_TOOLS: list[dict] = [
    {
        "name": "play_song",
        "description": (
            "Search SoundCloud for a track and play it immediately. "
            "If another song is playing and the user did not explicitly say 'play now' or 'replace', "
            "prefer queue_song instead. Use this when the user wants to start playing something right now."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for SoundCloud — e.g. 'K Cigarettes After Sex'",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "queue_song",
        "description": (
            "Search SoundCloud for a track and add it to the queue without interrupting playback. "
            "Use this when the user wants to add something for later, or when a song is already playing "
            "and they haven't explicitly asked to replace it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for SoundCloud",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "skip_song",
        "description": "Skip the currently playing track and move to the next one in the queue.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "pause_music",
        "description": "Pause the currently playing track.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "resume_music",
        "description": "Resume a paused track.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "stop_music",
        "description": "Stop playback entirely and clear the queue.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "now_playing",
        "description": "Get information about the currently playing track — title, artist, elapsed time, duration.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "show_queue",
        "description": "Get the list of tracks currently in the queue.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "remove_from_queue",
        "description": (
            "Remove a specific track from the queue by its 1-based position. "
            "Interpret natural language like 'the third song' as index 3."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "1-based position in the queue to remove",
                },
            },
            "required": ["index"],
        },
    },
    {
        "name": "clear_queue",
        "description": "Remove all tracks from the queue without stopping the current track.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "set_volume",
        "description": (
            "Set the playback volume. "
            "Convert natural language (e.g. 'louder', 'quieter', 'turn it down', 'max') to a numeric level. "
            "'louder' = current + 20, 'quieter'/'turn it down' = current - 20, 'max' = 100, 'mute' = 0."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "level": {
                    "type": "integer",
                    "description": "Volume level 0–100",
                    "minimum": 0,
                    "maximum": 100,
                },
            },
            "required": ["level"],
        },
    },
    {
        "name": "leave_voice",
        "description": "Disconnect the bot from the voice channel.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

# OpenAI / Groq format — used by ai/client.py
MUSIC_TOOLS_OPENAI: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in MUSIC_TOOLS
]

# Fast lookup by name — used in dispatcher
TOOL_NAMES: set[str] = {t["name"] for t in MUSIC_TOOLS}
