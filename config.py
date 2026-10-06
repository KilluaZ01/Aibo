"""
config.py — centralised configuration loaded from environment variables.
All other modules import from here; nothing reads os.environ directly.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(f"Required environment variable '{key}' is not set.")
    return value


def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _flag(key: str, default: str = "true") -> bool:
    return _optional(key, default).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    # Discord
    discord_token: str

    # LLM — Hugging Face Inference, same token as the Nima/Arik/Zidan bots.
    # Models are tried in order until one answers.
    hf_token: str
    hf_models: tuple[str, ...]
    hf_calls_per_min: int

    # Bot behaviour
    bot_name: str
    log_level: str
    max_context_messages: int       # bounded conversation window per guild
    song_comments: bool             # occasionally talk about a song when the queue moves on

    # Voice listening (wake word → speech-to-text → same brain as text chat)
    voice_listen: bool
    wake_words: tuple[str, ...]
    stt_backend: str                # "local" (faster-whisper, free) or "hf" (HF Whisper)
    stt_local_model: str            # faster-whisper size: tiny / base / small
    stt_hf_model: str

    # YouTube from a datacenter IP (Oracle) gets "sign in to confirm you're not a bot".
    # A cookies.txt from a throwaway account fixes it; SoundCloud is the fallback.
    ytdlp_cookies: str
    soundcloud_fallback: bool

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            discord_token=_require("DISCORD_TOKEN"),
            hf_token=_require("HUGGINGFACE_TOKEN"),
            hf_models=tuple(
                m.strip()
                for m in _optional(
                    "HF_MODELS",
                    "meta-llama/Llama-3.1-8B-Instruct,"
                    "Qwen/Qwen2.5-7B-Instruct,"
                    "mistralai/Mistral-7B-Instruct-v0.3",
                ).split(",")
                if m.strip()
            ),
            hf_calls_per_min=int(_optional("HF_CALLS_PER_MIN", "20")),
            bot_name=_optional("BOT_NAME", "Aibo"),
            log_level=_optional("LOG_LEVEL", "INFO"),
            max_context_messages=int(_optional("MAX_CONTEXT_MESSAGES", "12")),
            song_comments=_flag("SONG_COMMENTS", "true"),
            voice_listen=_flag("VOICE_LISTEN", "true"),
            wake_words=tuple(
                w.strip().lower()
                for w in _optional(
                    "WAKE_WORDS",
                    "aibo,ai bo,aibou,aybo,eibo,ibo,i bo,eye bo,hi bo,haibo",
                ).split(",")
                if w.strip()
            ),
            stt_backend=_optional("STT_BACKEND", "local").strip().lower(),
            stt_local_model=_optional("STT_LOCAL_MODEL", "base"),
            stt_hf_model=_optional("STT_HF_MODEL", "openai/whisper-large-v3-turbo"),
            ytdlp_cookies=_optional("YTDLP_COOKIES", ""),
            soundcloud_fallback=_flag("SOUNDCLOUD_FALLBACK", "true"),
        )


# Singleton — imported everywhere
settings = Config.from_env()
