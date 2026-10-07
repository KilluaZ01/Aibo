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


def _default_wake_words() -> str:
    """Spellings speech-to-text produces for the bot's name."""
    name = _optional("BOT_NAME", "Milo").strip().lower()
    known = {
        "milo": "milo,mylo,meelo,mailo,milo's,my lo,mi lo,nilo,my love,main love,"
                "mile,my low,me low,mellow,melo,mello,below,willow,my loan,my lord,miller,mela,"
                "mill lo,mill love,mill low,mi love,mi low",
        "ai": "ai,a i,a.i,aye i",
        "aibo": "aibo,ai bo,aibou,aybo,eibo,ibo,i bo,eye bo,hi bo,haibo,"
                "i bow,eye bow,hi bow,ai bow,aibu,eibu,ivo,ebo,ai boo,i boo",
    }
    return known.get(name, name)


@dataclass(frozen=True)
class Config:
    # Discord
    discord_token: str

    # LLM — NVIDIA's API first (free tier, fast), Hugging Face as a fallback.
    # Models are tried in order until one answers.
    nvidia_api_key: str
    nvidia_base_url: str
    nvidia_models: tuple[str, ...]
    hf_token: str
    hf_models: tuple[str, ...]
    hf_intent_models: tuple[str, ...]   # bigger models for understanding requests
    hf_calls_per_min: int

    # Bot behaviour
    bot_name: str
    log_level: str
    max_context_messages: int       # bounded conversation window per guild
    song_comments: bool             # occasionally talk about a song when the queue moves on

    # Voice listening (wake word → speech-to-text → same brain as text chat)
    voice_listen: bool
    wake_words: tuple[str, ...]
    debug_save_audio: bool          # save every heard phrase to logs/audio/*.wav (testing only)
    voice_name_only: bool           # only react to phrases that start with the bot's name
    voice_direct_commands: bool     # "play X" / "skip" work without saying the wake word
    stt_backend: str                # "google" (free web API), "local" (faster-whisper) or "hf"
    stt_language: str               # Google language code, e.g. en-IN, en-US, ne-NP
    stt_local_model: str            # faster-whisper size: tiny / base / small
    stt_hf_model: str

    # Talking back in voice chat (spoken requests get a spoken answer)
    tts_reply: bool
    tts_engine: str                 # "edge" (natural neural voices) or "gtts" (Google)
    tts_voice: str                  # edge-tts voice for English / romanised Nepali
    tts_voice_ne: str               # edge-tts voice for Devanagari Nepali
    tts_rate: str                   # edge-tts speed, e.g. "+10%"
    tts_pitch: str                  # edge-tts pitch, e.g. "+20Hz" (higher = cuter)

    # YouTube from a datacenter IP (Oracle) gets "sign in to confirm you're not a bot".
    # A cookies.txt from a throwaway account fixes it; SoundCloud is the fallback.
    ytdlp_cookies: str
    soundcloud_fallback: bool

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            discord_token=_require("DISCORD_TOKEN"),
            nvidia_api_key=_optional("NVIDIA_API_KEY", ""),
            nvidia_base_url=_optional("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            # Super 120B answers in under a second; Ultra 550B is the backup
            # (often "overloaded"). Thinking is switched off for speed.
            nvidia_models=tuple(
                m.strip()
                for m in _optional(
                    "NVIDIA_MODELS",
                    "nvidia/nemotron-3-super-120b-a12b,nvidia/nemotron-3-ultra-550b-a55b",
                ).split(",")
                if m.strip()
            ),
            hf_token=_optional("HUGGINGFACE_TOKEN", ""),
            # Llama 3.1 8B: the model the Nima/Arik/Zidan bots actually get answers
            # from. (Mistral-7B and Zephyr are "not a chat model" on HF's router.)
            hf_models=tuple(
                m.strip()
                for m in _optional(
                    "HF_MODELS",
                    "meta-llama/Llama-3.1-8B-Instruct",
                ).split(",")
                if m.strip()
            ),
            # Understanding requests uses the same models unless HF_INTENT_MODELS
            # is set (bigger models are better at it but cost more credits).
            hf_intent_models=tuple(
                m.strip()
                for m in _optional("HF_INTENT_MODELS", "").split(",")
                if m.strip()
            ),
            hf_calls_per_min=int(_optional("HF_CALLS_PER_MIN", "20")),
            bot_name=_optional("BOT_NAME", "Milo"),
            log_level=_optional("LOG_LEVEL", "INFO"),
            max_context_messages=int(_optional("MAX_CONTEXT_MESSAGES", "12")),
            song_comments=_flag("SONG_COMMENTS", "false"),
            voice_listen=_flag("VOICE_LISTEN", "true"),
            wake_words=tuple(
                w.strip().lower()
                for w in _optional("WAKE_WORDS", _default_wake_words()).split(",")
                if w.strip()
            ),
            debug_save_audio=_flag("DEBUG_SAVE_AUDIO", "false"),
            voice_name_only=_flag("VOICE_NAME_ONLY", "true"),
            voice_direct_commands=_flag("VOICE_DIRECT_COMMANDS", "true"),
            stt_backend=_optional("STT_BACKEND", "google").strip().lower(),
            stt_language=_optional("STT_LANGUAGE", "en-IN"),
            stt_local_model=_optional("STT_LOCAL_MODEL", "base"),
            stt_hf_model=_optional("STT_HF_MODEL", "openai/whisper-large-v3-turbo"),
            tts_reply=_flag("TTS_REPLY", "true"),
            tts_engine=_optional("TTS_ENGINE", "edge").strip().lower(),
            tts_voice=_optional("TTS_VOICE", "en-US-BrianNeural"),
            tts_voice_ne=_optional("TTS_VOICE_NE", "ne-NP-SagarNeural"),
            tts_rate=_optional("TTS_RATE", "+8%"),
            tts_pitch=_optional("TTS_PITCH", "+15Hz"),
            ytdlp_cookies=_optional("YTDLP_COOKIES", ""),
            soundcloud_fallback=_flag("SOUNDCLOUD_FALLBACK", "true"),
        )


# Singleton — imported everywhere
settings = Config.from_env()

if not settings.nvidia_api_key and not settings.hf_token:
    raise EnvironmentError("Set NVIDIA_API_KEY (recommended) or HUGGINGFACE_TOKEN for the AI.")
