"""
config.py — centralised configuration loaded from environment variables.
All other modules import from here; nothing reads os.environ directly.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(f"Required environment variable '{key}' is not set.")
    return value


def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default)


@dataclass(frozen=True)
class Config:
    # Discord
    discord_token: str

    # LLM
    llm_api_key: str
    llm_model: str
    llm_base_url: str

    # Lavalink
    lavalink_host: str
    lavalink_port: int
    lavalink_password: str
    lavalink_secure: bool

    # Bot behaviour
    bot_name: str
    command_prefix: str  # kept minimal — bot does NOT use prefix commands for music
    log_level: str

    # AI routing — which patterns trigger the LLM
    bot_mention_triggers: bool      # always True; bot is always triggered when mentioned
    max_context_messages: int       # bounded conversation window per guild

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            discord_token=_require("DISCORD_TOKEN"),
            llm_api_key=_require("LLM_API_KEY"),
            llm_model=_optional("LLM_MODEL", "claude-sonnet-4-6"),
            llm_base_url=_optional("LLM_BASE_URL", "https://api.anthropic.com"),
            lavalink_host=_optional("LAVALINK_HOST", "127.0.0.1"),
            lavalink_port=int(_optional("LAVALINK_PORT", "2333")),
            lavalink_password=_optional("LAVALINK_PASSWORD", "youshallnotpass"),
            lavalink_secure=_optional("LAVALINK_SECURE", "false").lower() == "true",
            bot_name=_optional("BOT_NAME", "Nova"),
            command_prefix=_optional("COMMAND_PREFIX", "!"),
            log_level=_optional("LOG_LEVEL", "INFO"),
            bot_mention_triggers=True,
            max_context_messages=int(_optional("MAX_CONTEXT_MESSAGES", "12")),
        )


# Singleton — imported everywhere
settings = Config.from_env()
