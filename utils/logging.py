"""
utils/logging.py — configure structured logging for the whole bot.
"""

import logging
import sys
from config import settings


def setup_logging() -> None:
    """Call once at startup before any other module logs."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    # Quieten noisy libraries
    logging.getLogger("discord").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("discord.ext.voice_recv").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
