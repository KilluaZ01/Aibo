"""
utils/logging.py — configure structured logging for the whole bot.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
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

    # Same log to logs/bot.log (git-ignored) so it can be read after a test.
    os.makedirs("logs", exist_ok=True)
    file_handler = RotatingFileHandler(
        "logs/bot.log", maxBytes=2_000_000, backupCount=2, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # Quieten noisy libraries
    logging.getLogger("discord").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    # INFO so decryption / connection problems on the receive side show up.
    logging.getLogger("discord.ext.voice_recv").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
