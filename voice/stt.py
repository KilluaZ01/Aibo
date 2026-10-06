"""
voice/stt.py — speech-to-text for voice commands.

Two backends:
- "local" (default): faster-whisper on the CPU. Free, nothing leaves the
  machine, and it can afford to hear every phrase to spot the wake word.
- "hf": Hugging Face Whisper with the same HUGGINGFACE_TOKEN. More accurate
  on song names but every phrase spends inference credits.

Audio in is 16 kHz mono float32 (see voice/listener.py).
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave

import numpy as np

from config import settings

log = logging.getLogger("nova.voice.stt")

SAMPLE_RATE = 16000

# Nudges Whisper towards spelling the wake word and common requests right.
_PROMPT = f"{settings.bot_name}, play a song. {settings.bot_name}, skip. {settings.bot_name}, pause."

_local_model = None
_backend = settings.stt_backend


def _load_local():
    global _local_model, _backend
    if _local_model is not None:
        return _local_model
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        log.warning("[STT] faster-whisper not installed — falling back to Hugging Face Whisper")
        _backend = "hf"
        return None
    log.info("[STT] Loading faster-whisper '%s' (first time downloads the model)", settings.stt_local_model)
    _local_model = WhisperModel(settings.stt_local_model, device="cpu", compute_type="int8")
    return _local_model


def _transcribe_local(audio: np.ndarray) -> str:
    model = _load_local()
    if model is None:
        return ""
    segments, _info = model.transcribe(
        audio,
        beam_size=1,
        initial_prompt=_PROMPT,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    return " ".join(s.text for s in segments).strip()


def _to_wav(audio: np.ndarray) -> bytes:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


async def _transcribe_hf(audio: np.ndarray) -> str:
    from huggingface_hub import AsyncInferenceClient

    client = AsyncInferenceClient(token=settings.hf_token)
    out = await client.automatic_speech_recognition(_to_wav(audio), model=settings.stt_hf_model)
    return (getattr(out, "text", "") or "").strip()


async def warm_up() -> None:
    """Load the local model at startup so the first command isn't slow."""
    if _backend == "local":
        await asyncio.to_thread(_load_local)


async def transcribe(audio: np.ndarray) -> str:
    try:
        if _backend == "local":
            text = await asyncio.to_thread(_transcribe_local, audio)
            if _backend == "local":
                return text
        return await _transcribe_hf(audio)
    except Exception as exc:
        log.warning("[STT] Transcription failed: %s", str(exc)[:200])
        return ""
