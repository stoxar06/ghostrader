"""Local speech-to-text for strategy videos — offline, via faster-whisper.

Kept dependency-light and pluggable: the heavy `faster-whisper` import is lazy, so the
rest of the pipeline (rule extraction + honest testing) works without it via `--text`.
faster-whisper decodes audio with ffmpeg/PyAV, so both must be installed for real media.
"""
from __future__ import annotations

from pathlib import Path

from src.logutil import get_logger

log = get_logger(__name__)

INSTALL_HINT = (
    "Local transcription needs faster-whisper + ffmpeg:\n"
    "  pip install faster-whisper\n"
    "  sudo apt-get install -y ffmpeg     # or: brew install ffmpeg\n"
    "Then re-run `python -m src learn <video>`. "
    "To test a transcript you already have, use:  python -m src learn --text \"...\""
)


def faster_whisper_transcribe(media_path: str, model_size: str = "base") -> str:
    """Transcribe a local audio/video file with faster-whisper on CPU (int8)."""
    if not Path(media_path).exists():
        raise FileNotFoundError(f"media not found: {media_path}")
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper is not installed.\n" + INSTALL_HINT) from exc
    log.info("Transcribing %s with faster-whisper (%s, cpu/int8)…", media_path, model_size)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(str(media_path))
    return " ".join(seg.text.strip() for seg in segments).strip()


def transcribe(media_path: str, model_size: str = "base", backend=None) -> str:
    """Transcribe `media_path` to text. `backend(media_path) -> str` overrides the engine
    (used by tests and to swap in a hosted API later); defaults to local faster-whisper."""
    if backend is None:
        return faster_whisper_transcribe(media_path, model_size)
    return backend(media_path)
