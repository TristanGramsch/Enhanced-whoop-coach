from __future__ import annotations

from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel


def transcribe_audio_to_text(audio_path: str, model_size: str = "tiny") -> str:
    """Transcribe an audio file to text using faster-whisper.

    Parameters
    ----------
    audio_path: Path to an audio file (e.g., wav, mp3, m4a, webm)
    model_size: whisper model size (tiny, base, small, medium). CPU recommended: tiny/base
    """
    # CPU inference
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe(audio_path, beam_size=1, vad_filter=True)
    texts = []
    for seg in segments:
        texts.append(seg.text)
    return " ".join(t.strip() for t in texts if t and t.strip()) 