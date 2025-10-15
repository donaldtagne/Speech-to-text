from __future__ import annotations

from pathlib import Path

import whisper


def transcribe_audio(audio_path: Path, model_size: str = "small") -> str:
    """Convert an audio file to text using OpenAI Whisper."""

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = whisper.load_model(model_size)
    result = model.transcribe(str(audio_path))
    return result["text"].strip()
