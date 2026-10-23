from __future__ import annotations

from pathlib import Path
from typing import Any

import requests
import whisperx

from .config import load_config
from .textgrid import write_textgrid


class WhisperXTranscriptionPipeline:
    """A wrapper around a WhisperX-based transcription workflow."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config = load_config(config_path)

    def transcribe(self, audio_path: str | Path) -> dict[str, Any]:
        """Transcribe an audio file with WhisperX."""
        audio_path = Path(audio_path)
        model_name = self.config["whisperx"].get("model", "large-v3")
        language = self.config["whisperx"].get("language", "nl")
        device = self.config["whisperx"].get("device", "cpu")
        compute_type = self.config["whisperx"].get("compute_type", "int8")

        model = whisperx.load_model(model_name, device=device, compute_type=compute_type, language=language)
        result = model.transcribe(str(audio_path))
        return {
            "provider": "whisperx",
            "audio_path": str(audio_path),
            "segments": result.get("segments", []),
            "language": result.get("language", language),
            "raw": result,
        }

    def export_textgrid(self, output_path: str | Path, segments: list[dict[str, Any]], *, duration: float | None = None) -> Path:
        return write_textgrid(output_path, segments, duration=duration)


class ElevenLabsTranscriptionPipeline:
    """A wrapper around an ElevenLabs transcription workflow."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config = load_config(config_path)
        self.api_key = self.config["elevenlabs"].get("api_key", "")

    def transcribe(self, audio_path: str | Path) -> dict[str, Any]:
        """Transcribe an audio file with ElevenLabs."""
        if not self.api_key:
            raise RuntimeError("ElevenLabs API key is missing; set ELEVENLABS_API_KEY or config.json")

        audio_path = Path(audio_path)
        model = self.config["elevenlabs"].get("model", "scribe_v1")
        language_code = self.config["elevenlabs"].get("language_code", "nl")

        with audio_path.open("rb") as handle:
            response = requests.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers={"xi-api-key": self.api_key},
                params={"model_id": model, "language_code": language_code},
                files={"file": handle},
                timeout=600,
            )
            response.raise_for_status()

        payload = response.json()
        segments = []
        for item in payload.get("segments", []) or []:
            text = item.get("text", "")
            start = item.get("start", 0.0)
            end = item.get("end", start)
            segments.append({"text": text, "start": start, "end": end, "words": []})

        return {
            "provider": "elevenlabs",
            "audio_path": str(audio_path),
            "segments": segments,
            "language": language_code,
            "raw": payload,
        }

    def export_textgrid(self, output_path: str | Path, segments: list[dict[str, Any]], *, duration: float | None = None) -> Path:
        return write_textgrid(output_path, segments, duration=duration)
