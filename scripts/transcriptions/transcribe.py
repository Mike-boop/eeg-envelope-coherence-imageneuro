#!/usr/bin/env python3
"""Command-line transcription entry point for this project."""

from __future__ import annotations

import argparse
import gc
import sys
import wave
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PROVIDER = "whisperx"
CONFIG_PATH = ROOT / "src" / "package_data" / "config.json"

from transcription.config import load_config
from transcription.textgrid import write_textgrid


def clear_cuda_memory() -> None:
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    gc.collect()


def run_whisperx(audio_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    try:
        import whisperx
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("WhisperX support requires the transcription extra dependencies.") from exc

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("PyTorch is required for WhisperX transcription.") from exc

    model_name = config["whisperx"].get("model", "large-v3")
    language = config["whisperx"].get("language", "nl")
    device = config["whisperx"].get("device", "cpu")
    compute_type = config["whisperx"].get("compute_type", "int8")

    print(f"WhisperX device request: {device}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for WhisperX, but PyTorch reports no CUDA devices available.")

    result: dict[str, Any] | None = None
    model = None
    try:
        model = whisperx.load_model(model_name, device=device, compute_type=compute_type, language=language)
        result = model.transcribe(str(audio_path))
    finally:
        if model is not None:
            del model
        clear_cuda_memory()

    if result is None:
        raise RuntimeError("WhisperX did not return any transcription output")

    detected_language = result.get("language", language)
    segments = result.get("segments", [])

    return {
        "provider": "whisperx",
        "audio_path": str(audio_path),
        "segments": segments,
        "aligned_segments": align_segments(whisperx, segments, audio_path, detected_language, device),
        "language": detected_language,
        "raw": result,
    }


def align_segments(whisperx, segments, audio_path: Path, language: str, device: str) -> list[dict[str, Any]]:
    """
    Re-split ASR segments into sentences and attach word timings.

    `model.transcribe()` returns VAD-merged chunks of up to 30s. `whisperx.align()`
    runs each chunk's text through NLTK's Punkt sentence tokenizer and emits one
    segment per sentence, with per-word start/end times. These sentence-level
    segments are what the forced alignment consumes, so skipping this step would
    hand MFA 30-second utterances.
    """
    if not segments:
        return []

    align_model = None
    try:
        align_model, align_meta = whisperx.load_align_model(language_code=language, device=device)
        aligned = whisperx.align(
            segments, align_model, align_meta, str(audio_path), device,
            return_char_alignments=False,
        )
    finally:
        if align_model is not None:
            del align_model
        clear_cuda_memory()

    return aligned.get("segments", [])


def audio_duration(audio_path: Path) -> float:
    """Recording length in seconds, so the TextGrid's xmax matches the wav."""
    with wave.open(str(audio_path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def run_elevenlabs(audio_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    import requests

    api_key = config["elevenlabs"].get("api_key", "")
    if not api_key:
        raise RuntimeError("ElevenLabs API key is missing; set ELEVENLABS_API_KEY or your config file")

    model = config["elevenlabs"].get("model", "scribe_v1")
    language_code = config["elevenlabs"].get("language_code", "nl")

    with audio_path.open("rb") as handle:
        response = requests.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": api_key},
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
        "aligned_segments": [],
        "language": language_code,
        "raw": payload,
    }


def discover_audio_files(input_path: str | Path) -> list[Path]:
    input_path = Path(input_path).expanduser().resolve()
    if input_path.is_dir():
        return sorted(
            path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() == ".wav"
        )

    if input_path.is_file() and input_path.suffix.lower() == ".wav":
        return [input_path]
    raise FileNotFoundError(f"No audio files found in {input_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing normalised TextGrids instead of leaving them alone.",
    )
    args = parser.parse_args()

    config = load_config(CONFIG_PATH)

    output_dir = Path(config.get("raw_transcripts_dir", "data/raw_transcripts")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    normalised_dir = Path(
        config.get("normalised_transcripts_dir", "data/normalised_transcripts")
    ).expanduser().resolve()
    normalised_dir.mkdir(parents=True, exist_ok=True)

    raw_wav_dir = Path(config.get("raw_waves_dir") or str(output_dir / "raw_waves")).expanduser().resolve()
    if not raw_wav_dir.exists():
        raise FileNotFoundError(f"Precomputed WAV directory not found: {raw_wav_dir}")

    print(f"Using precomputed WAV directory: {raw_wav_dir}")

    audio_files = discover_audio_files(raw_wav_dir)
    if not audio_files:
        raise FileNotFoundError(f"No WAV files found in {raw_wav_dir}")

    for audio_path in audio_files:
        if PROVIDER == "whisperx":
            result = run_whisperx(audio_path, config)
        else:
            result = run_elevenlabs(audio_path, config)

        clear_cuda_memory()

        print(f"Transcribed {audio_path} via {result['provider']}")
        for segment in result.get("segments", []):
            start = segment.get("start", 0.0)
            end = segment.get("end", start)
            text = segment.get("text", "")
            print(f"{start:.2f}s - {end:.2f}s: {text}")

        duration = audio_duration(audio_path)

        raw_path = output_dir / f"{audio_path.stem}.TextGrid"
        write_textgrid(raw_path, result["segments"], duration=duration)
        print(f"Raw (30s chunk) TextGrid written to {raw_path}")

        aligned = result.get("aligned_segments") or []
        if not aligned:
            print(f"No aligned segments for {audio_path.stem}; skipping normalised TextGrid")
            continue

        normalised_path = normalised_dir / f"{audio_path.stem}.TextGrid"
        if normalised_path.exists() and not args.overwrite:
            # The checked-in normalised transcripts also carry Dutch text
            # normalisation this script does not do, so never clobber them silently.
            print(f"Normalised TextGrid already exists, leaving it: {normalised_path}")
            continue

        write_textgrid(normalised_path, aligned, duration=duration)
        print(f"Sentence-level TextGrid written to {normalised_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
