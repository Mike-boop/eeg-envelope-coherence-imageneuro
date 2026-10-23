#!/usr/bin/env python3
"""Precompute WAV files from .npz or .wav inputs into a dedicated directory."""

from __future__ import annotations

import shutil
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CONFIG_PATH = ROOT / "src" / "package_data" / "config.json"


def prepare_input_audio(audio_path: str | Path, output_dir: str | Path | None = None) -> Path:
    """Convert .npz audio payloads to WAV and copy .wav inputs into a target directory."""
    audio_path = Path(audio_path).expanduser().resolve()
    target_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else audio_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    suffix = audio_path.suffix.lower()
    if suffix == ".wav":
        output_path = target_dir / audio_path.name.split("_-_")[0] + ".wav"
        if output_path != audio_path:
            shutil.copy2(audio_path, output_path)
        return output_path

    if suffix != ".npz":
        raise ValueError(f"Unsupported audio format: {audio_path}")

    output_path = target_dir / f"{audio_path.stem.split('_-_')[0]}.wav"
    with np.load(audio_path) as data:
        if "waveform" not in data or "fs" not in data:
            raise ValueError("NPZ audio input must contain 'waveform' and 'fs' keys")
        samples = np.asarray(data["waveform"])
        sample_rate = int(np.asarray(data["fs"]).reshape(()).item())

    if samples.ndim > 1:
        if samples.shape[0] == 1:
            samples = samples[0]
        elif samples.shape[1] == 1:
            samples = samples[:, 0]
        else:
            samples = samples.mean(axis=1)

    samples = np.asarray(samples, dtype=np.float64)
    samples = np.clip(samples, -1.0, 1.0)
    pcm = np.round(samples * 32767).astype(np.int16)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())

    return output_path


def discover_audio_files(input_path: str | Path) -> list[Path]:
    input_path = Path(input_path).expanduser().resolve()
    if input_path.is_dir():
        return sorted(
            path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() in {".wav", ".npz"}
        )
    if input_path.is_file() and input_path.suffix.lower() in {".wav", ".npz"}:
        return [input_path]
    raise FileNotFoundError(f"No audio files found in {input_path}")


def main() -> int:
    from transcription.config import load_config

    config = load_config(CONFIG_PATH)
    source_root = Path(config.get("sparrkulee_dir", "/data_nfs/gu08wuvu/datasets/accou2023/")).expanduser().resolve()
    source_dir = source_root / "standardised-flat" / "audio"
    if not source_dir.exists():
        raise FileNotFoundError(f"Source audio directory not found: {source_dir}")

    output_dir = Path(config.get("raw_waves_dir") or config.get("raw_transcripts_dir", "data/raw_transcripts")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = discover_audio_files(source_dir)
    if not audio_files:
        raise FileNotFoundError(f"No .wav or .npz files found in {source_dir}")

    print(f"Preparing {len(audio_files)} audio file(s) in {output_dir}")
    for audio_path in audio_files:
        prepared_path = prepare_input_audio(audio_path, output_dir=output_dir)
        print(f"- {audio_path.name} -> {prepared_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
