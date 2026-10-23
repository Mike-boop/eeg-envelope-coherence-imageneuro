#!/usr/bin/env python3
"""Extract continuous audio features (gammatone cochleogram, broadband envelope) per recording."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from tqdm import tqdm

from audio_features.factory import (
    AudioExtractor,
    Brian2HearsGammatoneCochleogramExtractor,
    BroadbandEnvelopeExtractor,
    FeatureExtractor,
    GammatoneCochleogramExtractor,
    extract_features_from_wav,
)
from transcription.config import load_config

# Features to extract for each recording. Comment out any extractor to skip it,
# or tweak its parameters (e.g. GammatoneCochleogramExtractor(n_channels=32)).
SELECTED_FEATURES: list[FeatureExtractor] = [
    Brian2HearsGammatoneCochleogramExtractor(),
    # AudioExtractor(),
]


def resolve_configured_inputs(config: dict[str, Any], root: str | Path | None = None) -> tuple[Path, Path]:
    """Resolve the raw audio and feature output directories from the project config."""
    root_path = Path(root or ROOT).expanduser().resolve()

    audio_dir = Path(config.get("raw_waves_dir")).expanduser()
    output_dir = Path(config.get("audio_features_dir", "data/audio_features")).expanduser()

    if not audio_dir.is_absolute():
        audio_dir = (root_path / audio_dir).resolve()
    if not output_dir.is_absolute():
        output_dir = (root_path / output_dir).resolve()

    return audio_dir, output_dir


def main() -> int:
    config = load_config(ROOT / "src" / "package_data" / "config.json")
    audio_dir, output_dir = resolve_configured_inputs(config, ROOT)

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(audio_dir.glob("*.wav"))
    print(audio_files)
    if not audio_files:
        raise FileNotFoundError(f"No .wav files found in {audio_dir}")

    written = 0
    for audio_path in tqdm(audio_files, desc="Extracting features", unit="file"):
        output_path = output_dir / f"{audio_path.stem}.npz"
        # if output_path.exists():
        #     continue

        features = extract_features_from_wav(audio_path, extractors=tuple(SELECTED_FEATURES))
        np.savez(output_path, **features)
        written += 1

    print(f"Wrote features for {written} recording(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
