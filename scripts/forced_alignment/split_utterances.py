#!/usr/bin/env python3
"""Create MFA-style utterance WAV/text pairs from a TextGrid and source audio."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forced_alignment.alignment import discover_alignment_pairs, prepare_mfa_inputs
from transcription.config import load_config


def resolve_configured_inputs(config: dict[str, Any], root: str | Path | None = None) -> tuple[Path, Path, Path]:
    """Resolve the audio/transcript/output directories from the project config."""
    root_path = Path(root or ROOT).expanduser().resolve()

    audio_dir = Path(config.get("raw_waves_dir")).expanduser()
    transcript_dir = Path(config.get("normalised_transcripts_dir")).expanduser()
    output_dir = Path(config.get("mfa_output_dir", "")).expanduser()

    if not audio_dir.is_absolute():
        audio_dir = (root_path / audio_dir).resolve()
    if not transcript_dir.is_absolute():
        transcript_dir = (root_path / transcript_dir).resolve()
    if not output_dir.is_absolute():
        output_dir = (root_path / output_dir).resolve()

    return audio_dir, transcript_dir, output_dir


def main() -> int:
    config = load_config(ROOT / "src" / "package_data" / "config.json")
    audio_dir, transcript_dir, output_dir = resolve_configured_inputs(config, ROOT)

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    if not transcript_dir.exists():
        raise FileNotFoundError(f"Transcript directory not found: {transcript_dir}")

    pairs = discover_alignment_pairs(audio_dir, transcript_dir)
    if not pairs:
        raise FileNotFoundError(f"No matching audio/transcript pairs found in {audio_dir} and {transcript_dir}")

    prepared_dirs: list[Path] = []
    for audio_path, textgrid_path in pairs:
        pair_output_dir = output_dir / "split_utterances" / audio_path.stem
        prepared = prepare_mfa_inputs(audio_path, textgrid_path, pair_output_dir)
        prepared_dirs.append(prepared["segments_dir"])
        print(f"Prepared utterance folders in {prepared['segments_dir']}")

    if prepared_dirs:
        print(f"Prepared {len(prepared_dirs)} utterance set(s) from config-driven directories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
