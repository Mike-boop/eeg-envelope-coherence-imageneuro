#!/usr/bin/env python3
"""Stitch per-utterance MFA TextGrid outputs back into one master TextGrid per source recording."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forced_alignment.alignment import write_recombined_textgrid
from transcription.config import load_config


def resolve_configured_inputs(config: dict[str, Any], root: str | Path | None = None) -> tuple[Path, Path]:
    """Resolve the raw audio and MFA output directories from the project config."""
    root_path = Path(root or ROOT).expanduser().resolve()

    audio_dir = Path(config.get("raw_waves_dir")).expanduser()
    mfa_output_dir = Path(config.get("mfa_output_dir", "")).expanduser()

    if not audio_dir.is_absolute():
        audio_dir = (root_path / audio_dir).resolve()
    if not mfa_output_dir.is_absolute():
        mfa_output_dir = (root_path / mfa_output_dir).resolve()

    return audio_dir, mfa_output_dir


def main() -> int:
    config = load_config(ROOT / "src" / "package_data" / "config.json")
    audio_dir, mfa_output_dir = resolve_configured_inputs(config, ROOT)

    aligned_root = mfa_output_dir / "raw_mfa_outputs"
    combined_root = mfa_output_dir / "combined_textgrids"

    if not aligned_root.exists():
        raise FileNotFoundError(f"Aligned output directory not found: {aligned_root}")

    written = 0
    for aligned_dir in sorted(path for path in aligned_root.iterdir() if path.is_dir()):
        audio_stem = aligned_dir.name
        audio_path = audio_dir / f"{audio_stem}.wav"
        if not audio_path.exists():
            print(f"Skipping {audio_stem}: source audio not found at {audio_path}")
            continue

        output_path = combined_root / f"{audio_stem}.TextGrid"
        write_recombined_textgrid(audio_path, aligned_dir, output_path)
        print(f"Wrote combined TextGrid to {output_path}")
        written += 1

    print(f"Combined {written} recording(s) into {combined_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
