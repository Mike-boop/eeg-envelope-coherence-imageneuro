"""Shared path/config plumbing for the speech-unit and pitch extraction scripts.

Every script in this folder reads the same three locations out of
`src/package_data/config.json`:

- `mfa_output_dir/combined_textgrids` — master TextGrids (`words`, `phones` tiers)
  produced by `scripts/forced_alignment/recombine_textgrids.py`
- `raw_waves_dir` — the stimulus `.wav` files
- `speech_units_dir` — where the JSON / npz outputs of these scripts are written
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription.config import load_config  # noqa: E402

CONFIG_PATH = ROOT / "src" / "package_data" / "config.json"
SCRIPT_DIR = Path(__file__).resolve().parent

# Non-phone labels emitted by MFA on the `phones` tier. These are not speech
# units and must never contribute an onset or count towards a rate.
NON_SPEECH_LABELS = frozenset({"", "sil", "sp", "spn", "sp:", "<eps>", "silence"})


@dataclass(frozen=True)
class Paths:
    """Resolved input/output directories for this script family."""

    textgrid_dir: Path
    audio_dir: Path
    output_dir: Path
    transcript_dir: Path


def _resolve(value: str | Path | None, default: str | Path) -> Path:
    path = Path(value if value else default).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def load_paths(*, require_textgrids: bool = False, require_audio: bool = False) -> Paths:
    """Resolve the shared directories, creating the output directory if needed."""
    config: dict[str, Any] = load_config(CONFIG_PATH)

    mfa_output_dir = _resolve(config.get("mfa_output_dir"), "data/forced_alignment")
    textgrid_dir = mfa_output_dir / "combined_textgrids"
    audio_dir = _resolve(config.get("raw_waves_dir"), "data/raw_waves")
    output_dir = _resolve(config.get("speech_units_dir"), "data/speech_units")
    transcript_dir = _resolve(config.get("normalised_transcripts_dir"), "data/normalised_transcripts")

    if require_textgrids and not textgrid_dir.is_dir():
        raise FileNotFoundError(
            f"TextGrid directory not found: {textgrid_dir}\n"
            "Run scripts/forced_alignment/recombine_textgrids.py first."
        )
    if require_audio and not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    return Paths(
        textgrid_dir=textgrid_dir,
        audio_dir=audio_dir,
        output_dir=output_dir,
        transcript_dir=transcript_dir,
    )


def textgrid_files(textgrid_dir: Path) -> list[Path]:
    """All master TextGrids, sorted by stimulus name."""
    files = sorted(textgrid_dir.glob("*.TextGrid"))
    if not files:
        raise FileNotFoundError(f"No .TextGrid files found in {textgrid_dir}")
    return files


def audio_files(audio_dir: Path) -> list[Path]:
    """All stimulus wavs, sorted by stimulus name."""
    files = sorted(audio_dir.glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No .wav files found in {audio_dir}")
    return files


def paired_files(textgrid_dir: Path, audio_dir: Path) -> list[tuple[str, Path, Path]]:
    """Match TextGrids to wavs by stem; skip (and report) stimuli missing one side."""
    pairs: list[tuple[str, Path, Path]] = []
    unmatched: list[str] = []
    for tg_path in textgrid_files(textgrid_dir):
        wav_path = audio_dir / f"{tg_path.stem}.wav"
        if wav_path.exists():
            pairs.append((tg_path.stem, tg_path, wav_path))
        else:
            unmatched.append(tg_path.stem)

    if unmatched:
        print(f"WARNING: no .wav in {audio_dir} for {len(unmatched)} TextGrid(s): {', '.join(unmatched)}")
    if not pairs:
        raise FileNotFoundError(f"No TextGrid/wav pairs found between {textgrid_dir} and {audio_dir}")
    return pairs


def is_speech_label(label: str) -> bool:
    """False for empty intervals and MFA's silence / unknown-word placeholders."""
    return label.strip().lower() not in NON_SPEECH_LABELS


_INTERVAL_RE = re.compile(
    r'xmin = ([\d.]+)\s*\n\s*xmax = ([\d.]+)\s*\n\s*text = "([^"]*)"'
)


def read_interval_tier(textgrid_path: Path, tier_name: str) -> list[tuple[float, float, str]]:
    """
    Read one interval tier as (start, end, text), tolerantly.

    praatio refuses TextGrids whose intervals overlap, which the WhisperX-derived
    transcripts in `normalised_transcripts_dir` occasionally do. This reads the
    tier as text instead, so those files stay usable.
    """
    contents = textgrid_path.read_text(encoding="utf-8")
    for block in re.split(r"\titem \[\d+\]:", contents)[1:]:
        match = re.search(r'name = "([^"]*)"', block)
        if match and match.group(1) == tier_name:
            return [(float(a), float(b), text) for a, b, text in _INTERVAL_RE.findall(block)]
    raise KeyError(f'No tier "{tier_name}" in {textgrid_path}')


def write_json(payload: Any, path: Path, *, quiet: bool = False) -> None:
    """Write `payload` as indented JSON and report where it went."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, ensure_ascii=False)
    if not quiet:
        print(f"  wrote {path}")
