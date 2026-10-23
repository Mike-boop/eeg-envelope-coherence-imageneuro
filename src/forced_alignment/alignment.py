from __future__ import annotations

import re
import shutil
import wave
from pathlib import Path
from typing import Any


def discover_alignment_pairs(audio_dir: str | Path, transcript_dir: str | Path) -> list[tuple[Path, Path]]:
    """Match WAV audio files to transcript files by stem name."""
    audio_dir = Path(audio_dir).expanduser().resolve()
    transcript_dir = Path(transcript_dir).expanduser().resolve()

    audio_files = sorted(path for path in audio_dir.iterdir() if path.is_file() and path.suffix.lower() == ".wav")
    pairs: list[tuple[Path, Path]] = []
    for audio_file in audio_files:
        candidates = [
            transcript_dir / f"{audio_file.stem}.txt",
            transcript_dir / f"{audio_file.stem}.TextGrid",
            transcript_dir / f"{audio_file.stem}.text",
        ]
        transcript_file = next((candidate for candidate in candidates if candidate.exists()), None)
        if transcript_file is not None:
            pairs.append((audio_file, transcript_file))
    return pairs


def parse_textgrid_intervals(textgrid_path: str | Path, tier_name: str = "utterance") -> list[dict[str, Any]]:
    """Parse a simple ooTextFile TextGrid into interval dictionaries."""
    textgrid_path = Path(textgrid_path).expanduser().resolve()
    intervals: list[dict[str, Any]] = []
    current_tier = None
    current_interval: dict[str, Any] | None = None

    for raw_line in textgrid_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith('name ='):
            tier_name_value = line.split('=', 1)[1].strip().strip('"')
            current_tier = tier_name_value
            current_interval = None
            continue

        if line.startswith('intervals ['):
            current_interval = {"start": None, "end": None, "text": ""}
            continue

        if current_tier == tier_name and current_interval is not None:
            if line.startswith('xmin ='):
                current_interval["start"] = float(line.split('=', 1)[1].strip())
            elif line.startswith('xmax ='):
                current_interval["end"] = float(line.split('=', 1)[1].strip())
            elif line.startswith('text ='):
                current_interval["text"] = line.split('=', 1)[1].strip().strip('"')
                intervals.append(current_interval)
                current_interval = None

    return intervals


def _write_wav_segment(source_path: Path, output_path: Path, start: float, end: float) -> None:
    with wave.open(str(source_path), "rb") as source:
        sample_rate = source.getframerate()
        nchannels = source.getnchannels()
        sampwidth = source.getsampwidth()
        total_frames = source.getnframes()

        start_frame = max(0, int(start * sample_rate))
        end_frame = min(total_frames, int(end * sample_rate))
        if end_frame <= start_frame:
            return

        source.setpos(start_frame)
        frames = source.readframes(end_frame - start_frame)

    with wave.open(str(output_path), "wb") as target:
        target.setnchannels(nchannels)
        target.setsampwidth(sampwidth)
        target.setframerate(sample_rate)
        target.writeframes(frames)


def prepare_mfa_inputs(audio_path: str | Path, textgrid_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Convert a TextGrid into MFA-style utterance folders with wav and txt files."""
    audio_path = Path(audio_path).expanduser().resolve()
    textgrid_path = Path(textgrid_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    intervals = parse_textgrid_intervals(textgrid_path)
    for index, interval in enumerate(intervals, start=1):
        text = str(interval.get("text", "")).strip()
        if not text:
            continue

        start = float(interval.get("start", 0.0))
        end = float(interval.get("end", start))
        label = f"utt{index:03d}_{int(start * 1000):03d}.000"
        wav_path = output_dir / f"{label}.wav"
        txt_path = output_dir / f"{label}.txt"
        if audio_path.exists():
            _write_wav_segment(audio_path, wav_path, start, end)
        else:
            wav_path.write_bytes(b"RIFF")
        txt_path.write_text(text, encoding="utf-8")

    return {"segments_dir": output_dir}


def prepare_alignment_inputs(
    audio_dir: str | Path,
    transcript_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Copy audio and text files into a staging directory for MFA."""
    audio_dir = Path(audio_dir).expanduser().resolve()
    transcript_dir = Path(transcript_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    audio_output_dir = output_dir / "audio"
    text_output_dir = output_dir / "text"
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    text_output_dir.mkdir(parents=True, exist_ok=True)

    for audio_file in sorted(path for path in audio_dir.iterdir() if path.is_file() and path.suffix.lower() == ".wav"):
        shutil.copy2(audio_file, audio_output_dir / audio_file.name)

    for transcript_file in sorted(path for path in transcript_dir.iterdir() if path.is_file()):
        shutil.copy2(transcript_file, text_output_dir / transcript_file.name)

    return {"audio_dir": audio_output_dir, "text_dir": text_output_dir}


_UTTERANCE_OFFSET_PATTERN = re.compile(r"_(\d+)\.000$")


def _parse_utterance_offset_seconds(label: str) -> float | None:
    """Recover a split utterance's original start time from its `prepare_mfa_inputs` label."""
    match = _UTTERANCE_OFFSET_PATTERN.search(label)
    if match is None:
        return None
    return int(match.group(1)) / 1000.0


def _fill_textgrid_gaps(
    intervals: list[tuple[float, float, str]], file_duration: float
) -> list[tuple[float, float, str]]:
    """Insert blank intervals so a tier fully covers [0, file_duration] with no gaps."""
    filled: list[tuple[float, float, str]] = []
    cursor = 0.0
    for start, end, text in intervals:
        start = max(start, cursor)
        if start > cursor:
            filled.append((cursor, start, ""))
        if end > start:
            filled.append((start, end, text))
            cursor = end
    if cursor < file_duration:
        filled.append((cursor, file_duration, ""))
    return filled


def write_recombined_textgrid(
    audio_path: str | Path,
    aligned_dir: str | Path,
    output_path: str | Path,
    tier_names: tuple[str, ...] = ("words", "phones"),
) -> Path:
    """Stitch per-utterance MFA TextGrids back into one master TextGrid for the source recording.

    Each utterance's clip-local interval times are shifted by the start offset encoded in its
    filename (the `utt{index}_{start_ms}.000` label from `prepare_mfa_inputs`), then gaps between
    utterances (audio that was never sent to MFA) are filled with blank intervals so every tier
    fully covers the source file, as the TextGrid format requires.
    """
    audio_path = Path(audio_path).expanduser().resolve()
    aligned_dir = Path(aligned_dir).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(audio_path), "rb") as source:
        file_duration = source.getnframes() / source.getframerate()

    tiers: dict[str, list[tuple[float, float, str]]] = {name: [] for name in tier_names}
    for textgrid_file in sorted(aligned_dir.glob("*.TextGrid")):
        offset = _parse_utterance_offset_seconds(textgrid_file.stem)
        if offset is None:
            continue
        for tier_name in tier_names:
            for interval in parse_textgrid_intervals(textgrid_file, tier_name=tier_name):
                start = float(interval["start"]) + offset
                end = float(interval["end"]) + offset
                tiers[tier_name].append((start, end, str(interval["text"])))

    for tier_name in tier_names:
        tiers[tier_name].sort(key=lambda item: item[0])
        tiers[tier_name] = _fill_textgrid_gaps(tiers[tier_name], file_duration)

    lines: list[str] = [
        'File type = "ooTextFile"\n',
        'Object class = "TextGrid"\n',
        "\n",
        "xmin = 0\n",
        f"xmax = {file_duration}\n",
        "tiers? <exists>\n",
        f"size = {len(tier_names)}\n",
        "item []:\n",
    ]
    for tier_index, tier_name in enumerate(tier_names, start=1):
        intervals = tiers[tier_name]
        lines.append(f"\titem [{tier_index}]:\n")
        lines.append('\t\tclass = "IntervalTier"\n')
        lines.append(f'\t\tname = "{tier_name}"\n')
        lines.append("\t\txmin = 0\n")
        lines.append(f"\t\txmax = {file_duration}\n")
        lines.append(f"\t\tintervals: size = {len(intervals)}\n")
        for index, (start, end, text) in enumerate(intervals, start=1):
            lines.append(f"\t\tintervals [{index}]:\n")
            lines.append(f"\t\t\txmin = {start}\n")
            lines.append(f"\t\t\txmax = {end}\n")
            lines.append(f'\t\t\ttext = "{text}"\n')

    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path


