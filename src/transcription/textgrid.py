from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable


def _clean_label(text: str) -> str:
    return re.sub(r"[^A-Za-zÀ-ÿ0-9'\s]", "", text).strip()


def write_textgrid(
    output_path: str | Path,
    segments: Iterable[dict[str, Any]],
    *,
    sr: int = 16000,
    duration: float | None = None,
) -> Path:
    """Write utterance and word interval tiers to an ooTextFile TextGrid."""
    output_path = Path(output_path)
    segments = list(segments)

    if duration is None:
        if segments:
            duration = max(float(seg.get("end", 0.0)) for seg in segments)
        else:
            duration = 0.0

    def q(value: float) -> str:
        return f"{value:.6f}"

    lines: list[str] = []
    lines.append('File type = "ooTextFile"\n')
    lines.append('Object class = "TextGrid"\n')
    lines.append("\n")
    lines.append("xmin = 0\n")
    lines.append(f"xmax = {q(duration)}\n")
    lines.append("tiers? <exists>\n")
    lines.append("size = 2\n")
    lines.append("item []:\n")

    lines.append("\titem [1]:\n")
    lines.append('\t\tclass = "IntervalTier"\n')
    lines.append('\t\tname = "utterance"\n')
    lines.append("\t\txmin = 0\n")
    lines.append(f"\t\txmax = {q(duration)}\n")
    lines.append(f"\t\tintervals: size = {len(segments)}\n")

    for index, segment in enumerate(segments, start=1):
        start = max(0.0, float(segment.get("start", 0.0)))
        end = min(duration, float(segment.get("end", duration)))
        text = _clean_label(str(segment.get("text", ""))).replace('"', '\\"') or " "
        lines.append(f"\t\tintervals [{index}]:\n")
        lines.append(f"\t\t\txmin = {q(start)}\n")
        lines.append(f"\t\t\txmax = {q(end)}\n")
        lines.append(f"\t\t\ttext = \"{text}\"\n")

    word_intervals: list[dict[str, Any]] = []
    for segment in segments:
        for word in segment.get("words", []) or []:
            if word.get("start") is None or word.get("end") is None:
                continue
            word_intervals.append(word)

    lines.append("\titem [2]:\n")
    lines.append('\t\tclass = "IntervalTier"\n')
    lines.append('\t\tname = "words"\n')
    lines.append("\t\txmin = 0\n")
    lines.append(f"\t\txmax = {q(duration)}\n")
    lines.append(f"\t\tintervals: size = {len(word_intervals)}\n")

    for index, word in enumerate(word_intervals, start=1):
        start = max(0.0, float(word.get("start", 0.0)))
        end = min(duration, float(word.get("end", duration)))
        label = _clean_label(str(word.get("word", ""))).replace('"', '\\"') or " "
        lines.append(f"\t\tintervals [{index}]:\n")
        lines.append(f"\t\t\txmin = {q(start)}\n")
        lines.append(f"\t\t\txmax = {q(end)}\n")
        lines.append(f"\t\t\ttext = \"{label}\"\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path
