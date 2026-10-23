#!/usr/bin/env python3
"""
Extracts word onsets..

Directly extracts the word onsets from the Praat TextGrid files. Outputs a JSON file with keys equal to audio names and values equal to a list of word onsets (in seconds) for each audio file.
Based on /home/gu08wuvu/neural_tracking_across_spectrum/stimulus_analysis/textgrid_analysis/words_nuclei.ipynb
"""

from __future__ import annotations

import numpy as np
from praatio import textgrid

from _common import is_speech_label, load_paths, textgrid_files, write_json


def main() -> int:
    paths = load_paths(require_textgrids=True)

    word_onsets: dict[str, list[float]] = {}
    word_rates: dict[str, float] = {}

    for textgrid_path in textgrid_files(paths.textgrid_dir):
        audio_name = textgrid_path.stem
        tg = textgrid.openTextgrid(str(textgrid_path), includeEmptyIntervals=False)
        duration = float(tg.maxTimestamp - tg.minTimestamp)

        onsets = [float(entry.start) for entry in tg.getTier("words").entries if is_speech_label(entry.label)]
        word_onsets[audio_name] = onsets
        word_rates[audio_name] = len(onsets) / duration

        percentiles = np.percentile(np.diff(onsets), [5, 25, 50, 75, 95]) if len(onsets) > 1 else []
        print(f"{audio_name}: {len(onsets)} words, {word_rates[audio_name]:.2f} Hz over {duration:.1f} s")
        print(f"  word onset intervals (s), [5 25 50 75 95]th pct: {np.round(percentiles, 3)}")

    write_json(word_onsets, paths.output_dir / "word_onsets.json")
    write_json(word_rates, paths.output_dir / "word_rates.json")

    print(f"Wrote word onsets/rates for {len(word_onsets)} recording(s) to {paths.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
