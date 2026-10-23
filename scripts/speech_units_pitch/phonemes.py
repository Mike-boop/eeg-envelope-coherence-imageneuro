#!/usr/bin/env python3
"""
Extracts phoneme onsets from TextGrid files and outputs a JSON file with keys equal to audio names and values equal to a list of phoneme onsets (in seconds) for each audio file.
/home/gu08wuvu/neural_tracking_across_spectrum/stimulus_analysis/textgrid_analysis/phonemes.ipynb

Specifically produces files for:
- fricatives
- plosives
- vowels
- consonants
- sonorants
- phonemes (overall)
"""

from __future__ import annotations

from pathlib import Path

from praatio import textgrid

from _common import is_speech_label, load_paths, textgrid_files, write_json

# IPA phone inventory of the `dutch_cv` MFA dictionary, grouped into the categories
# we report. The groups overlap by design: `sonorants` is vowels + nasals + liquids +
# glides, and every phone is in either `vowels` or `consonants`.
PHONETIC_CATEGORIES: dict[str, set[str]] = {
    "vowels": {
        "oː", "uː", "øː", "y", "ɛ̈", "yː", "ʏ", "iː", "ɛ", "ɔ",
        "ʏ̈", "ɪ", "i", "eː", "a", "u", "œ", "aː", "ɑ", "ə",
    },
    "consonants": {
        "h", "m", "ŋ", "f", "k", "v", "c", "d", "l", "ɣ",
        "p", "ʃ", "t", "j", "ɥ", "x", "r", "w", "z",
        "ʋ", "b", "s", "n",
    },
    "sonorants": {
        # vowels
        "oː", "uː", "øː", "y", "ɛ̈", "yː", "ʏ", "iː", "ɛ", "ɔ",
        "ʏ̈", "ɪ", "i", "eː", "a", "u", "œ", "aː", "ɑ", "ə",
        # nasals
        "m", "n", "ŋ",
        # liquids
        "l", "r",
        # approximants / glides
        "j", "w", "ɥ", "ʋ",
    },
    "fricatives": {"f", "v", "s", "z", "ʃ", "x", "ɣ", "h"},
    "plosives": {"p", "b", "t", "d", "k", "c"},
}


def phone_entries(textgrid_path: Path) -> tuple[list, float]:
    """Return the speech phones of one TextGrid and the recording duration in seconds."""
    tg = textgrid.openTextgrid(str(textgrid_path), includeEmptyIntervals=False)
    entries = [entry for entry in tg.getTier("phones").entries if is_speech_label(entry.label)]
    duration = float(tg.maxTimestamp - tg.minTimestamp)
    return entries, duration


def main() -> int:
    paths = load_paths(require_textgrids=True)

    phoneme_onsets: dict[str, list[float]] = {}
    phoneme_rates: dict[str, float] = {}
    categorical_onsets: dict[str, dict[str, list[float]]] = {}
    categorical_rates: dict[str, dict[str, float]] = {}
    observed_phones: set[str] = set()

    for textgrid_path in textgrid_files(paths.textgrid_dir):
        audio_name = textgrid_path.stem
        entries, duration = phone_entries(textgrid_path)
        observed_phones.update(entry.label.strip() for entry in entries)

        phoneme_onsets[audio_name] = [float(entry.start) for entry in entries]
        phoneme_rates[audio_name] = len(entries) / duration

        categorical_onsets[audio_name] = {}
        categorical_rates[audio_name] = {}
        for category, phonemes in PHONETIC_CATEGORIES.items():
            onsets = [float(entry.start) for entry in entries if entry.label.strip() in phonemes]
            categorical_onsets[audio_name][category] = onsets
            categorical_rates[audio_name][category] = len(onsets) / duration

        print(f"{audio_name}: {len(entries)} phones, {phoneme_rates[audio_name]:.2f} Hz over {duration:.1f} s")

    # A phone the category map does not know about is silently absent from every
    # categorical count, so surface it rather than let the rates quietly under-report.
    uncategorised = observed_phones - set().union(*PHONETIC_CATEGORIES.values())
    if uncategorised:
        print(f"WARNING: {len(uncategorised)} phone(s) in no category: {sorted(uncategorised)}")

    write_json(phoneme_onsets, paths.output_dir / "phoneme_onsets.json")
    write_json(phoneme_rates, paths.output_dir / "phoneme_rates.json")
    write_json(categorical_rates, paths.output_dir / "categorical_phoneme_rates.json")
    for category in PHONETIC_CATEGORIES:
        onsets_by_audio = {name: cats[category] for name, cats in categorical_onsets.items()}
        write_json(onsets_by_audio, paths.output_dir / f"{category}_onsets.json")

    print(f"Wrote phoneme onsets/rates for {len(phoneme_onsets)} recording(s) to {paths.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
