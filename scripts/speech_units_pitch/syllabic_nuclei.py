#!/usr/bin/env python3
"""
Extracts the syllabic nuclei onsets from the speech files (.wav files).

In the first instance, use heuristic linguistic rules for Dutch to identify nuclei from the TextGrid files, based on /home/gu08wuvu/neural_tracking_across_spectrum/stimulus_analysis/textgrid_analysis/words_nuclei.ipynb
Also uses a second DeJong method by applying the Praat script "syllabic_nuclei.praat" to the .wav files to identify syllabic nuclei. 

Outputs JSON files with keys equal to audio names and values equal to a list of syllabic nuclei onsets (in seconds) for each audio file. One JSON file per method.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from praatio import textgrid
from tqdm import tqdm

from _common import SCRIPT_DIR, audio_files, is_speech_label, load_paths, textgrid_files, write_json

# ---------------------------------------------------------------- heuristic method

# Vowel(-like) phones actually present in the corpus.
DUTCH_VOWEL_LIKE = {
    "a", "aː",
    "eː",
    "i", "iː",
    "oː",
    "u", "uː",
    "y", "yː",
    "øː",
    "œ",
    "ɑ",
    "ɔ",
    "ɛ",
    "ɪ",
    "ʏ",
    "ɛ̈",
    "ʏ̈",
    "ə",
}
# Sonorants we are willing to treat as syllabic when there is no vowel.
SONORANT_CONS = {"m", "n", "ŋ", "l", "r"}
DIPHTHONG_SEQS = {
    ("øː", "ʏ"),   # ui/eu (nauwkeurige, bespeuren, reusachtige, heuse)
    ("iː", "eː"),  # ie (wie, die)
    ("iː", "ɛ"),   # ie variant
    ("uː", "eː"),  # oe (hoe, toe, naartoe)
    ("uː", "ɛ"),   # oe variant (voeten, proeven, toevallig, zoeken, hoeveel, vroegen)
    ("eː", "ɪ"),   # nucleus + offglide in ik, is, in, hoe
    ("aː", "ɑ"),   # aa split in daar-/waar- words
}

MIN_SEPARATION_S = 0.02


def normalise_phone(label: str) -> str:
    """Lowercase and strip spaces/digits (for stress etc. if added later)."""
    lab = label.strip().lower()
    lab = re.sub(r"[0-9ˈˌ]", "", lab)
    return lab


def is_vowel_like(label: str) -> bool:
    """True if phone is one of the vowel(-like) symbols in this corpus."""
    return normalise_phone(label) in DUTCH_VOWEL_LIKE


def get_syllable_nuclei(phone_tier, word_tier, min_separation: float = 0.03) -> list[tuple[float, str]]:
    """
    Estimate syllable nuclei.

    1. First pass: vowel-like phones -> nuclei, with explicit diphthong handling.
    2. Second pass: for each word with no nucleus, choose a sonorant consonant
       inside that word as a fallback nucleus.

    Nucleus times are phone midpoints (or diphthong span midpoints).
    """
    phones = list(phone_tier.entries)
    words = list(word_tier.entries)

    nuclei: list[tuple[float, str]] = []
    last_nucleus_time: float | None = None
    n_phones = len(phones)

    i = 0
    while i < n_phones:
        ph = phones[i]
        raw_label = ph.label.strip()
        lab = normalise_phone(raw_label)

        # Skip non-vowels
        if not lab or not is_vowel_like(raw_label):
            i += 1
            continue

        # ---------- DIPHTHONG CHECK ----------
        diph_used = False
        if i + 1 < n_phones:
            ph2 = phones[i + 1]
            lab2 = normalise_phone(ph2.label)

            # The two phones must actually abut. Empty intervals are dropped when
            # the TextGrid is read, so the phones either side of an utterance seam
            # look adjacent in this list; merging them would put a "diphthong"
            # nucleus in the middle of the silence between two utterances.
            contiguous = abs(ph2.start - ph.end) < 1e-6

            if contiguous and (lab, lab2) in DIPHTHONG_SEQS:
                # Combined nucleus
                t = 0.5 * (ph.start + ph2.end)
                combined_label = f"{raw_label}+{ph2.label.strip()}"

                if last_nucleus_time is None or (t - last_nucleus_time) >= min_separation:
                    nuclei.append((t, combined_label))
                    last_nucleus_time = t

                i += 2
                diph_used = True

        if not diph_used:
            # Monophthong nucleus (midpoint)
            mid = 0.5 * (ph.start + ph.end)
            if last_nucleus_time is None or (mid - last_nucleus_time) >= min_separation:
                nuclei.append((mid, raw_label))
                last_nucleus_time = mid

            i += 1

    # ---------- WORD-LEVEL FALLBACK ----------
    def word_has_nucleus(w_start: float, w_end: float) -> bool:
        return any(w_start <= t < w_end for t, _ in nuclei)

    for w in words:
        # MFA's <eps> pause intervals are not words and must not be given a
        # nucleus. They contain only `sil`, so the sonorant search below would
        # come up empty anyway — but filter by intent rather than by luck.
        if not is_speech_label(w.label):
            continue

        w_start, w_end = w.start, w.end
        if word_has_nucleus(w_start, w_end):
            continue

        # Find best sonorant inside the word
        w_mid = 0.5 * (w_start + w_end)
        best = None
        best_dist = None
        best_time = None

        for ph in phones:
            lab = normalise_phone(ph.label)
            if lab not in SONORANT_CONS:
                continue
            ph_mid = 0.5 * (ph.start + ph.end)
            if not (w_start <= ph_mid < w_end):
                continue
            dist = abs(ph_mid - w_mid)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = ph
                best_time = ph_mid

        if best is not None:
            nuclei.append((best_time, best.label.strip()))

    nuclei.sort(key=lambda x: x[0])
    return nuclei


def run_heuristic(paths) -> None:
    """Nuclei from the aligned phone/word tiers, using Dutch vowel heuristics."""
    nuclei_onsets: dict[str, list[float]] = {}
    syllabic_rates: dict[str, float] = {}

    for textgrid_path in textgrid_files(paths.textgrid_dir):
        audio_name = textgrid_path.stem
        tg = textgrid.openTextgrid(str(textgrid_path), includeEmptyIntervals=False)
        duration = float(tg.maxTimestamp - tg.minTimestamp)

        nuclei = get_syllable_nuclei(tg.getTier("phones"), tg.getTier("words"), min_separation=MIN_SEPARATION_S)
        times = [float(t) for t, _ in nuclei]

        nuclei_onsets[audio_name] = times
        syllabic_rates[audio_name] = len(times) / duration

        intervals = np.percentile(np.diff(times), [5, 25, 50, 75, 95]) if len(times) > 1 else []
        print(f"{audio_name}: {len(times)} nuclei, {syllabic_rates[audio_name]:.2f} Hz over {duration:.1f} s")
        print(f"  nucleus intervals (s), [5 25 50 75 95]th pct: {np.round(intervals, 3)}")

    write_json(nuclei_onsets, paths.output_dir / "nuclei_onsets_heuristic.json")
    write_json(syllabic_rates, paths.output_dir / "syllabic_rates_heuristic.json")


# ------------------------------------------------------------------ deJong method

PRAAT_SCRIPT = SCRIPT_DIR / "syllable_nuclei_v3.praat"
# de Jong & Wempe's recommended settings; "None" pre-processing keeps the raw signal.
PRAAT_ARGS = [
    "None",              # Pre_processing
    "-25",               # Silence_threshold (dB)
    "2",                 # Minimum_dip_near_peak (dB)
    "0.3",               # Minimum_pause_duration (s)
    "yes",               # Detect_Filled_Pauses
    "Dutch",             # Language
    "1.0",               # Filled_Pause_threshold
    "TextGrid(s) only",  # Data (destination of output)
    "OverWriteData",     # DataCollectionType
    "no",                # Keep_Objects — release each Sound/TextGrid after saving
]


def run_praat(wav_path: Path, staging_dir: Path) -> Path:
    """
    Run syllable_nuclei_v3.praat on one wav and return the `.auto.TextGrid` it writes.

    The script writes its output next to its input, so the wav is symlinked into
    `staging_dir` first and the result is left there rather than in raw_waves_dir.
    The FileSpec has to be a glob — Praat's "Create Strings as file list" rejects a
    bare filename — so the staging directory holds exactly one wav at a time.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    for stale in staging_dir.glob("*.wav"):
        stale.unlink()

    link_path = staging_dir / wav_path.name
    link_path.symlink_to(wav_path)
    try:
        result = subprocess.run(
            ["praat", "--run", str(PRAAT_SCRIPT), f"{staging_dir}/*.wav", *PRAAT_ARGS],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
        )
    finally:
        link_path.unlink(missing_ok=True)

    output_path = staging_dir / f"{wav_path.stem}.auto.TextGrid"
    if result.returncode != 0 or not output_path.exists():
        raise RuntimeError(
            f"Praat failed on {wav_path.name} (exit {result.returncode})\n"
            f"stdout: {result.stdout.strip()}\nstderr: {result.stderr.strip()}"
        )
    return output_path


def run_dejong(paths, *, force: bool = False) -> None:
    """Nuclei from intensity peaks in the waveform, via the de Jong Praat detector."""
    if shutil.which("praat") is None:
        raise RuntimeError("`praat` not found on PATH; needed for the de Jong syllable nuclei method.")

    # The Praat script writes its output next to its input, so give it a scratch
    # directory of symlinks rather than letting it litter raw_waves_dir.
    work_dir = paths.output_dir / "dejong_textgrids"
    work_dir.mkdir(parents=True, exist_ok=True)

    nuclei_onsets: dict[str, list[float]] = {}
    syllabic_rates: dict[str, float] = {}

    wav_paths = audio_files(paths.audio_dir)

    for wav_path in tqdm(wav_paths, desc="de Jong nuclei", unit="file"):
        audio_name = wav_path.stem
        auto_path = work_dir / f"{audio_name}.auto.TextGrid"

        if force or not auto_path.exists():
            run_praat(wav_path, work_dir)

        tg = textgrid.openTextgrid(str(auto_path), includeEmptyIntervals=False)
        times = [float(point.time) for point in tg.getTier("Nuclei").entries]
        duration = float(tg.maxTimestamp - tg.minTimestamp)

        nuclei_onsets[audio_name] = np.round(times, 3).tolist()
        syllabic_rates[audio_name] = len(times) / duration

    write_json(nuclei_onsets, paths.output_dir / "nuclei_onsets_dejong.json")
    write_json(syllabic_rates, paths.output_dir / "syllabic_rates_dejong.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["heuristic", "dejong", "both"],
        default="both",
        help="Which nucleus detector to run (default: both).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run Praat even where a .auto.TextGrid already exists.",
    )
    args = parser.parse_args()

    paths = load_paths(
        require_textgrids=args.method in ("heuristic", "both"),
        require_audio=args.method in ("dejong", "both"),
    )

    if args.method in ("heuristic", "both"):
        print("== heuristic (TextGrid vowel/sonorant rules) ==")
        run_heuristic(paths)
    if args.method in ("dejong", "both"):
        print("== de Jong (Praat intensity peaks) ==")
        run_dejong(paths, force=args.force)

    print(f"Wrote syllable nuclei to {paths.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
