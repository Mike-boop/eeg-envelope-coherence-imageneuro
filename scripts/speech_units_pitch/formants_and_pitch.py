#!/usr/bin/env python3
"""
Extracts F0 and formant histograms from the speech files (.wav files)

For pitch (F0): histogram (of counts) of F0 with 2561 bin centres corresponding to range 0-512Hz with 0.2Hz resolution.
Output should be a npz files with fields [counts, bin_centres, audio_duration] (one file per stimulus .wav file). 

For formants: histogram of first 5 formants with bin edges corresponding to range 0-8000Hz with 10Hz resolution.
Output should be a npz files with fields [counts, bin_centres, audio_duration] (one file per stimulus .wav file).

Logic for these calulations based on /home/gu08wuvu/neural_tracking_across_spectrum/stimulus_analysis/acoustic_analysis/pitch_formants.ipynb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import parselmouth
from scipy.interpolate import interp1d
from tqdm import tqdm

from _common import audio_files, load_paths

# Pitch histogram bins. The 2561 centres are 0, 0.2, ..., 512.0 Hz — the same
# frequency axis the coherence spectra are computed on, so the two can be
# overlaid directly. Edges sit half a bin either side of each centre.
PITCH_RESOLUTION_HZ = 0.2
PITCH_N_BINS = 2561
PITCH_BIN_CENTRES = np.arange(PITCH_N_BINS) * PITCH_RESOLUTION_HZ
PITCH_BIN_EDGES = np.append(PITCH_BIN_CENTRES - PITCH_RESOLUTION_HZ / 2, PITCH_BIN_CENTRES[-1] + PITCH_RESOLUTION_HZ / 2)

# Formant histogram bins: edges 0, 10, ..., 8000 Hz.
FORMANT_RESOLUTION_HZ = 10.0
FORMANT_BIN_EDGES = np.arange(0.0, 8000.0 + FORMANT_RESOLUTION_HZ, FORMANT_RESOLUTION_HZ)
FORMANT_BIN_CENTRES = FORMANT_BIN_EDGES[:-1] + FORMANT_RESOLUTION_HZ / 2

N_FORMANTS = 5
TIME_STEP_S = 0.01
PITCH_FLOOR_HZ = 30.0
PITCH_CEILING_HZ = 370.0
# Formant frames quieter than this percentile of the intensity contour are
# dropped: formant tracking in near-silence returns noise, not vocal tract poles.
INTENSITY_PERCENTILE = 25.0


def load_mono_sound(wav_path: Path) -> parselmouth.Sound:
    """Read a wav as a single-channel parselmouth Sound (first channel if multichannel)."""
    sound = parselmouth.Sound(str(wav_path))
    if sound.n_channels > 1:
        sound = sound.extract_channel(1)
    return sound


def pitch_histogram(sound: parselmouth.Sound) -> np.ndarray:
    """Counts of voiced F0 estimates per pitch bin."""
    pitch = sound.to_pitch(time_step=TIME_STEP_S, pitch_floor=PITCH_FLOOR_HZ, pitch_ceiling=PITCH_CEILING_HZ)
    f0 = pitch.selected_array["frequency"]  # Hz; unvoiced frames are 0
    voiced = f0[f0 > 0]
    counts, _ = np.histogram(voiced, bins=PITCH_BIN_EDGES)
    return counts


def formant_histograms(sound: parselmouth.Sound) -> np.ndarray:
    """Counts per formant bin for F1..F5, restricted to the louder frames. Shape (5, n_bins)."""
    formant = sound.to_formant_burg(
        time_step=TIME_STEP_S,
        max_number_of_formants=N_FORMANTS,
        maximum_formant=5500.0,
        window_length=0.025,
        pre_emphasis_from=50.0,
    )
    times = np.arange(0.0, sound.duration, TIME_STEP_S)

    intensity = sound.to_intensity(time_step=TIME_STEP_S, minimum_pitch=60)
    intensity_fn = interp1d(
        intensity.xs(),
        intensity.values.squeeze(),
        kind="nearest",
        bounds_error=False,
        fill_value=np.nan,
    )
    intensity_values = intensity_fn(times)
    loud = intensity_values > np.nanpercentile(intensity_values, INTENSITY_PERCENTILE)

    counts = np.zeros((N_FORMANTS, len(FORMANT_BIN_CENTRES)), dtype=np.int64)
    for index in range(N_FORMANTS):
        values = np.array([formant.get_value_at_time(index + 1, t) for t in times])
        counts[index], _ = np.histogram(values[loud], bins=FORMANT_BIN_EDGES)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute stimuli whose output npz already exists.",
    )
    args = parser.parse_args()

    paths = load_paths(require_audio=True)
    pitch_dir = paths.output_dir / "pitch"
    formant_dir = paths.output_dir / "formants"
    pitch_dir.mkdir(parents=True, exist_ok=True)
    formant_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for wav_path in tqdm(audio_files(paths.audio_dir), desc="Pitch and formants", unit="file"):
        audio_name = wav_path.stem
        pitch_path = pitch_dir / f"{audio_name}_pitch.npz"
        formant_path = formant_dir / f"{audio_name}_formants.npz"
        if not args.force and pitch_path.exists() and formant_path.exists():
            continue

        sound = load_mono_sound(wav_path)
        duration = float(sound.duration)

        np.savez(
            pitch_path,
            counts=pitch_histogram(sound),
            bin_centres=PITCH_BIN_CENTRES,
            audio_duration=duration,
        )
        np.savez(
            formant_path,
            counts=formant_histograms(sound),
            bin_centres=FORMANT_BIN_CENTRES,
            audio_duration=duration,
        )
        written += 1

    print(f"Wrote pitch and formant histograms for {written} recording(s) to {paths.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
