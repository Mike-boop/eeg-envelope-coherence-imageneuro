#!/usr/bin/env python3
"""Cross-spectral coherence between audio envelope features and EEG, per recording.

Ported from ``neural_tracking_across_spectrum``'s per-dataset ``get_coherences.py`` scripts:
same Welch window sizing (5 s windows, 80% overlap) and average-referenced EEG. Unlike those
scripts, which batched every subject's coherence into one file per stimulus, this writes one
coherence file per EEG recording (keyed by its own filename) plus one stimulus-PSD file per
audio stimulus, and computes against this repo's ``audio_features`` factory outputs instead of
re-deriving envelopes inline.

Mastoid/reference channels (A1/A2, typed "misc" in the recordings) are dropped rather than
re-referenced: the old scripts' attempt to re-reference them by subtracting the post-average-
reference EEG mean was a no-op (that mean is ~0 by construction), leaving them effectively raw
in the old outputs.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mne
import numpy as np
from scipy.signal import welch
from tqdm import tqdm

from coherence.coherence import coherence_from_prep, null_coherence_shuffled_windows_from_prep, prepare_coherence
from transcription.config import load_config

mne.set_log_level("WARNING")

# Audio feature arrays (keys in audio_features_dir/*.npz, produced by
# scripts/feature_extraction/extract_features.py) to compute coherence against, and the sample
# rate they + the EEG recordings share.
SUBBAND_FEATURE_KEY = "subband_envelopes"
GAMMATONE_FEATURE_KEY = "gammatone_envelope"
STIMULUS_FS = 1024.0

# Welch windowing, shared between the stimulus PSDs and the coherence calculation.
WINDOW_SECONDS = 5.0
OVERLAP_FRACTION = 0.8

# Coherence is computed in complex128 (numpy's default) but stored as complex64: subband_coherence
# is by far the largest array on disk (n_freqs x n_subbands x n_eeg_channels), and coherence
# magnitudes/phases don't need double precision once the FFT/averaging itself is done.
COHERENCE_DTYPE = np.complex64

# Null coherence: per recording, N_NULL draws of null_coherence_shuffled_windows in "global_shift"
# mode — a random circular shift of the EEG's window order, rather than a full window shuffle.
# A full shuffle destroys the window-to-window correlation that Welch overlap gives real data
# too, making that null unrealistically low; a shift preserves it while still mismatching audio
# and EEG. NULL_MIN_SHIFT_SECONDS is the requested lower bound (1 minute); the upper bound is
# each stimulus's own duration minus that same margin (see null_shift_bounds), shrinking further
# for the handful of stimuli too short to leave a minute of clearance on both sides.
# NULL_RNG_SEED is fixed so reruns reproduce the same null draws.
N_NULL = 5
NULL_MIN_SHIFT_SECONDS = 60.0
NULL_RNG_SEED = 0


def null_shift_bounds(duration_seconds: float) -> tuple[float, float]:
    """(min_shift_seconds, max_shift_seconds) for the global-shift null, scaled to the stimulus's
    own duration: max is duration minus the same margin as the minimum, so shifts stay clear of
    wrapping back near a 0-lag pairing. For the rare stimulus too short to leave
    NULL_MIN_SHIFT_SECONDS of clearance on both sides (available frames span roughly
    ``duration - WINDOW_SECONDS`` seconds), both bounds shrink to a quarter of what's available.
    """
    available = duration_seconds - WINDOW_SECONDS
    min_shift = NULL_MIN_SHIFT_SECONDS
    if 2.0 * min_shift >= available:
        min_shift = available / 4.0
    max_shift = available - min_shift
    return min_shift, max_shift

# EEG recordings for a stimulus are processed by a pool of worker processes (peak RSS for one
# recording's real + null coherence is ~7.5 GB measured on real data, so this is kept conservative
# rather than matching core count). Set to 1 to process every recording sequentially in-process.
N_WORKERS = 3

# EEG filenames look like "sub-043_-_ses-001_-_trial-003_-_cond-natural_-_audio01-podcast_14_-_eeg.fif";
# this glob matches every recording of a given stimulus across subjects/sessions/trials/conditions.
EEG_GLOB_TEMPLATE = "*_-_audio01-{audio_name}_-_eeg.fif"


def resolve_configured_inputs(
    config: dict[str, Any], root: str | Path | None = None
) -> tuple[Path, Path, Path, Path]:
    """Resolve the audio-feature, EEG, coherence-output and stimulus-PSD directories."""
    root_path = Path(root or ROOT).expanduser().resolve()

    def _resolve(key: str, default: str) -> Path:
        path = Path(config.get(key, default)).expanduser()
        return path if path.is_absolute() else (root_path / path).resolve()

    audio_features_dir = _resolve("audio_features_dir", "data/audio_features")
    eeg_dir = _resolve("eeg_dir", "data/eeg")
    coherence_dir = _resolve("coherence_dir", "data/coherence")
    stimulus_psd_dir = _resolve("stimulus_psd_dir", "data/stimulus_psd")

    return audio_features_dir, eeg_dir, coherence_dir, stimulus_psd_dir


def find_matching_eeg_files(eeg_dir: Path, audio_name: str) -> list[Path]:
    """Every EEG recording of a given audio stimulus, across subjects/sessions/trials."""
    return sorted(eeg_dir.glob(EEG_GLOB_TEMPLATE.format(audio_name=audio_name)))


def stimulus_window_sizes(fs: float) -> tuple[int, int]:
    nperseg = int(round(WINDOW_SECONDS * fs))
    noverlap = int(round(OVERLAP_FRACTION * nperseg))
    return nperseg, noverlap


def load_stimulus_features(npz_path: Path) -> np.ndarray:
    """Stack the subband envelopes and gammatone envelope into one (n_features, n_samples) array."""
    data = np.load(npz_path)
    return np.concatenate([data[SUBBAND_FEATURE_KEY], data[GAMMATONE_FEATURE_KEY]], axis=0)


def load_eeg(fif_path: Path) -> tuple[np.ndarray, float]:
    """Average-referenced EEG data (eeg-type channels only) and the recording's sample rate."""
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    raw.pick("eeg")
    raw.set_eeg_reference("average", verbose=False)
    return raw.get_data(), float(raw.info["sfreq"])


def compute_stimulus_psd(
    features: np.ndarray, fs: float, nperseg: int, noverlap: int
) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD (n_freqs, n_features) of each feature channel, same windowing as the coherence call."""
    freqs, psd = welch(
        features, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hann", detrend="constant", axis=-1
    )
    return freqs, psd.T


def compute_and_save_coherence(
    eeg_path: Path,
    features: np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int,
    n_subbands: int,
    audio_name: str,
    duration: float,
    coherence_dir: Path,
) -> Path:
    """Load one EEG recording and save its real + null coherence against ``features``.

    A standalone, module-level function (rather than inlined in ``process_audio_stimulus``'s
    loop) so it can be handed to worker processes as-is.
    """
    output_path = coherence_dir / f"{eeg_path.stem}_coherence.npz"
    if output_path.exists():
        return output_path

    eeg_data, eeg_fs = load_eeg(eeg_path)
    if eeg_fs != fs:
        raise ValueError(f"{eeg_path.name}: EEG sample rate {eeg_fs} does not match stimulus rate {fs}")

    # One windowing+FFT pass (via prepare_coherence) feeds both the real coherence estimate and
    # the null draws below, instead of each redoing it independently.
    prep = prepare_coherence(features, eeg_data, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1)
    result = coherence_from_prep(prep)
    min_shift_seconds, max_shift_seconds = null_shift_bounds(duration)
    null_Cxy = null_coherence_shuffled_windows_from_prep(
        prep,
        n_null=N_NULL,
        min_shift_seconds=min_shift_seconds,
        max_shift_seconds=max_shift_seconds,
        mode="global_shift",
        rng=NULL_RNG_SEED,
    )

    np.savez(
        output_path,
        audio_name=audio_name,
        duration=duration,
        frequencies=result.freqs,
        subband_coherence=result.Cxy[:, :n_subbands, :].astype(COHERENCE_DTYPE),
        gammatone_coherence=result.Cxy[:, n_subbands:, :].astype(COHERENCE_DTYPE),
        eeg_psd=result.Pyy,
        null_subband_coherence=null_Cxy[:, :, :n_subbands, :].astype(COHERENCE_DTYPE),
        null_gammatone_coherence=null_Cxy[:, :, n_subbands:, :].astype(COHERENCE_DTYPE),
    )
    return output_path


def process_audio_stimulus(npz_path: Path, eeg_dir: Path, coherence_dir: Path, stimulus_psd_dir: Path) -> None:
    audio_name = npz_path.stem
    features = load_stimulus_features(npz_path)
    n_subbands = features.shape[0] - 1
    fs = STIMULUS_FS
    duration = features.shape[1] / fs
    nperseg, noverlap = stimulus_window_sizes(fs)

    stimulus_psd_path = stimulus_psd_dir / f"{audio_name}_stimulus_psd.npz"
    if not stimulus_psd_path.exists():
        freqs, psd = compute_stimulus_psd(features, fs, nperseg, noverlap)
        np.savez(
            stimulus_psd_path,
            audio_name=audio_name,
            duration=duration,
            frequencies=freqs,
            subband_psd=psd[:, :n_subbands],
            gammatone_psd=psd[:, n_subbands:],
        )

    eeg_files = find_matching_eeg_files(eeg_dir, audio_name)
    pending = [
        eeg_path for eeg_path in eeg_files if not (coherence_dir / f"{eeg_path.stem}_coherence.npz").exists()
    ]

    if N_WORKERS <= 1 or len(pending) <= 1:
        for eeg_path in tqdm(pending, desc=audio_name, unit="recording", leave=False):
            compute_and_save_coherence(eeg_path, features, fs, nperseg, noverlap, n_subbands, audio_name, duration, coherence_dir)
        return

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [
            executor.submit(
                compute_and_save_coherence,
                eeg_path,
                features,
                fs,
                nperseg,
                noverlap,
                n_subbands,
                audio_name,
                duration,
                coherence_dir,
            )
            for eeg_path in pending
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=audio_name, unit="recording", leave=False):
            future.result()


def main() -> int:
    config = load_config(ROOT / "src" / "package_data" / "config.json")
    audio_features_dir, eeg_dir, coherence_dir, stimulus_psd_dir = resolve_configured_inputs(config, ROOT)

    coherence_dir.mkdir(parents=True, exist_ok=True)
    stimulus_psd_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(audio_features_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz feature files found in {audio_features_dir}")

    for npz_path in tqdm(npz_files, desc="Audio stimuli", unit="stimulus"):
        process_audio_stimulus(npz_path, eeg_dir, coherence_dir, stimulus_psd_dir)

    print(f"Wrote coherence files to {coherence_dir} and stimulus PSDs to {stimulus_psd_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
