#!/usr/bin/env python3
"""Pool get_coherence.py's per-trial coherence into per-stimulus aggregates.

get_coherence.py writes one real+null coherence file per EEG recording, with only N_NULL=5 null
draws each. This script pools across every trial of a stimulus into a single duration-weighted
average, and computes fresh null draws so N_NULL can be pushed higher for the pooled estimate
without inflating get_coherence.py's own per-trial output.

Only the null coherence is actually computed here. The real coherence and both auto-power spectra
(Pxx, Pyy) don't depend on N_NULL, and get_coherence.py already saved all of them -- recomputing
them would mean redoing the Welch/matmul pass over every window for no new information, so they're
loaded instead (per-recording ``{eeg_stem}_coherence.npz``: subband/gammatone_coherence, eeg_psd;
per-stimulus ``{audio_name}_stimulus_psd.npz``: subband/gammatone_psd). The one piece a null draw
needs that *isn't* saved anywhere is the per-window complex spectrum itself -- Pxx/Pyy are a
real-valued summary of it, not a substitute -- so that is recomputed, once per stimulus for the
audio side and once per trial for the EEG side.

Currently pools over stimuli (grouping by ``audio_name``). Pooling over participants instead
(grouping by ``sub``) is a planned follow-up, as a switch rather than a second always-on pass.

Inclusion: every ``cond-natural`` recording is pooled, deliberately including the ``_shifted``
(shifted-audio calibration) and ``_artefact`` stimuli -- they get their own pooled output like any
other stimulus. A shifted-audio stimulus is the useful case to have pooled: it's the empirical
floor the genuine stimuli's coherence should be read against, and pooling it the same way makes
that comparison like-for-like.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (SRC, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

from coherence.coherence import WindowedSpectrum, _auto_power, _conj_transpose_for_matmul, _onesided_fold, windowed_fft
from transcription.config import load_config

from get_coherence import (
    COHERENCE_DTYPE,
    NULL_RNG_SEED,
    STIMULUS_FS,
    load_eeg,
    load_stimulus_features,
    null_shift_bounds,
    resolve_configured_inputs,
    stimulus_window_sizes,
)

mne.set_log_level("WARNING")

# Null draws are recomputed from scratch here rather than reused from get_coherence.py, so this is
# free to exceed its N_NULL=5 -- that's the point of pooling. Cost is linear in N_NULL on both
# axes: runtime (~0.4 s per draw per trial, measured at F=2561 Cx=61 Cy=64 on an unloaded machine)
# and the accumulator below. Raise with both in mind.
N_NULL = 50

# The running null sum is (N_NULL, F, Cx, Cy) and is the script's dominant memory cost: ~4 GB at
# N_NULL=50 in complex64, double that in complex128. complex64 is enough here -- it's the dtype
# the result is saved as either way, and a handful of trials' worth of averaging loses far less
# precision than that final downcast already does.
ACCUMULATOR_DTYPE = np.complex64

# Only "natural" listening trials are pooled; see main().
POOLED_CONDITION = "natural"


def resolve_pooled_inputs(
    config: dict[str, Any], root: str | Path | None = None
) -> tuple[Path, Path, Path, Path, Path]:
    """get_coherence.py's directories, plus this script's own per-stimulus output directory,
    nested under the configured ``null_coherence_dir`` (one directory this script owns, on
    whatever drive has room for it) rather than its own independently-configured location."""
    root_path = Path(root or ROOT).expanduser().resolve()
    audio_features_dir, eeg_dir, coherence_dir, stimulus_psd_dir = resolve_configured_inputs(config, root_path)

    path = Path(config.get("null_coherence_dir", "data/null_coherence")).expanduser()
    null_coherence_dir = path if path.is_absolute() else (root_path / path).resolve()
    stimulus_coherence_dir = null_coherence_dir / "stimulus_coherence"

    return audio_features_dir, eeg_dir, coherence_dir, stimulus_psd_dir, stimulus_coherence_dir


def parse_coherence_filename(coherence_file: Path) -> dict[str, Any]:
    """Parse a get_coherence.py output filename into its components.

    e.g. "sub-084_-_ses-001_-_trial-002_-_cond-natural_-_audio01-audiobook_14_2_-_eeg_coherence.npz"
    """
    parts = coherence_file.stem.split("_-_")
    assert len(parts) == 6, f"Unexpected coherence filename format: {coherence_file}"
    return {
        "sub": parts[0].removeprefix("sub-"),
        "ses": parts[1].removeprefix("ses-"),
        "trial": parts[2].removeprefix("trial-"),
        "cond": parts[3].removeprefix("cond-"),
        "audio_name": parts[4].removeprefix("audio01-"),
        "coherence_file": coherence_file,
        "eeg_stem": coherence_file.stem.removesuffix("_coherence"),
    }


def load_coherence_index(coherence_dir: Path) -> pd.DataFrame:
    """One row per get_coherence.py output file, with its filename fields parsed out."""
    return pd.DataFrame([parse_coherence_filename(f) for f in sorted(coherence_dir.glob("*.npz"))])


def shifted_cross_spectrum(Xf_t: np.ndarray, Yf_t: np.ndarray, shift: int, scale: float, nperseg: int) -> np.ndarray:
    """Cross-spectral density (F, Cx, Cy) between x and a circularly shifted y, without ever
    materializing the shifted copy of y.

    ``np.roll(Yf, shift, axis=0)[t] == Yf[(t - shift) % n_frames]``, so the sum over frames splits
    into two contiguous runs -- x's first ``shift`` frames pairing with y's last ``shift``, and x's
    remainder with y's head -- each a matmul over plain array views. Equivalent to coherence.py's
    ``_cross_spectrum(Xf_t, np.roll(Yf, shift, axis=0), ...)`` up to floating-point summation
    order, but skips copying the whole (n_frames, F, Cy) spectrum on every draw. That copy, not
    the arithmetic, dominates at realistic sizes: measured ~2.8x faster end-to-end at
    F=2561, Cx=61, Cy=64, n_frames=1048, where it avoids a ~2.75 GB copy per draw.

    Takes y already transposed to (F, n_frames, Cy) so that transpose is hoisted out of the
    multi-draw loop too.
    """
    n_frames = Yf_t.shape[1]
    shift %= n_frames
    # At shift == 0 the first matmul is a zero-length contraction: numpy returns the correct
    # all-zeros (F, Cx, Cy), leaving the second term alone as the unshifted cross-spectrum.
    head = np.matmul(Xf_t[:, :, :shift], Yf_t[:, n_frames - shift :, :])
    tail = np.matmul(Xf_t[:, :, shift:], Yf_t[:, : n_frames - shift, :])
    return _onesided_fold((head + tail) / scale, nperseg)


def truncate_spectrum(spectrum: WindowedSpectrum, n_frames: int) -> WindowedSpectrum:
    """The first ``n_frames`` windows of an already-computed spectrum.

    get_coherence.py truncated x and y to a common sample length *before* windowing. Window k
    always covers samples [k*hop, k*hop+nperseg), so dropping trailing samples only drops trailing
    windows -- the surviving ones are bit-identical. That's what lets the stimulus's windowed FFT
    be computed once per stimulus and reused across trials whose EEG lengths differ slightly,
    instead of being recomputed per trial at each trial's own truncation point.

    ``scale`` (fs * window_energy * n_frames) is the only field that has to be rescaled.
    """
    if n_frames == spectrum.n_frames:
        return spectrum
    return WindowedSpectrum(
        freqs=spectrum.freqs,
        Xf=spectrum.Xf[:n_frames],
        hop=spectrum.hop,
        n_frames=n_frames,
        nperseg=spectrum.nperseg,
        scale=spectrum.scale / spectrum.n_frames * n_frames,
    )


def iter_null_coherence(
    Xf_t: np.ndarray,
    y_spectrum: WindowedSpectrum,
    denom: np.ndarray,
    fs: float,
    hop: int,
    scale: float,
    nperseg: int,
    min_shift_seconds: float,
    max_shift_seconds: float,
):
    """Yield N_NULL global-shift null complex coherencies, one (F, Cx, Cy) draw at a time.

    Mirrors coherence.py's ``null_coherence_shuffled_windows_from_prep`` in mode="global_shift"
    -- same bounds, same fixed-seed RNG call sequence, so draw i is the same shift it would be
    there -- but routes each draw through ``shifted_cross_spectrum`` and reuses get_coherence.py's
    cached Pxx/Pyy (folded into ``denom`` by the caller) instead of recomputing them.

    Yields draws instead of returning the stacked (N_NULL, F, Cx, Cy) array that function returns:
    the caller only ever folds each draw into a running sum, so materializing all of them at once
    would double the script's peak memory for nothing (~8 GB at N_NULL=50, on top of the
    same-size accumulator). This way one draw -- ~160 MB -- is live at a time.

    Re-seeded from NULL_RNG_SEED per trial rather than sharing one generator across trials, so a
    given trial's draw i never depends on how many trials preceded it.
    """
    rng = np.random.default_rng(NULL_RNG_SEED)
    n_frames = y_spectrum.n_frames

    min_shift_frames = int(np.ceil(min_shift_seconds * fs / hop))
    if min_shift_frames * 2 >= n_frames:
        raise ValueError(
            f"min_shift_seconds={min_shift_seconds} needs {min_shift_frames} frames of clearance "
            f"in each direction, but only {n_frames} frames are available."
        )
    max_shift_frames = min(n_frames - min_shift_frames, int(np.floor(max_shift_seconds * fs / hop)))
    if max_shift_frames < min_shift_frames:
        raise ValueError(f"max_shift_seconds={max_shift_seconds} is smaller than min_shift_seconds={min_shift_seconds}")

    Yf_t = y_spectrum.Xf.transpose(1, 0, 2)  # (F, n_frames, Cy) view; shift-independent, so hoisted
    for _ in range(N_NULL):
        shift = int(rng.integers(min_shift_frames, max_shift_frames + 1))
        yield shifted_cross_spectrum(Xf_t, Yf_t, shift, scale, nperseg) / denom


def pool_stimulus_trials(
    audio_name: str,
    group: pd.DataFrame,
    npz_path: Path,
    eeg_dir: Path,
    stimulus_psd_dir: Path,
    output_path: Path,
) -> None:
    """Pool every trial of one stimulus into a duration-weighted average real + null coherence.

    Everything that depends only on the stimulus -- its features' windowed FFT, its conjugate
    transpose, its cached Pxx -- is computed once here and reused across the trial loop.
    """
    stimulus_psd = np.load(stimulus_psd_dir / f"{audio_name}_stimulus_psd.npz")
    n_subbands = int(stimulus_psd["subband_psd"].shape[1])
    cached_Pxx = np.concatenate([stimulus_psd["subband_psd"], stimulus_psd["gammatone_psd"]], axis=1)

    features = load_stimulus_features(npz_path)
    fs = STIMULUS_FS
    nperseg, noverlap = stimulus_window_sizes(fs)
    min_shift_seconds, max_shift_seconds = null_shift_bounds(features.shape[1] / fs)

    x_spectrum_full = windowed_fft(features, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1)

    Cxy_sum: np.ndarray | None = None
    null_sum: np.ndarray | None = None
    total_duration = 0.0

    for _, row in tqdm(list(group.iterrows()), desc=audio_name, unit="trial", leave=False):
        trial = np.load(row["coherence_file"])
        Cxy = np.concatenate([trial["subband_coherence"], trial["gammatone_coherence"]], axis=1)
        Pyy = trial["eeg_psd"]
        duration = float(trial["duration"])

        eeg_data, eeg_fs = load_eeg(eeg_dir / f"{row['eeg_stem']}.fif")
        if eeg_fs != fs:
            raise ValueError(f"{row['eeg_stem']}: EEG sample rate {eeg_fs} does not match stimulus rate {fs}")
        y_spectrum = windowed_fft(eeg_data, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1)

        # get_coherence.py truncated both signals to a common length; reproduce that here by
        # dropping trailing windows from whichever side has more (see truncate_spectrum).
        n_frames = min(x_spectrum_full.n_frames, y_spectrum.n_frames)
        x_spectrum = truncate_spectrum(x_spectrum_full, n_frames)
        y_spectrum = truncate_spectrum(y_spectrum, n_frames)

        # The cached stimulus PSD was computed over the *full* feature length. It's the right Pxx
        # only when no stimulus window was dropped above; otherwise recompute it for this trial
        # (rare -- EEG is usually within a couple of samples of the stimulus, which the window
        # hop absorbs -- so this almost never costs anything).
        Pxx = cached_Pxx if n_frames == x_spectrum_full.n_frames else _auto_power(x_spectrum)
        denom = np.sqrt(Pxx[:, :, None] * Pyy[:, None, :])

        if Cxy_sum is None:
            Cxy_sum = np.zeros(Cxy.shape, dtype=complex)
            null_sum = np.zeros((N_NULL, *denom.shape), dtype=ACCUMULATOR_DTYPE)
        Cxy_sum += duration * Cxy
        for null_idx, null_draw in enumerate(
            iter_null_coherence(
                _conj_transpose_for_matmul(x_spectrum),
                y_spectrum,
                denom,
                fs,
                x_spectrum.hop,
                x_spectrum.scale,
                nperseg,
                min_shift_seconds,
                max_shift_seconds,
            )
        ):
            null_sum[null_idx] += duration * null_draw
        total_duration += duration

    if Cxy_sum is None:
        return

    Cxy_avg = Cxy_sum / total_duration
    null_avg = null_sum / total_duration
    np.savez(
        output_path,
        name=audio_name,
        n_trials=len(group),
        total_duration=total_duration,
        frequencies=x_spectrum_full.freqs,
        subband_coherence=Cxy_avg[:, :n_subbands, :].astype(COHERENCE_DTYPE),
        gammatone_coherence=Cxy_avg[:, n_subbands:, :].astype(COHERENCE_DTYPE),
        null_subband_coherence=null_avg[:, :, :n_subbands, :].astype(COHERENCE_DTYPE),
        null_gammatone_coherence=null_avg[:, :, n_subbands:, :].astype(COHERENCE_DTYPE),
    )


def main() -> int:
    config = load_config(ROOT / "src" / "package_data" / "config.json")
    audio_features_dir, eeg_dir, coherence_dir, stimulus_psd_dir, stimulus_coherence_dir = resolve_pooled_inputs(
        config, ROOT
    )

    assert coherence_dir.exists(), f"{coherence_dir} does not exist; run scripts/coherence/get_coherence.py first"
    stimulus_coherence_dir.mkdir(parents=True, exist_ok=True)

    df = load_coherence_index(coherence_dir)
    df = df[df["cond"] == POOLED_CONDITION]

    for audio_name, group in tqdm(df.groupby("audio_name"), desc="Stimuli", unit="stimulus"):
        output_path = stimulus_coherence_dir / f"{audio_name}_stimulus_coherence.npz"
        if output_path.exists():
            continue
        pool_stimulus_trials(
            audio_name, group, audio_features_dir / f"{audio_name}.npz", eeg_dir, stimulus_psd_dir, output_path
        )

    print(f"Wrote per-stimulus pooled coherence to {stimulus_coherence_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
