"""Per-trial coherence between the stimulus envelope and the AMPLITUDE of EEG carrier bands.

For each carrier band b (delta ... high gamma) and channel c, the EEG band's Hilbert amplitude
A_bc(t) is treated as a signal in its own right, and its Welch cross-spectrum with the broadband
envelope x(t) is formed over modulation frequency f_m:

    S_xA(f_m, b, c) = E[ conj(X(f_m)) * A_bc(f_m) ]

The three parts of the coherence are saved undivided, as get_transfer_functions.py saves them:
the cross-spectrum, the envelope's PSD and each amplitude series' PSD. That leaves the pooling
(complex first or magnitude first, by trial or by participant) to the analysis. The coherence is
|S_xA|**2 / (S_xx S_AA). Its phase is the amplitude's lag behind the envelope: a positive slope of
phase against f_m, -d(phase)/d(2 pi f_m), is a positive group delay, with the EEG later.

This differs from the main coherence pipeline in the same way that power tracking differs from
phase locking. Coherence of the raw EEG at 100 Hz asks whether the EEG phase-locks to the
envelope's 100 Hz content (the frequency-following response). Coherence of the 80-200 Hz
AMPLITUDE at 4 Hz asks whether high-frequency power rises and falls with the syllables.

A carrier band can only change in amplitude as fast as its own width allows: one W Hz wide (W * 1.5
with its raised-cosine edges) carries modulation up to about W. So above its own width a carrier's
coherence is zero by construction, not by measurement. Each band's usable limit is saved as
``max_modulation_hz``. The analysis should mask above it.

Implementation
--------------
The same decomposition as the coupling scripts: one FFT of the whole trial, each carrier band
demodulated onto a 64 Hz grid. Bands too wide for that grid (80-200 Hz is 180 Hz wide with its
edges) are demodulated on a finer one and their amplitude is brought down to 64 Hz by a zero-phase
anti-aliasing FIR. The envelope is evaluated on the same grid by keeping its spectrum below the grid's
Nyquist frequency and inverse-transforming. Then 5 s Hann windows at 80% overlap, as everywhere
else in this repo, give modulation frequencies 0.2-32 Hz at 0.2 Hz.

The null is ``--null-draws`` circular shifts of the envelope's grid series against the EEG amplitudes
(at least 60 s each way), seeded **per trial** from the file name so that listeners of the same
stimulus get independent draws. A draw costs one 1-channel windowed FFT and one contraction, so 50
are cheap.

Outputs::

    <coupling_dir>/amplitude_coherence_grid-64Hz/<eeg_stem>.npz        --band-set classic (default)
    <coupling_dir>/amplitude_coherence_fine_grid-64Hz/<eeg_stem>.npz   --band-set fine
    <coupling_dir>/amplitude_coherence_high30_grid-128Hz/<eeg_stem>.npz --band-set high30 --grid-fs 128

with F modulation frequencies, B carrier bands, D null draws, C = 64 channels:

    cross_spectrum       (F, B, C)     complex64  S_xA
    cross_spectrum_null  (D, F, B, C)  complex64
    envelope_psd         (F,)          S_xx (the null's is the same: a circular shift keeps it)
    amplitude_psd        (F, B, C)     S_AA
    frequencies (F,), carrier_bands (B, 2), max_modulation_hz (B,), null_shifts_s (D,)

plus ``channel_names``, ``analysed_duration``, ``grid_fs``, ``n_frames``, ``stimulus``,
``subject``, ``eeg_stem``. About 26 MB a trial for the 6 classic bands. The null dominates and
scales with bands x draws, so the 35 fine bands are run with 20 draws: about 60 MB a trial,
~33 GB for the 563.

Band sets: ``classic`` is 1-4, 4-8, 8-13, 13-30, 30-80, 80-200 Hz. ``fine`` is 0.2-2, 2-4, 4-6, 6-8,
8-10, 10-12, 12-20, 20-30 Hz, then 10 Hz bands from 30 to 300 Hz.

Examples
--------
    python scripts/coupling/get_amplitude_coherence.py --max-trials 2 --workers 1
    python scripts/coupling/get_amplitude_coherence.py --workers 4
    python scripts/coupling/get_amplitude_coherence.py --band-set fine --null-draws 20 --workers 4
    python scripts/coupling/get_amplitude_coherence.py --band-set high30 --grid-fs 128 --workers 4
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import zlib
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Importing get_coupling caps the BLAS/FFT threads from --workers before numpy loads, and brings the
# band decomposition with it: the same windows, notch and demodulation.
import get_coupling as coupling  # noqa: E402  (must precede numpy)

import numpy as np  # noqa: E402
import scipy.fft as sfft  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402
from tqdm import tqdm  # noqa: E402

from coherence.coherence import windowed_fft  # noqa: E402
from coherence.pipeline import (  # noqa: E402
    NULL_RNG_SEED,
    OVERLAP_FRACTION,
    STIMULUS_FS,
    WINDOW_SECONDS,
    build_trial_index,
    fsync_path,
    load_referenced_eeg,
    null_shift_bounds,
    resolve_configured_inputs,
)
from transcription.config import load_config  # noqa: E402

GRID_FS = 64.0
NULL_DRAWS = 50
DEFAULT_WORKERS = coupling.DEFAULT_WORKERS
# Carrier bands, Hz: the classical ranges, with gamma split where the coherence spectrum has its
# two high-frequency peaks (~45 and ~95 Hz) on either side.
CARRIER_BANDS = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 80.0), (80.0, 200.0))
# A finer set: 2 Hz steps through the low bands, beta split at 20 Hz, then 10 Hz bands to 300 Hz.
# A 10 Hz band carries modulation up to ~10 Hz, the 2 Hz ones only to ~2 Hz.
FINE_BANDS = ((0.2, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 10.0), (10.0, 12.0),
              (12.0, 20.0), (20.0, 30.0)) + tuple((f, f + 10.0) for f in range(30, 300, 10))
# Wide carriers for fast modulation: 30 Hz bands from 100 to 250 Hz, each carrying modulation up
# to ~30 Hz. Run on a 128 Hz grid (--grid-fs 128) so the envelope reaches past that.
HIGH_BANDS = tuple((f, f + 30.0) for f in range(100, 250, 30))
# Each set writes to its own folder; "classic" keeps the first run's name.
BAND_SETS = {"classic": (CARRIER_BANDS, ""), "fine": (FINE_BANDS, "_fine"),
             "high30": (HIGH_BANDS, "_high30")}
# Edge seconds masked at each end of the trial, as in the coupling scripts: the FFT is circular.
EDGE_SECONDS = 3.0

_WORKER: dict[str, Any] = {}


def amplitude_on_grid(spectrum, frequencies, bands, notch, n_fft, m, grid_fs):
    """(m, B, rows) float32: each band's Hilbert amplitude on the grid, anti-aliased to grid_fs.

    A band `support` Hz wide (edges included) has an amplitude with content up to `support` Hz,
    the widest difference between two of its frequencies. On a working grid of rate g, content at
    f > g/2 folds to g - f. That lands harmlessly above the output's Nyquist (where the
    anti-aliasing FIR removes it) only if g - support >= grid_fs / 2, so the working grid is the
    smallest multiple of grid_fs with g >= support + grid_fs / 2. (Until 2026-10-07 it was only
    required to exceed `support`, which let the classic 80-200 Hz band fold its fastest amplitude
    fluctuations, which are weak at the band's edges, down to ~12 Hz of modulation.)
    """
    out = np.empty((m, len(bands), spectrum.shape[0]), np.float32)
    for index, (low, high) in enumerate(bands):
        support = (high - low) * (1 + coupling.TAPER_FRACTION)
        factor = max(1, int(np.ceil((support + grid_fs / 2) / grid_fs)))
        z, _ = coupling.demodulated_bands(spectrum, frequencies, low, high, notch, n_fft, m * factor)
        magnitude = np.abs(z)
        if factor > 1:
            magnitude = resample_poly(magnitude, 1, factor, axis=1)
        out[:, index] = magnitude.T
    return out


def envelope_on_grid(env_spectrum, frequencies, n_fft, m, grid_fs):
    """(m,) float64: the envelope low-passed below the grid's Nyquist and evaluated on the grid.

    The rfft coefficients below grid_fs / 2 (with a raised-cosine roll-off over the top 10%) go into
    a length-m inverse rfft. That evaluates the same trigonometric polynomial at the grid times, so
    the envelope and the amplitudes sit on exactly the same time axis.
    """
    nyquist = grid_fs / 2
    keep = frequencies < nyquist
    taper = np.ones(int(keep.sum()))
    ramp = frequencies[keep] > 0.9 * nyquist
    taper[ramp] = 0.5 + 0.5 * np.cos(np.pi * (frequencies[keep][ramp] - 0.9 * nyquist)
                                     / (0.1 * nyquist))
    coefficients = np.zeros(m // 2 + 1, complex)
    coefficients[:keep.sum()] = env_spectrum[0, keep] * taper
    return sfft.irfft(coefficients, n=m) * (m / n_fft)


def compute_trial(eeg_path: Path, envelope: np.ndarray, bands: np.ndarray, grid_fs: float,
                  n_null: int) -> dict[str, Any]:
    loaded = load_referenced_eeg(eeg_path, "avg")
    assert loaded is not None, "the average reference applies to every recording"
    eeg, fs, channel_names = loaded
    if fs != STIMULUS_FS:
        raise ValueError(f"{eeg_path.name}: EEG rate {fs} != stimulus rate {STIMULUS_FS}")

    n = min(eeg.shape[1], envelope.size)
    n_fft = sfft.next_fast_len(n, real=True)
    eeg_spectrum = sfft.rfft(eeg[:, :n].astype(np.float32), n_fft, axis=1, workers=-1)
    env_spectrum = sfft.rfft(envelope[None, :n].astype(np.float32), n_fft, axis=1, workers=-1)
    del eeg
    frequencies = sfft.rfftfreq(n_fft, 1.0 / fs)
    notch = coupling.mains_mask(frequencies, fs)

    m = sfft.next_fast_len(int(np.ceil(n_fft * grid_fs / fs)))
    step = n_fft / m / fs
    analysed = n / fs
    start = int(np.ceil(EDGE_SECONDS / step))
    stop = int(np.floor((analysed - EDGE_SECONDS) / step)) + 1

    amplitude = amplitude_on_grid(eeg_spectrum, frequencies, bands, notch, n_fft, m, 1.0 / step)
    amplitude = amplitude[start:stop].reshape(stop - start, -1)            # (T, B*C)
    env = envelope_on_grid(env_spectrum, frequencies, n_fft, m, 1.0 / step)[start:stop]
    del eeg_spectrum

    grid_rate = 1.0 / step
    nperseg = int(round(WINDOW_SECONDS * grid_rate))
    noverlap = int(round(OVERLAP_FRACTION * nperseg))
    y = windowed_fft(amplitude, fs=grid_rate, nperseg=nperseg, noverlap=noverlap, axis=0)
    x = windowed_fft(env, fs=grid_rate, nperseg=nperseg, noverlap=noverlap, axis=0)
    Y = y.Xf.astype(np.complex64)                                           # (frames, F, B*C)

    def cross(x_spectrum):
        return np.einsum("nf,nfc->fc", np.conj(x_spectrum.Xf[:, :, 0]).astype(np.complex64), Y,
                         optimize=True) / np.float32(x_spectrum.scale)

    n_bands, n_channels = len(bands), len(channel_names)
    shape = (len(y.freqs), n_bands, n_channels)
    cross_spectrum = cross(x).reshape(shape)
    envelope_psd = (np.abs(x.Xf[:, :, 0]) ** 2).sum(0) / x.scale
    amplitude_psd = ((np.abs(Y) ** 2).sum(0) / y.scale).reshape(shape)

    # Per-trial seed: listeners of the same stimulus must not share draws (see the CFC script).
    T = stop - start
    min_shift, max_shift = null_shift_bounds(T * step)
    rng = np.random.default_rng([NULL_RNG_SEED, zlib.crc32(eeg_path.stem.encode())])
    null_samples = rng.integers(int(np.ceil(min_shift / step)), int(np.floor(max_shift / step)) + 1,
                                size=n_null)
    null = np.empty((n_null, *shape), np.complex64)
    for d, shift in enumerate(null_samples):
        shifted = windowed_fft(np.roll(env, shift), fs=grid_rate, nperseg=nperseg,
                               noverlap=noverlap, axis=0)
        null[d] = cross(shifted).reshape(shape)

    # A band W wide has an amplitude that changes no faster than about W: the widest difference
    # between two frequencies inside its flat part, plus the edges' share. Taken as W itself.
    max_modulation = bands[:, 1] - bands[:, 0]
    return {
        "cross_spectrum": cross_spectrum, "cross_spectrum_null": null,
        "envelope_psd": envelope_psd, "amplitude_psd": amplitude_psd,
        "frequencies": y.freqs, "max_modulation_hz": max_modulation,
        "null_shifts_s": null_samples * step, "grid_fs": grid_rate, "n_frames": y.n_frames,
        "channel_names": channel_names, "analysed_duration": analysed,
    }


def save_trial(trial, task, output_path, context):
    payload = {
        **{key: value for key, value in trial.items() if key != "channel_names"},
        "channel_names": np.array(trial["channel_names"]),
        "carrier_bands": context["bands"],
        "stimulus": task["audio_name"],
        "subject": task["sub"],
        "eeg_stem": task["eeg_stem"],
        "reference": "avg",
    }
    temporary = output_path.with_suffix(".npz.tmp")
    with open(temporary, "wb") as handle:
        np.savez(handle, **payload)
        handle.flush()
    fsync_path(temporary)
    temporary.rename(output_path)


def _init_worker(context):
    _WORKER.update(context)
    _WORKER["envelopes"] = {}


def _envelope(audio_name):
    cache = _WORKER["envelopes"]
    if audio_name not in cache:
        cache.clear()
        with np.load(_WORKER["audio_features_dir"] / f"{audio_name}.npz") as saved:
            cache[audio_name] = np.asarray(saved[coupling.GAMMATONE_FEATURE_KEY], float).reshape(-1)
    return cache[audio_name]


def _process_trial(task):
    try:
        trial = compute_trial(_WORKER["eeg_dir"] / f"{task['eeg_stem']}.fif",
                              _envelope(task["audio_name"]), _WORKER["bands"], _WORKER["grid_fs"],
                              _WORKER["null_draws"])
        save_trial(trial, task, _WORKER["output_dir"] / f"{task['eeg_stem']}.npz", _WORKER)
        return task["eeg_stem"], "written", None
    except Exception as exc:  # one bad recording must not kill the run
        return task["eeg_stem"], "failed", f"{type(exc).__name__}: {exc}"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid-fs", type=float, default=GRID_FS,
                        help=f"Grid rate (default {GRID_FS:g} Hz); modulation runs to half of it")
    parser.add_argument("--band-set", choices=sorted(BAND_SETS), default="classic",
                        help="Carrier bands: 'classic' (6 bands), 'fine' (35 bands, 0.2-300 Hz) "
                             "or 'high30' (30 Hz bands 100-250 Hz; use --grid-fs 128)")
    parser.add_argument("--null-draws", type=int, default=NULL_DRAWS,
                        help=f"Circular-shift null draws per trial (default {NULL_DRAWS})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Worker processes (default {DEFAULT_WORKERS})")
    parser.add_argument("--max-trials", type=int, default=None,
                        help="Smoke-test on the first N trials only")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute trials whose output already exists")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    audio_features_dir, eeg_dir, _, _ = resolve_configured_inputs(config)
    base = coupling.resolve_coupling_dir(config, args.grid_fs, False).parent
    band_list, suffix = BAND_SETS[args.band_set]
    output_dir = base / f"amplitude_coherence{suffix}_grid-{args.grid_fs:g}Hz"
    output_dir.mkdir(parents=True, exist_ok=True)
    bands = np.array(band_list)

    trials = build_trial_index(eeg_dir).sort_values(["audio_name", "sub"]).reset_index(drop=True)
    if args.max_trials is not None:
        trials = trials.head(args.max_trials)
    if not args.overwrite:
        done = np.array([(output_dir / f"{stem}.npz").exists() for stem in trials["eeg_stem"]])
        if done.any():
            print(f"{int(done.sum())} trials already written; skipping (use --overwrite)")
            trials = trials[~done]
    if trials.empty:
        print("nothing to do")
        return 0

    print(f"carrier bands {', '.join(f'{lo:g}-{hi:g}' for lo, hi in bands)} Hz; grid "
          f"{args.grid_fs:g} Hz; {WINDOW_SECONDS:g} s windows; {args.null_draws} null draws, "
          "seeded per trial")
    print(f"{len(trials)} trials over {trials['sub'].nunique()} participants -> {output_dir}")

    context = {"eeg_dir": eeg_dir, "audio_features_dir": audio_features_dir,
               "output_dir": output_dir, "bands": bands, "grid_fs": args.grid_fs,
               "null_draws": args.null_draws}
    tasks = trials.to_dict("records")
    counts, failures = {"written": 0, "failed": 0}, []

    def collect(result):
        stem, status, error = result
        counts[status] += 1
        if error is not None:
            failures.append((stem, error))
            tqdm.write(f"FAILED {stem}: {error}")

    if args.workers <= 1:
        _init_worker(context)
        for task in tqdm(tasks, desc="trials", unit="trial"):
            collect(_process_trial(task))
    else:
        with mp.get_context("fork").Pool(processes=args.workers, initializer=_init_worker,
                                         initargs=(context,)) as pool:
            for result in tqdm(pool.imap(_process_trial, tasks), total=len(tasks),
                               desc="trials", unit="trial"):
                collect(result)

    print(f"\nwrote {counts['written']} trials"
          + (f", {counts['failed']} FAILED" if counts["failed"] else ""))
    for stem, error in failures[:5]:
        print(f"  {stem}: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
