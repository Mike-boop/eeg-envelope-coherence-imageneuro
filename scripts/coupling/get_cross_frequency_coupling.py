"""Per-trial cross-frequency coupling between the envelope and the EEG, in both directions.

    PAC   envelope phase  (band p) -> EEG amplitude      (band a, channel c)
    APC   EEG phase (band p, channel c) -> envelope amplitude (band a)

each as the complex mean-vector sum

    M(a, p, lag, c) = sum_t  A(t) * exp(i phi(t'))

where the stimulus-side quantity is taken at t - lag and the EEG-side one at t, so a positive lag
always means the EEG follows the stimulus. The EEG is average-referenced, with the 64 scalp
channels.

Bands come in tiers
-------------------
An amplitude band can only change in amplitude as fast as its own width allows. So coupling to a
phase at f_p is only visible in amplitude bands wide enough to pass both sidebands f_a +/- f_p
(Aru et al. 2015). The first version of this script used get_coupling.py's 1 Hz and 5 Hz bands on
both sides, which made almost every cell zero by construction. Now each phase band is paired with
amplitude bands of the narrowest tier that can carry it:

    phase 0.2-2 Hz  (1 band)    x  amplitude bands  6 Hz wide
    phase 2-8 Hz    (6 bands)   x  amplitude bands 24 Hz wide
    phase 8-30 Hz   (22 bands)  x  amplitude bands 80 Hz wide

Amplitude centres step every 5 Hz up to 300 Hz, starting where the band's lower edge clears the
tier's highest phase frequency. The phase bands are get_coupling.py's (1 Hz above 2 Hz).

The sums are kept raw, along with each amplitude series' sum and sum of squares and the sample
count per lag. That way any of the usual normalisations can be formed afterwards: the raw MVL
(Canolty), MVL / mean amplitude, or the direct PAC estimate |M| / sqrt(n * sum A**2) (Ozkurt). M
is complex, so it can be pooled across trials complex first or magnitude first.

Why it is fast
--------------
The band decomposition is `get_coupling.py`'s: one FFT per trial, then each band demodulated onto a
coarse grid. Here the grid is 64 Hz, not 128. Phase bands stop at 30 Hz, and a complex phasor
needs a sample rate above its bandwidth only, not twice it. Phases get their carrier back,
``exp(i 2 pi f0 t)``, because cross-frequency coupling needs the true phase, not the demodulated
one. The 80 Hz amplitude bands are ~120 Hz wide with their edges, too wide for that grid, so they
are demodulated on a 2x finer one and their amplitude is brought down to 64 Hz by a zero-phase
anti-aliasing FIR. That keeps everything below 32 Hz, which is all a 30 Hz phase can couple to.

After that, every (amplitude band x channel, phase band) pair at one lag is a single matrix
product: amplitudes (T x A_k*64) against phasors (T x P_k), as two real float32 GEMMs, for the
cosine and the sine. There is no FFT shortcut across shifts here, because the pairs number in the
tens of thousands. So each lag and each null draw costs one GEMM per direction and tier: 5 lags x
(1 + 10 null draws) x 2 directions x 3 tiers.

Lags and null
-------------
Lags: 0-250 ms every 62.5 ms by default (`--lags-ms`). The null is the coupling pipeline's
convention: circular shifts of the stimulus-side series against the EEG-side one, at least 60 s,
fixed seed per trial. Each null draw is stored over the same lags as the real data, so a
peak over lags has an exact null, as in `get_coupling.py`. The circular shift wraps inside the
valid block (the edges masked for every band), and the lag is then applied by slicing, without
wrapping.

Outputs::

    <coupling_dir>/cfc_tiers_grid-64Hz/<eeg_stem>.npz

per tier k (0, 1, 2), with A_k amplitude bands, P_k phase bands, L lags, D draws, C = 64:

    pac_t{k}            (A_k, P_k, L, C)       complex64  envelope phase -> EEG amplitude
    pac_null_t{k}       (A_k, P_k, D, L, C)
    apc_t{k}            (A_k, P_k, L, C)       complex64  EEG phase -> envelope amplitude
    apc_null_t{k}       (A_k, P_k, D, L, C)
    eeg_amp_sum_t{k}, eeg_amp_sumsq_t{k}   (L, A_k, C)
    env_amp_sum_t{k}, env_amp_sumsq_t{k}   (L, A_k)
    env_phasor_sum_t{k}     (S, L, P_k)    envelope phasor summed over each lag's samples, per
                                           shift (S = 1 + D: the real data first, then each draw)
    eeg_phasor_sum_t{k}     (L, P_k, C)    the EEG's, which is never shifted
    env_amp_sum_shifted_t{k} (S, L, A_k)   envelope amplitude summed per lag and shift
    amplitude_bands_t{k} (A_k, 2), phase_bands_t{k} (P_k, 2)

The three phasor and shifted sums are for debiasing (van Driel et al. 2015). M - (sum A)(sum u)/n
removes the mean amplitude times the phasor's non-zero mean, which otherwise dominates the slowest
phase bands.

and ``n`` (L,), ``lags_ms``, ``null_shifts_s``, ``n_tiers``, ``tier_limits``, ``grid_fs``,
``channel_names``, ``analysed_duration``, ``stimulus``, ``subject``, ``eeg_stem``.

The null draws are seeded per trial (from the file name). The first version used one fixed seed,
which gave every listener of a stimulus identical shifts, so complex-first null draws were one draw
per stimulus rather than per trial.

Examples
--------
    python scripts/coupling/get_cross_frequency_coupling.py --max-trials 2 --workers 1
    python scripts/coupling/get_cross_frequency_coupling.py --workers 4
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Importing get_coupling caps the BLAS/FFT threads from --workers before numpy loads, and brings the
# band decomposition with it: the same bands, windows, notch and demodulation.
import get_coupling as coupling  # noqa: E402  (must precede numpy)

import numpy as np  # noqa: E402
import scipy.fft as sfft  # noqa: E402
import zlib  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402
from tqdm import tqdm  # noqa: E402

from coherence.pipeline import (  # noqa: E402
    NULL_RNG_SEED,
    STIMULUS_FS,
    build_trial_index,
    fsync_path,
    load_referenced_eeg,
    null_shift_bounds,
    resolve_configured_inputs,
)
from transcription.config import load_config  # noqa: E402

GRID_FS = 64.0
LAGS_MS = (0.0, 62.5, 125.0, 187.5, 250.0)
NULL_DRAWS = 10

# Tiers: (highest phase frequency, amplitude band width), both Hz. An amplitude band can only
# change in amplitude as fast as its own width allows, so to show coupling to phase f_p it must pass
# both sidebands f_a +/- f_p (Aru et al. 2015). With the raised-cosine edges the flat part of a band
# W wide is W * (1 - TAPER_FRACTION / 2), so W >= 2 f_p / 0.75 = 2.67 f_p. Each phase band is paired
# only with the amplitude bands of the narrowest tier that can carry it: 0.2-2 Hz with 6 Hz bands,
# 2-8 Hz with 24 Hz bands, 8-30 Hz with 80 Hz bands.
TIERS = ((2.0, 6.0), (8.0, 24.0), (30.0, 80.0))
# Amplitude centres every AMP_STEP_HZ, from where the band's lower edge clears its tier's highest
# phase frequency (so the amplitude band never contains the phase band) to AMP_MAX_CENTRE_HZ.
AMP_STEP_HZ, AMP_MAX_CENTRE_HZ = 5.0, 300.0
DEFAULT_WORKERS = coupling.DEFAULT_WORKERS

_WORKER: dict[str, Any] = {}


def tier_bands() -> list[dict[str, np.ndarray]]:
    """Per tier, its phase bands (get_coupling.py's) and its amplitude bands."""
    phase_all = coupling.default_bands()
    tiers, floor = [], 0.0
    for phase_max, width in TIERS:
        phase = phase_all[(phase_all[:, 1] > floor) & (phase_all[:, 1] <= phase_max)]
        centres = np.arange(phase_max + width / 2, AMP_MAX_CENTRE_HZ + 1e-9, AMP_STEP_HZ)
        tiers.append({"phase_bands": phase,
                      "amplitude_bands": np.stack([centres - width / 2, centres + width / 2], 1)})
        floor = phase_max
    return tiers


def decompose(spectrum, frequencies, bands, notch, n_fft, m, step, start, stop, phase):
    """Every band of `spectrum` on the grid's valid block: amplitudes, or true-phase phasors.

    Returns (T, B, rows) with T = stop - start: float32 amplitudes, or complex64 unit phasors
    with the carrier restored, so their angle is the band's real instantaneous phase.

    A band wider than the grid carries (the 80 Hz amplitude bands, ~120 Hz with their edges) is
    demodulated on a grid `factor` times finer, and its amplitude is then brought down to the
    common grid by `resample_poly`'s zero-phase anti-aliasing FIR. That keeps the amplitude's
    content below grid_fs / 2, which is all a phase of at most 30 Hz can couple to.
    """
    out = np.empty((stop - start, len(bands), spectrum.shape[0]),
                   np.complex64 if phase else np.float32)
    times = (np.arange(start, stop) * step).astype(np.float64)
    grid_fs = 1.0 / step
    for index, (low, high) in enumerate(bands):
        support = (high - low) * (1 + coupling.TAPER_FRACTION)
        factor = max(1, int(np.ceil(support / grid_fs)))
        if phase and factor > 1:
            raise ValueError(f"phase band {low:g}-{high:g} Hz is wider than the {grid_fs:g} Hz grid")
        z, f0 = coupling.demodulated_bands(spectrum, frequencies, low, high, notch, n_fft,
                                           m * factor)
        if phase:
            z = z[:, start:stop]
            magnitude = np.abs(z)
            carrier = np.exp(2j * np.pi * f0 * times).astype(np.complex64)
            out[:, index] = (z / np.maximum(magnitude, np.finfo(np.float32).tiny) * carrier).T
        else:
            magnitude = np.abs(z)
            if factor > 1:
                magnitude = resample_poly(magnitude, 1, factor, axis=1)
            out[:, index] = magnitude[:, start:stop].T
    return out


def pac_sums(eeg_amplitude, env_phasor, lag_samples, shifts):
    """(len(shifts), L, Na, Np): EEG amplitude at t against envelope phasor at t - lag - shift."""
    T = eeg_amplitude.shape[0]
    out = np.empty((len(shifts), len(lag_samples), eeg_amplitude.shape[1], env_phasor.shape[1]),
                   np.complex64)
    for d, shift in enumerate(shifts):
        rolled = np.roll(env_phasor, shift, axis=0) if shift else env_phasor
        cos, sin = np.ascontiguousarray(rolled.real), np.ascontiguousarray(rolled.imag)
        for k, lag in enumerate(lag_samples):
            a = eeg_amplitude[lag:]                    # EEG at t
            out[d, k].real = a.T @ cos[:T - lag]       # stimulus at t - lag
            out[d, k].imag = a.T @ sin[:T - lag]
    return out


def apc_sums(eeg_phasor, env_amplitude, lag_samples, shifts):
    """(len(shifts), L, Np_eeg, Na): EEG phasor at t against envelope amplitude at t - lag - shift."""
    T = eeg_phasor.shape[0]
    cos, sin = np.ascontiguousarray(eeg_phasor.real), np.ascontiguousarray(eeg_phasor.imag)
    out = np.empty((len(shifts), len(lag_samples), eeg_phasor.shape[1], env_amplitude.shape[1]),
                   np.complex64)
    for d, shift in enumerate(shifts):
        rolled = np.roll(env_amplitude, shift, axis=0) if shift else env_amplitude
        for k, lag in enumerate(lag_samples):
            a = rolled[:T - lag]                       # stimulus at t - lag
            out[d, k].real = cos[lag:].T @ a           # EEG at t
            out[d, k].imag = sin[lag:].T @ a
    return out


def compute_trial(eeg_path: Path, envelope: np.ndarray, tiers: list[dict[str, np.ndarray]],
                  grid_fs: float, lags_ms: np.ndarray, n_null: int) -> dict[str, Any]:
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
    # One valid block for every band of every tier, set by the narrowest band's edge: pairs mix
    # bands, so they must share their time axis.
    all_bands = np.concatenate([np.concatenate([t["amplitude_bands"], t["phase_bands"]])
                                for t in tiers])
    narrowest = float(np.min(all_bands[:, 1] - all_bands[:, 0]))
    edge = max(coupling.EDGE_SECONDS, coupling.EDGE_CYCLES / narrowest)
    start, stop = int(np.ceil(edge / step)), int(np.floor((analysed - edge) / step)) + 1
    T = stop - start

    lag_samples = np.round(np.asarray(lags_ms) / 1000 / step).astype(int)
    if (lag_samples < 0).any():
        raise ValueError("lags must be >= 0: the stimulus side leads")
    # Seeded per TRIAL, from its file name. A fixed seed gave every trial of one stimulus the same
    # shifts (same length, same bounds, same generator), so each null draw added coherently across
    # all listeners of that stimulus when pooled complex first -- one draw per stimulus, not per
    # trial. A per-trial seed makes the draws independent across trials and still reproducible.
    min_shift, max_shift = null_shift_bounds(T * step)
    rng = np.random.default_rng([NULL_RNG_SEED, zlib.crc32(eeg_path.stem.encode())])
    null_samples = rng.integers(int(np.ceil(min_shift / step)), int(np.floor(max_shift / step)) + 1,
                                size=n_null)
    shifts = np.concatenate([[0], null_samples])

    def bands_of(spectrum, bands, phase):
        return decompose(spectrum, frequencies, bands, notch, n_fft, m, step, start, stop, phase)

    n_channels = len(channel_names)
    result: dict[str, Any] = {}
    for k, tier in enumerate(tiers):
        amplitude_bands, phase_bands = tier["amplitude_bands"], tier["phase_bands"]
        n_amp, n_phase = len(amplitude_bands), len(phase_bands)

        # PAC: EEG amplitude (T, A*C) against envelope phase (T, P).
        eeg_amplitude = bands_of(eeg_spectrum, amplitude_bands, False).reshape(T, n_amp * n_channels)
        env_phasor = bands_of(env_spectrum, phase_bands, True).reshape(T, n_phase)
        pac = pac_sums(eeg_amplitude, env_phasor, lag_samples, shifts)        # (S, L, A*C, P)
        # For debiasing (van Driel et al. 2015): the stimulus phasor's own sum over exactly the
        # samples each lag and shift used. M - (sum A) * (sum u) / n removes the mean amplitude
        # times the phasor's non-zero mean, which dominates the slowest phase bands.
        result[f"env_phasor_sum_t{k}"] = np.stack([np.stack(
            [(np.roll(env_phasor, shift, axis=0) if shift else env_phasor)[:T - lag].sum(0)
             for lag in lag_samples]) for shift in shifts]).astype(np.complex64)   # (S, L, P)
        pac = pac.reshape(len(shifts), len(lag_samples), n_amp, n_channels, n_phase)
        pac = pac.transpose(2, 4, 0, 1, 3)                                     # (A, P, S, L, C)
        result[f"eeg_amp_sum_t{k}"] = np.stack(
            [eeg_amplitude[lag:].sum(0) for lag in lag_samples]
        ).reshape(len(lag_samples), n_amp, n_channels)
        result[f"eeg_amp_sumsq_t{k}"] = np.stack(
            [(eeg_amplitude[lag:].astype(np.float64) ** 2).sum(0) for lag in lag_samples]
        ).reshape(len(lag_samples), n_amp, n_channels)
        del eeg_amplitude, env_phasor

        # APC: EEG phase (T, P*C) against envelope amplitude (T, A).
        eeg_phasor = bands_of(eeg_spectrum, phase_bands, True).reshape(T, n_phase * n_channels)
        env_amplitude = bands_of(env_spectrum, amplitude_bands, False).reshape(T, n_amp)
        apc = apc_sums(eeg_phasor, env_amplitude, lag_samples, shifts)        # (S, L, P*C, A)
        # The same debiasing terms for APC: the EEG phasor is never shifted, the envelope
        # amplitude is, so its sum is kept per shift.
        result[f"eeg_phasor_sum_t{k}"] = np.stack(
            [eeg_phasor[lag:].sum(0) for lag in lag_samples]
        ).reshape(len(lag_samples), n_phase, n_channels).astype(np.complex64)      # (L, P, C)
        result[f"env_amp_sum_shifted_t{k}"] = np.stack([np.stack(
            [(np.roll(env_amplitude, shift, axis=0) if shift else env_amplitude)[:T - lag].sum(0)
             for lag in lag_samples]) for shift in shifts])                        # (S, L, A)
        apc = apc.reshape(len(shifts), len(lag_samples), n_phase, n_channels, n_amp)
        apc = apc.transpose(4, 2, 0, 1, 3)                                     # (A, P, S, L, C)
        result[f"env_amp_sum_t{k}"] = np.stack([env_amplitude[:T - lag].sum(0)
                                                for lag in lag_samples])
        result[f"env_amp_sumsq_t{k}"] = np.stack(
            [(env_amplitude[:T - lag].astype(np.float64) ** 2).sum(0) for lag in lag_samples])
        del eeg_phasor, env_amplitude

        result[f"pac_t{k}"] = np.ascontiguousarray(pac[:, :, 0])
        result[f"pac_null_t{k}"] = np.ascontiguousarray(pac[:, :, 1:])
        result[f"apc_t{k}"] = np.ascontiguousarray(apc[:, :, 0])
        result[f"apc_null_t{k}"] = np.ascontiguousarray(apc[:, :, 1:])

    result.update({
        "n": T - lag_samples,
        "lags_ms": lag_samples * step * 1000, "null_shifts_s": null_samples * step,
        "grid_fs": 1.0 / step, "channel_names": channel_names, "analysed_duration": analysed,
        "edge_seconds": edge,
    })
    return result


def save_trial(trial, task, output_path, context):
    payload = {
        **{key: value for key, value in trial.items() if key != "channel_names"},
        "channel_names": np.array(trial["channel_names"]),
        "n_tiers": len(context["tiers"]),
        "tier_limits": np.array(TIERS),
        **{f"amplitude_bands_t{k}": t["amplitude_bands"] for k, t in enumerate(context["tiers"])},
        **{f"phase_bands_t{k}": t["phase_bands"] for k, t in enumerate(context["tiers"])},
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
                              _envelope(task["audio_name"]), _WORKER["tiers"],
                              _WORKER["grid_fs"], _WORKER["lags_ms"], _WORKER["null_draws"])
        save_trial(trial, task, _WORKER["output_dir"] / f"{task['eeg_stem']}.npz", _WORKER)
        return task["eeg_stem"], "written", None
    except Exception as exc:  # one bad recording must not kill the run
        return task["eeg_stem"], "failed", f"{type(exc).__name__}: {exc}"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid-fs", type=float, default=GRID_FS,
                        help=f"Grid rate (default {GRID_FS:g} Hz); must exceed --max-phase-hz")
    parser.add_argument("--lags-ms", type=float, nargs="+", default=list(LAGS_MS),
                        help="Lags, stimulus leading, all >= 0 (default %(default)s)")
    parser.add_argument("--null-draws", type=int, default=NULL_DRAWS,
                        help=f"Circular-shift null draws per trial (default {NULL_DRAWS})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Worker processes (default {DEFAULT_WORKERS}); ~3 GB each")
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
    # "_tiers": phase-dependent amplitude bandwidths and per-trial null seeds. The first run's
    # cfc_grid-64Hz used one band set for both sides and a shared seed; never mix the two.
    output_dir = base / f"cfc_tiers_grid-{args.grid_fs:g}Hz"
    output_dir.mkdir(parents=True, exist_ok=True)

    tiers = tier_bands()
    highest_phase = max(float(t["phase_bands"][:, 1].max()) for t in tiers)
    phase_support = highest_phase * (1 + coupling.TAPER_FRACTION / 2)
    if phase_support >= args.grid_fs:
        raise SystemExit(f"--grid-fs {args.grid_fs:g} cannot carry phases up to "
                         f"{phase_support:g} Hz")

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

    for k, t in enumerate(tiers):
        print(f"tier {k}: phase {t['phase_bands'][0, 0]:g}-{t['phase_bands'][-1, 1]:g} Hz "
              f"({len(t['phase_bands'])} bands) x {len(t['amplitude_bands'])} amplitude bands "
              f"{t['amplitude_bands'][0, 1] - t['amplitude_bands'][0, 0]:g} Hz wide, "
              f"centres {t['amplitude_bands'][0].mean():g}-{t['amplitude_bands'][-1].mean():g} Hz")
    print(f"grid {args.grid_fs:g} Hz; lags {args.lags_ms} ms; {args.null_draws} null draws over "
          "the same lags, seeded per trial")
    print(f"{len(trials)} trials over {trials['sub'].nunique()} participants -> {output_dir}")

    context = {
        "eeg_dir": eeg_dir, "audio_features_dir": audio_features_dir, "output_dir": output_dir,
        "tiers": tiers,
        "grid_fs": args.grid_fs, "lags_ms": np.asarray(args.lags_ms, float),
        "null_draws": args.null_draws,
    }
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
