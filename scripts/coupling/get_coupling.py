"""Per-trial phase-phase and amplitude-amplitude coupling between the envelope and the EEG.

Both signals are cut into narrow bands, and the analytic signal of each band is taken. Then, per
band and per channel:

    phase-phase      PLV  = | mean_t  exp(i (phi_eeg(t) - phi_env(t - lag))) |
    amplitude-amp.   r    = corr_t( a_eeg(t), a_env(t - lag) )

along with the band's ordinary coherence at zero lag, as a check against the coherence pipeline.
The EEG is average-referenced (64 scalp channels; the mastoids are dropped).

Lags (default): -500 to +1000 ms, EEG relative to envelope.

Bands (default): 0.2-2 Hz, then 1 Hz wide to 30 Hz, then 5 Hz wide to 300 Hz -- 83 in all.

Why this is fast
----------------
Three things keep a 15-minute, 64-channel trial at seconds rather than minutes:

1. **One FFT per trial, not one filter per band.** The whole recording is transformed once.
   Every band is then a slice of that spectrum times a smooth (raised-cosine) window, so 83 band
   filters cost 83 slices, not 83 convolutions. Positive frequencies only, which makes the
   result the analytic signal directly, with no separate Hilbert transform.

2. **Complex demodulation onto a coarse grid.** A band's analytic signal is narrowband, so
   after shifting the band down to 0 Hz it only needs a sample rate a little above its own
   width. Each band is inverse-transformed at ``--grid-fs`` (128 Hz by default), not at
   1024 Hz: an inverse FFT 8x shorter. The shift multiplies both signals by the same
   ``exp(-i 2 pi f0 t)``, which cancels in ``phi_eeg - phi_env`` and leaves amplitudes
   untouched. So the PLV and amplitude correlation are exactly those of the full-rate signals,
   evaluated at fewer time points. Every band shares one grid, which is also what makes
   cross-frequency measures a matrix product later (see the note at the end).

3. **Every lag and every null draw from one cross-correlation.** The sum over t of
   ``u_eeg(t) conj(u_env(t - s))`` for *all* circular shifts s is one FFT product. So the lag
   profile and the null cost the same as a single lag: the lags are the small shifts, and the
   null draws are large circular shifts of the envelope against the EEG, using the coherence
   pipeline's own shift bounds (at least 60 s each way). The edges, where the circular FFT
   wraps, are masked out, and every lag is divided by its own count of valid overlapping
   samples.

Measured on this machine: ~2.5 s to read a trial over NFS, ~0.5 s for the full FFT, and a few
seconds for the 83 bands.

Outputs::

    <coupling_dir>/grid-128Hz_blocknull/<eeg_stem>.npz

each carrying, with B bands, L lags, D null draws and C = 64 channels:

    bands          (B, 2)        band edges, Hz
    lags_ms        (L,)          EEG relative to envelope; positive = EEG later
    null_shifts_s  (D,)          the circular shifts the null draws used
    plv            (B, L, C)     complex mean resultant; |.| is the PLV, angle the mean phase lag
    plv_null       (B, D, K, C)  the same for each null draw, over K block lags (below)
    aac            (B, L, C)     amplitude correlation
    aac_null       (B, D, K, C)
    null_block_lag_index (K,)    which of `lags_ms` the null blocks cover (every 8th, all of
                                 the lag range by default: K = 25)
    notched_fraction (B,)        share of each band's window power the mains notch removed
    cross          (B, C)        sum of z_eeg conj(z_env), zero lag -- with the next two this is
    power_env      (B,)            the band's coherence, as the coherence pipeline would give it
    power_eeg      (B, C)
    n_valid        (B,)          grid samples that entered each band (edges masked)

plus ``grid_fs``, ``channel_names``, ``analysed_duration``, ``stimulus``, ``subject``,
``eeg_stem`` and the settings. About 92 MB a trial, ~52 GB for the 563: 12 MB for the 193 real
lags and 80 MB for the null, whose lag blocks multiply it by K = 25.

The null is a block of lags per draw so that the peak over a lag window, taken on the pooled
data, has an exact null. Pool each draw's block exactly as the real lags are pooled, take the
same peak over the block, and the D results are the null distribution of that peak. Read the real
data on the same lags (``null_block_lag_index``) for an exact like-for-like comparison.

The PLV and the amplitude correlation are kept per trial and not pooled here. ``plv`` is complex
so it can be pooled the way the coherence is, by averaging the complex values across trials and
taking the magnitude last, or by magnitude first. The choice belongs in the analysis, as in
figure 5 of notebook 2.

Cross-frequency coupling (not computed here)
--------------------------------------------
Because every band lands on the same grid, envelope phase -> EEG amplitude coupling (PAC) is one
matrix product per trial: amplitudes ``A`` (T x B*C) against unit phase vectors ``P`` (T x B), with
``A.T @ P / sum(A)`` giving the mean vector length for every (amplitude band, channel, phase band)
at once. EEG phase -> envelope amplitude is the same with the roles swapped. The phase must be the
true phase, not the demodulated one, so the carrier is put back first with
``exp(i 2 pi f0 t)``. Only phase bands well below ``grid_fs / 2`` are usable. The null is what
costs: a circular shift per draw is another matrix product, with no FFT shortcut across all
shifts.

Examples
--------
    python scripts/coupling/get_coupling.py --max-trials 2 --workers 1
    python scripts/coupling/get_coupling.py --workers 4
    python scripts/coupling/get_coupling.py --log-amplitude --null-draws 100
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Defined before numpy is imported, so _cap_threads can read it.
DEFAULT_WORKERS = 4


def _cap_threads() -> None:
    """Split the cores between the workers before numpy and scipy.fft choose a thread count.

    The same trap ``get_transfer_functions.py`` documents: every process otherwise takes every
    core, and N workers oversubscribe N-fold. Read from argv because it has to happen before the
    imports. An explicit OMP_NUM_THREADS is left alone.
    """
    if any(os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                             "MKL_NUM_THREADS")):
        return
    workers = DEFAULT_WORKERS
    for flag in ("--workers", "-w"):
        if flag in sys.argv:
            index = sys.argv.index(flag) + 1
            if index < len(sys.argv):
                try:
                    workers = max(1, int(sys.argv[index]))
                except ValueError:
                    pass
    threads = max(1, (os.cpu_count() or 1) // max(1, workers))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = str(threads)


_cap_threads()

import mne
import numpy as np
import scipy.fft as sfft
from tqdm import tqdm

from coherence.pipeline import (
    GAMMATONE_FEATURE_KEY,
    NULL_RNG_SEED,
    STIMULUS_FS,
    build_trial_index,
    fsync_path,
    load_referenced_eeg,
    null_shift_bounds,
    resolve_configured_inputs,
)
from transcription.config import load_config

mne.set_log_level("WARNING")

# --- bands ---------------------------------------------------------------------------------
FIRST_BAND = (0.2, 2.0)
LOW_WIDTH, SPLIT_HZ = 1.0, 30.0          # 1 Hz bands from the first band's top edge to 30 Hz
HIGH_WIDTH, MAX_HZ = 5.0, 300.0          # then 5 Hz bands to 300 Hz

# Each band is flat inside and rolls off with a raised cosine centred on each edge, this fraction
# of the band's width wide in total. Centred on the edge, so neighbouring bands cross at half
# amplitude and the bank sums to one between them. The lower edge's roll-off is capped at half
# that edge's frequency, so the first band never reaches DC.
TAPER_FRACTION = 0.5

# Mains: bins within this of every harmonic up to Nyquist are zeroed, in both signals -- the
# notebooks' 2 Hz-wide notch. Measured 2026-10-06 on three recordings: where a line shows at all
# (up to ~20 dB at 150 Hz) it is never wider than +/-0.2 Hz at 0.05 Hz resolution, so +/-1 Hz
# covers it with room to spare. The 5 Hz bands have edges at every harmonic, so this takes a
# little off two bands each; how much is saved per band as `notched_fraction`.
MAINS_HZ, NOTCH_HALFWIDTH_HZ = 50.0, 1.0

# --- the grid, the edges, the lags, the null ------------------------------------------------
GRID_FS = 128.0
# Grid samples this close to either end of the recording are masked: the FFT is circular, so a
# band's response to the join between the end and the start spreads over a few of its cycles.
EDGE_SECONDS, EDGE_CYCLES = 1.0, 3.0    # the larger of 1 s and 3 / bandwidth
# Wide enough for the cortical response, and for the 1 Hz bands, whose lag resolution is ~1 s.
LAG_RANGE_MS = (-500.0, 1000.0)
NULL_DRAWS = 50
# The null is stored as a block of lags around each draw, not one shift, so that any statistic
# taken over lags of the real data -- a peak over a window, above all -- can be taken over the same
# lags of every null draw. Every NULL_BLOCK_EVERY-th grid lag inside NULL_BLOCK_MS (None: the whole
# lag range), a subset of the real lag axis, so the real data can be read on exactly the same lags.
# At 128 Hz every 8th lag is 62.5 ms apart -- a third of a 5 Hz band's ~200 ms lag resolution --
# and over -500..1000 ms that is 25 lags. Every 4th would double the null to ~165 MB a trial.
NULL_BLOCK_MS = None
NULL_BLOCK_EVERY = 8

COUPLING_DTYPE = np.complex64

_WORKER: dict[str, Any] = {}


def default_bands() -> np.ndarray:
    """0.2-2 Hz, 1 Hz bands to 30 Hz, 5 Hz bands to 300 Hz."""
    low = np.arange(FIRST_BAND[1], SPLIT_HZ, LOW_WIDTH)
    high = np.arange(SPLIT_HZ, MAX_HZ, HIGH_WIDTH)
    return np.array([FIRST_BAND]
                    + [(f, f + LOW_WIDTH) for f in low]
                    + [(f, f + HIGH_WIDTH) for f in high])


def band_window(frequencies: np.ndarray, low: float, high: float) -> np.ndarray:
    """Raised-cosine band window on `frequencies`: 1 inside, 0 outside, cosine across each edge."""
    half = TAPER_FRACTION * (high - low) / 2
    low_half = min(half, low / 2)
    window = np.zeros_like(frequencies)
    window[(frequencies >= low + low_half) & (frequencies <= high - half)] = 1.0
    for edge, width, rising in ((low, low_half, True), (high, half, False)):
        span = (frequencies > edge - width) & (frequencies < edge + width)
        phase = (frequencies[span] - (edge - width)) / (2 * width)        # 0 -> 1 across the edge
        ramp = 0.5 - 0.5 * np.cos(np.pi * phase)
        window[span] = ramp if rising else 1.0 - ramp
    return window


def mains_mask(frequencies: np.ndarray, fs: float) -> np.ndarray:
    keep = np.ones_like(frequencies)
    for harmonic in np.arange(MAINS_HZ, fs / 2, MAINS_HZ):
        keep[np.abs(frequencies - harmonic) <= NOTCH_HALFWIDTH_HZ] = 0.0
    return keep


# ---------------------------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------------------------


def demodulated_bands(spectrum: np.ndarray, frequencies: np.ndarray, low: float, high: float,
                      notch: np.ndarray, n_fft: int, m: int) -> tuple[np.ndarray, float]:
    """One band of `spectrum` (rows x rfft bins), as its baseband analytic signal on the grid.

    The band's bins are copied to the start of a length-`m` array and inverse-transformed. That
    evaluates the band's analytic signal, shifted down by the band's lowest bin frequency `f0`,
    at `m` evenly spaced times across the transform. Returns the signal and `f0`.

    The shift is common to the envelope and the EEG, so at zero lag it cancels in the phase
    difference. At a lag s it leaves a constant factor exp(-i 2 pi f0 s), which the caller removes.
    """
    window = band_window(frequencies, low, high) * notch
    support = np.flatnonzero(window > 0)
    first, last = support[0], support[-1] + 1
    if last - first > m:
        raise ValueError(f"{low:g}-{high:g} Hz spans {last - first} bins, more than the grid's {m}")
    block = np.zeros((spectrum.shape[0], m), COUPLING_DTYPE)
    block[:, :last - first] = spectrum[:, first:last] * window[first:last]
    # x2 for the analytic signal (positive frequencies only), and the scale that makes a length-m
    # inverse transform of n_fft-point coefficients come out in the signal's own units.
    return (sfft.ifft(block, axis=1, workers=-1) * np.float32(2.0 * m / n_fft),
            float(frequencies[first]))


def circular_lags(a: np.ndarray, b: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """sum_t a(t) conj(b(t - s)) for each s in `shifts`; `a` is (C, T), `b` is (T,)."""
    product = sfft.fft(a, axis=-1, workers=-1) * np.conj(sfft.fft(b, workers=-1))
    return sfft.ifft(product, axis=-1, workers=-1)[:, shifts % a.shape[-1]]


def compute_trial(eeg_path: Path, envelope: np.ndarray, bands: np.ndarray, grid_fs: float,
                  lag_range_ms: tuple[float, float], n_null: int, log_amplitude: bool,
                  null_block_ms: tuple[float, float] | None = NULL_BLOCK_MS,
                  null_block_every: int = NULL_BLOCK_EVERY) -> dict[str, Any]:
    if null_block_ms is None:
        null_block_ms = lag_range_ms
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
    notch = mains_mask(frequencies, fs)

    # The grid: m points spread evenly over the n_fft-sample transform.
    m = sfft.next_fast_len(int(np.ceil(n_fft * grid_fs / fs)))
    step = n_fft / m / fs                                     # seconds per grid sample
    grid_times = np.arange(m) * step
    analysed = n / fs

    lag_samples = np.arange(int(np.ceil(lag_range_ms[0] / 1000 / step)),
                            int(np.floor(lag_range_ms[1] / 1000 / step)) + 1)
    # The pipeline's null bounds and seeding, on this grid: re-seeded per trial, so draw i is
    # the same shift wherever this trial is pooled.
    min_shift, max_shift = null_shift_bounds(analysed)
    rng = np.random.default_rng(NULL_RNG_SEED)
    null_samples = rng.integers(int(np.ceil(min_shift / step)), int(np.floor(max_shift / step)) + 1,
                                size=n_null)
    # Each null draw is a block: the draw's shift plus every block lag, so the null at draw d and
    # block lag k is the real computation at lag k with the envelope moved a further shift away.
    block_lags_ms = lag_samples * step * 1000
    block_index = np.flatnonzero((block_lags_ms >= null_block_ms[0])
                                 & (block_lags_ms <= null_block_ms[1])
                                 & (lag_samples % null_block_every == 0))
    if block_index.size == 0:
        raise ValueError(f"no lag in {null_block_ms} ms is a multiple of {null_block_every} samples")
    block_offsets = lag_samples[block_index]
    null_block = (null_samples[:, None] + block_offsets[None, :]).ravel()
    shifts = np.concatenate([lag_samples, null_block])
    n_lags, n_block = lag_samples.size, block_offsets.size

    n_bands, n_channels = len(bands), len(channel_names)
    plv = np.empty((n_bands, shifts.size, n_channels), COUPLING_DTYPE)
    aac = np.empty((n_bands, shifts.size, n_channels), np.float32)
    cross = np.empty((n_bands, n_channels), COUPLING_DTYPE)
    power_env = np.empty(n_bands)
    power_eeg = np.empty((n_bands, n_channels))
    n_valid = np.empty(n_bands, int)
    notched_fraction = np.empty(n_bands)

    shift_seconds = shifts * step

    for index, (low, high) in enumerate(bands):
        z_eeg, f0 = demodulated_bands(eeg_spectrum, frequencies, low, high, notch, n_fft, m)
        z_env = demodulated_bands(env_spectrum, frequencies, low, high, notch, n_fft, m)[0][0]
        # How much of the band's window the mains notch removed, in power.
        window = band_window(frequencies, low, high)
        notched_fraction[index] = 1.0 - float(((window * notch) ** 2).sum() / (window ** 2).sum())

        # The valid samples are one contiguous block; everything outside it is zeroed by slicing,
        # which is far cheaper than multiplying whole arrays by a mask.
        edge = max(EDGE_SECONDS, EDGE_CYCLES / (high - low))
        start = int(np.ceil(edge / step))
        stop = int(np.floor((analysed - edge) / step)) + 1
        count = stop - start
        n_valid[index] = count
        # How many valid samples overlap at each circular shift, for a block of `count` in `m`.
        wrapped = shifts % m
        overlap = (np.maximum(count - wrapped, 0)
                   + np.maximum(count - (m - wrapped), 0)).astype(np.float32)
        overlap = np.maximum(overlap, 1.0)
        # Undoes the demodulation's exp(-i 2 pi f0 s) at each shift, so the stored angle is the
        # true phase lag of the band.
        carrier = np.exp(2j * np.pi * f0 * shift_seconds).astype(COUPLING_DTYPE)

        z_eeg_valid, z_env_valid = z_eeg[:, start:stop], z_env[start:stop]
        # Zero lag, amplitude-weighted: the band's coherence, for checking against the pipeline.
        cross[index] = z_eeg_valid @ np.conj(z_env_valid)
        power_env[index] = float(np.vdot(z_env_valid, z_env_valid).real)
        power_eeg[index] = (z_eeg_valid.real ** 2 + z_eeg_valid.imag ** 2).sum(-1)

        # Phase: unit vectors. A sample of exactly zero amplitude has no phase, and is left at 0.
        tiny = np.finfo(np.float32).tiny
        a_eeg = np.abs(z_eeg_valid)
        a_env = np.abs(z_env_valid)
        u_eeg = np.zeros_like(z_eeg)
        u_env = np.zeros_like(z_env)
        u_eeg[:, start:stop] = z_eeg_valid / np.maximum(a_eeg, tiny)
        u_env[start:stop] = z_env_valid / np.maximum(a_env, tiny)
        plv[index] = (circular_lags(u_eeg, u_env, shifts) * carrier / overlap).T
        del u_eeg, u_env, z_eeg

        # Amplitude: standardised over the valid samples, so the shifted products divided by the
        # overlap count are a correlation (exact at zero lag; at other shifts the moments are the
        # whole block's rather than the overlap's, which differ only at the block's ends).
        if log_amplitude:
            a_eeg = np.log(np.maximum(a_eeg, tiny))
            a_env = np.log(np.maximum(a_env, tiny))
        a_eeg = a_eeg - a_eeg.mean(-1, keepdims=True)
        a_eeg /= np.sqrt((a_eeg ** 2).mean(-1, keepdims=True))
        a_env = a_env - a_env.mean()
        a_env /= np.sqrt((a_env ** 2).mean())
        s_eeg = np.zeros((n_channels, m), np.float32)
        s_env = np.zeros(m, np.float32)
        s_eeg[:, start:stop] = a_eeg
        s_env[start:stop] = a_env
        aac[index] = np.real(circular_lags(s_eeg, s_env, shifts) / overlap).T

    null_shape = (n_bands, n_null, n_block, n_channels)
    return {
        "plv": plv[:, :n_lags], "plv_null": plv[:, n_lags:].reshape(null_shape),
        "aac": aac[:, :n_lags], "aac_null": aac[:, n_lags:].reshape(null_shape),
        "cross": cross, "power_env": power_env, "power_eeg": power_eeg, "n_valid": n_valid,
        "notched_fraction": notched_fraction,
        "lags_ms": block_lags_ms, "null_shifts_s": null_samples * step,
        "null_block_lag_index": block_index, "null_block_lags_ms": block_lags_ms[block_index],
        "grid_fs": 1.0 / step, "channel_names": channel_names, "analysed_duration": analysed,
    }


def save_trial(trial: dict[str, Any], task: dict[str, Any], output_path: Path,
               context: dict[str, Any]) -> None:
    """Write via a temporary file renamed into place, so a crash never leaves a partial file."""
    payload = {
        **{key: value for key, value in trial.items() if key != "channel_names"},
        "channel_names": np.array(trial["channel_names"]),
        "bands": context["bands"],
        "taper_fraction": TAPER_FRACTION,
        "mains_hz": MAINS_HZ,
        "notch_halfwidth_hz": NOTCH_HALFWIDTH_HZ,
        "log_amplitude": context["log_amplitude"],
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


def _init_worker(context: dict[str, Any]) -> None:
    _WORKER.update(context)
    _WORKER["envelopes"] = {}


def _envelope(audio_name: str) -> np.ndarray:
    """The broadband envelope, kept per worker: trials arrive in stimulus order."""
    cache = _WORKER["envelopes"]
    if audio_name not in cache:
        cache.clear()
        with np.load(_WORKER["audio_features_dir"] / f"{audio_name}.npz") as saved:
            cache[audio_name] = np.asarray(saved[GAMMATONE_FEATURE_KEY], float).reshape(-1)
    return cache[audio_name]


def _process_trial(task: dict[str, Any]) -> tuple[str, str, str | None]:
    try:
        trial = compute_trial(_WORKER["eeg_dir"] / f"{task['eeg_stem']}.fif",
                              _envelope(task["audio_name"]), _WORKER["bands"],
                              _WORKER["grid_fs"], _WORKER["lag_range_ms"],
                              _WORKER["null_draws"], _WORKER["log_amplitude"],
                              _WORKER["null_block_ms"], _WORKER["null_block_every"])
        save_trial(trial, task, _WORKER["output_dir"] / f"{task['eeg_stem']}.npz", _WORKER)
        return task["eeg_stem"], "written", None
    except Exception as exc:  # one bad recording must not kill the run
        return task["eeg_stem"], "failed", f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------


def resolve_coupling_dir(config: dict[str, Any], grid_fs: float, log_amplitude: bool) -> Path:
    """``references.avg.coupling_dir`` if configured, else a ``coupling`` folder beside the TRFs."""
    entry = config.get("references", {}).get("avg", {})
    base = entry.get("coupling_dir") or config.get("coupling_dir")
    if base is None:
        base = Path(entry.get("trf_dir") or config["trf_dir"]).parent / "coupling"
    # "_blocknull": the null is stored in lag blocks, a different layout from the first run's
    # single-shift null, so the two never share a folder.
    name = f"grid-{grid_fs:g}Hz_blocknull" + ("_logamp" if log_amplitude else "")
    return Path(base) / name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid-fs", type=float, default=GRID_FS,
                        help=f"Rate every band is evaluated at (default {GRID_FS:g} Hz); must "
                             "exceed the widest band's support")
    parser.add_argument("--lag-range", type=float, nargs=2, default=LAG_RANGE_MS,
                        metavar=("LOW_MS", "HIGH_MS"),
                        help="Lags kept, EEG relative to envelope (default %(default)s)")
    parser.add_argument("--null-draws", type=int, default=NULL_DRAWS,
                        help=f"Circular-shift null draws per trial (default {NULL_DRAWS})")
    parser.add_argument("--null-block", type=float, nargs=2, default=NULL_BLOCK_MS,
                        metavar=("LOW_MS", "HIGH_MS"),
                        help="Lag window stored for every null draw (default: the --lag-range)")
    parser.add_argument("--null-block-every", type=int, default=NULL_BLOCK_EVERY,
                        help="Keep every Nth grid lag of that window (default %(default)s)")
    parser.add_argument("--log-amplitude", action="store_true",
                        help="Correlate log amplitudes rather than amplitudes")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Worker processes (default {DEFAULT_WORKERS}); ~2 GB each")
    parser.add_argument("--max-trials", type=int, default=None,
                        help="Smoke-test on the first N trials only")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute trials whose output already exists")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    audio_features_dir, eeg_dir, _, _ = resolve_configured_inputs(config)
    output_dir = resolve_coupling_dir(config, args.grid_fs, args.log_amplitude)
    output_dir.mkdir(parents=True, exist_ok=True)

    bands = default_bands()
    widest = float(np.max(bands[:, 1] - bands[:, 0])) * (1 + TAPER_FRACTION)
    if widest >= args.grid_fs:
        raise SystemExit(f"--grid-fs {args.grid_fs:g} is below the widest band's support "
                         f"({widest:g} Hz)")

    trials = build_trial_index(eeg_dir)
    trials = trials.sort_values(["audio_name", "sub"]).reset_index(drop=True)
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

    print(f"{len(bands)} bands, {bands[0, 0]:g}-{bands[-1, 1]:g} Hz; grid {args.grid_fs:g} Hz; "
          f"lags {args.lag_range[0]:g} to {args.lag_range[1]:g} ms; {args.null_draws} null draws, "
          f"each over {(args.null_block or args.lag_range)[0]:g}-"
          f"{(args.null_block or args.lag_range)[1]:g} ms every "
          f"{args.null_block_every} grid lags")
    print(f"{len(trials)} trials over {trials['sub'].nunique()} participants -> {output_dir}")

    context = {
        "eeg_dir": eeg_dir, "audio_features_dir": audio_features_dir, "output_dir": output_dir,
        "bands": bands, "grid_fs": args.grid_fs, "lag_range_ms": tuple(args.lag_range),
        "null_draws": args.null_draws, "log_amplitude": args.log_amplitude,
        "null_block_ms": tuple(args.null_block or args.lag_range),
        "null_block_every": args.null_block_every,
    }
    tasks = trials.to_dict("records")
    counts = {"written": 0, "failed": 0}
    failures: list[tuple[str, str]] = []

    def collect(result: tuple[str, str, str | None]) -> None:
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
