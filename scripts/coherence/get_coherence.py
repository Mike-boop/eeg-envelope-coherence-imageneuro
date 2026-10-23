#!/usr/bin/env python3
"""Cross-spectral coherence between audio envelope features and EEG, per recording.

Ported from ``neural_tracking_across_spectrum``'s per-dataset ``get_coherences.py`` scripts:
same Welch window sizing (5 s windows, 80% overlap) and average-referenced EEG. Unlike those
scripts, which batched every subject's coherence into one file per stimulus, this writes one
coherence file per EEG recording (keyed by its own filename) plus one stimulus-PSD file per
audio stimulus, and computes against this repo's ``audio_features`` factory outputs instead of
re-deriving envelopes inline.

This is step 1 of the three-script pipeline in ``.claude/coherence_pipeline_plan.md``: it owns the
*real* per-trial coherence plus a single null draw, and the pooled null distributions are built
downstream by ``get_null_cochleogram_coherence.py`` and ``get_null_gammatone_coherence.py``.
Everything the three must agree on -- windowing, dtypes, the null-shift convention, directory
resolution, file naming -- lives in ``src/coherence/pipeline.py`` rather than here, so that the
later scripts import a library instead of importing this one.

Each recording's ``duration`` is derived from the EEG itself (not the stimulus, which can differ
by ~30 s) so that downstream weighted averages are correct; see ``compute_and_save_coherence``.

Performance notes (rewritten 2026-08-13)
----------------------------------------
The original structure opened a 3-worker pool *per stimulus* and called ``prepare_coherence`` per
trial. Five things were wrong with that, and all are fixed here. None change the maths -- only how
often it is done and on how many cores.

1. **The stimulus FFT was recomputed for every trial.** ``prepare_coherence`` windows *both*
   signals, so each of the 617 recordings paid a full 61-channel FFT of its stimulus. Since 617
   recordings share only ~72 stimuli, that is ~8.5x more stimulus FFTs than necessary.
   ``StimulusCache`` computes one per stimulus per worker and truncates its trailing windows per
   recording, which is exact rather than approximate -- see ``coherence.truncate_spectrum``.
   This is the largest saving here.

2. **BLAS was oversubscribed.** Nothing pinned thread counts, so each of 3 workers could spawn 24
   BLAS threads on a 24-core box. The ``Cx = 61`` matmul is compute-bound and genuinely wants
   threads (unlike ``get_null_gammatone_coherence.py``'s ``Cx = 1`` kernel, which is
   bandwidth-bound and must be pinned to one), so the fix is to divide the cores among workers
   rather than to disable threading -- see ``BLAS_THREADS_PER_WORKER``.

3. **The pool was per stimulus, so it idled.** Recordings per stimulus run 1 / 6 / 72
   (min/median/max), so a per-stimulus pool spends much of the run with fewer tasks than workers.
   One global pool over all (stimulus, recording) pairs, dispatched in stimulus order so the cache
   still hits, keeps every worker fed.

4. **The 468 MB feature array was pickled per task.** It was passed as a ``submit`` argument, so it
   was serialised and shipped once per recording -- ~289 GB of pickling across the run. Workers now
   load their own stimulus through the cache and nothing large crosses a process boundary.

5. **The null draw copied the whole EEG spectrum.** ``null_coherence_shuffled_windows_from_prep``
   materialises ``np.roll(Yf, shift, axis=0)`` -- 2.75 GB per draw. ``shifted_cross_spectrum``
   splits the frame sum into two contiguous runs instead, and the C-contiguous ``Yf_t`` it needs is
   materialised once and reused by *both* the real estimate and the null draw, so the copy that
   used to be pure cost now also speeds up the real matmul.

Because 1 and 5 bypass ``prepare_coherence`` / ``null_coherence_shuffled_windows_from_prep``, the
summation order differs from the previous implementation and results differ at float level. The
null draw's RNG call sequence is reproduced exactly by ``pipeline.null_shifts``, so draw *i* is the
same shift the shared implementation would give -- the property the downstream cross-checks rely
on, and the one ``get_null_gammatone_coherence.py`` validated against a saved run.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import multiprocessing as mp

import mne
import numpy as np
from scipy.signal import welch
from tqdm import tqdm

from coherence.coherence import (
    CoherencePrep,
    WindowedSpectrum,
    _auto_power,
    _conj_transpose_for_matmul,
    coherence_from_prep,
    contiguous_conj_transpose,
    contiguous_for_matmul,
    shifted_cross_spectrum,
    truncate_spectrum,
    windowed_fft,
)

# Re-exported rather than redefined: these are pipeline-wide decisions, and the downstream scripts
# now take them from the same place instead of from this module.
from coherence.pipeline import (
    COHERENCE_DTYPE,
    EEG_GLOB_TEMPLATE,
    GAMMATONE_FEATURE_KEY,
    NULL_MIN_SHIFT_SECONDS,
    NULL_RNG_SEED,
    OVERLAP_FRACTION,
    STIMULUS_FS,
    SUBBAND_FEATURE_KEY,
    WINDOW_SECONDS,
    find_matching_eeg_files,
    load_eeg,
    load_stimulus_features,
    null_shift_bounds,
    null_shifts,
    resolve_configured_inputs,
    stimulus_window_sizes,
)
from transcription.config import load_config

mne.set_log_level("WARNING")

# ---------------------------------------------------------------------------------------------
# Recompute existing outputs instead of skipping them.
#
# Every output this script writes is keyed by its input's filename and is skipped when it already
# exists, which makes an interrupted run resumable -- and makes a *deliberate* re-run a silent
# no-op. Set this True when a parameter that changes the contents has changed (N_NULL, the
# windowing, the null bounds) and the files on disk are stale.
#
# It rewrites in place, so the old outputs are gone as each new one lands. Move the directory
# aside first if you want to keep them for comparison; at N_NULL=1 the new set is 99.5 GB against
# the old 297 GB, so both fit on a 492 GB volume.
OVERWRITE = True
# ---------------------------------------------------------------------------------------------

# One draw, not five. The null draw dominates this file's size (a draw costs as much as the real
# coherence itself), and the pooled null distributions this feeds are built by the dedicated
# get_null_*_coherence.py scripts, which take their own ~500 draws. What is kept here is a
# per-recording sanity reference and a cross-check against those scripts, which one draw serves.
# At 5 it was 481 MB/recording and 297 GB across the dataset; at 1 it is 161 MB and 99.5 GB.
N_NULL = 1

# Recordings are processed by a pool of workers over all (stimulus, recording) pairs at once.
# Memory is the binding constraint, not cores -- see the measured breakdown in
# scripts/coherence/bench_get_coherence.py, which is what this number should be set from.
N_WORKERS = 6

# Cores per worker for the batched matmul. The Cx=61 cross-spectrum is compute-bound (~61 FLOP per
# byte of Yf at complex64), so BLAS threads are worth having -- but N_WORKERS of them each taking
# the whole box is the oversubscription trap recorded in get_null_gammatone_coherence.py. Splitting
# the cores keeps the product at one machine's worth.
BLAS_THREADS_PER_WORKER = max(1, (os.cpu_count() or N_WORKERS) // N_WORKERS)

# Print a per-phase timing breakdown (EEG load, FFTs, matmuls, save) for the first recordings a
# worker handles. Cheap, but noisy, so it is off by default.
TIMING = False
TIMING_FIRST_N = 3

_WORKER: dict[str, Any] = {}


def compute_stimulus_psd(
    features: np.ndarray, fs: float, nperseg: int, noverlap: int
) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD (n_freqs, n_features) of each feature channel, same windowing as the coherence call.

    Kept as the reference implementation, and used by the tests to pin the equivalence. The
    pipeline itself takes the PSD from the already-computed windowed spectrum via ``_auto_power``
    (see ``StimulusCache``), which is the same quantity -- identical window, detrend and density
    scaling -- without a second windowing pass over the 61-channel stimulus.
    """
    freqs, psd = welch(
        features, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hann", detrend="constant", axis=-1
    )
    return freqs, psd.T


@dataclass
class CachedStimulus:
    """One stimulus's windowed spectrum and everything derived from it that recordings share."""

    audio_name: str
    spectrum: WindowedSpectrum
    Xf_t: np.ndarray  # (F, Cx, n_frames) conjugated, C-contiguous, for the batched matmul
    Pxx: np.ndarray  # (F, Cx) auto-power over the full length
    n_samples: int
    n_subbands: int


class StimulusCache:
    """The most recently used stimulus's spectrum, held per worker.

    This is the fix for the largest inefficiency in the original: ``prepare_coherence`` windowed
    the 61-channel stimulus once per *recording*, and 617 recordings share ~72 stimuli. Tasks are
    dispatched in stimulus order so that consecutive tasks in a worker hit this cache; only one
    stimulus is held, because two would be ~5.2 GB.
    """

    def __init__(self, audio_features_dir: Path, fs: float, nperseg: int, noverlap: int):
        self.audio_features_dir = audio_features_dir
        self.fs, self.nperseg, self.noverlap = fs, nperseg, noverlap
        self.cached: CachedStimulus | None = None

    def get(self, audio_name: str) -> CachedStimulus:
        if self.cached is not None and self.cached.audio_name == audio_name:
            return self.cached

        self.cached = None  # drop the previous ~5.2 GB before allocating the next
        features = load_stimulus_features(self.audio_features_dir / f"{audio_name}.npz")
        spectrum = windowed_fft(features, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap, axis=1)
        self.cached = CachedStimulus(
            audio_name=audio_name,
            spectrum=spectrum,
            # Materialised, not the strided view _conj_transpose_for_matmul returns: a per-trial
            # copy would not pay for only two matmuls, but this one is per *stimulus* and is reused
            # by every recording of it, so ~72 copies serve all 617.
            Xf_t=contiguous_conj_transpose(spectrum.Xf),
            # Same quantity scipy.welch would give for this window/detrend/scaling, taken from the
            # spectrum that already exists rather than from a second windowing pass.
            Pxx=_auto_power(spectrum),
            n_samples=features.shape[1],
            n_subbands=features.shape[0] - 1,
        )
        return self.cached


def save_stimulus_psd(stimulus: CachedStimulus, stimulus_psd_dir: Path, fs: float) -> None:
    """Write one stimulus's PSD file, if it is missing or OVERWRITE is set.

    Written to a temporary name and renamed: with a global pool two workers can reach the same
    stimulus concurrently, and a half-written file must never be visible to the downstream scripts
    that read these.
    """
    output_path = stimulus_psd_dir / f"{stimulus.audio_name}_stimulus_psd.npz"
    if output_path.exists() and not OVERWRITE:
        return

    n_subbands = stimulus.n_subbands
    tmp_path = output_path.with_name(f"{output_path.name}.{os.getpid()}.partial")
    with tmp_path.open("wb") as handle:
        np.savez(
            handle,
            audio_name=stimulus.audio_name,
            # The stimulus's own duration -- correct here, since this file describes the stimulus.
            # Per-recording durations live in the coherence files and come from the EEG.
            duration=stimulus.n_samples / fs,
            frequencies=stimulus.spectrum.freqs,
            subband_psd=stimulus.Pxx[:, :n_subbands],
            gammatone_psd=stimulus.Pxx[:, n_subbands:],
        )
    tmp_path.replace(output_path)


def compute_and_save_coherence(
    eeg_path: Path,
    stimulus: CachedStimulus,
    fs: float,
    nperseg: int,
    noverlap: int,
    coherence_dir: Path,
    timing: dict[str, float] | None = None,
) -> Path:
    """Load one EEG recording and save its real + null coherence against a cached stimulus."""
    output_path = coherence_dir / f"{eeg_path.stem}_coherence.npz"
    if output_path.exists() and not OVERWRITE:
        return output_path

    def mark(phase: str, since: float) -> float:
        if timing is not None:
            timing[phase] = timing.get(phase, 0.0) + time.perf_counter() - since
        return time.perf_counter()

    tick = time.perf_counter()
    eeg_data, eeg_fs = load_eeg(eeg_path)
    if eeg_fs != fs:
        raise ValueError(f"{eeg_path.name}: EEG sample rate {eeg_fs} does not match stimulus rate {fs}")
    tick = mark("eeg_load", tick)

    y_full = windowed_fft(eeg_data, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1)
    tick = mark("eeg_fft", tick)

    # Reproduce the common-length truncation by dropping trailing windows from whichever side has
    # more; the surviving windows are bit-identical either way.
    n_frames = min(stimulus.spectrum.n_frames, y_full.n_frames)
    x_spectrum = truncate_spectrum(stimulus.spectrum, n_frames)
    y_spectrum = truncate_spectrum(y_full, n_frames)

    # The cached Pxx covers the full-length stimulus, so it is the right one only when no stimulus
    # window was dropped; otherwise recompute for this recording (rare -- the EEG is usually within
    # a couple of samples of the stimulus, which the window hop absorbs).
    if n_frames == stimulus.spectrum.n_frames:
        Xf_t, Pxx = stimulus.Xf_t, stimulus.Pxx
    else:
        Xf_t = contiguous_conj_transpose(x_spectrum.Xf)
        Pxx = _auto_power(x_spectrum)
    Pyy = _auto_power(y_spectrum)
    denom = np.sqrt(Pxx[:, :, None] * Pyy[:, None, :])
    tick = mark("prep", tick)

    prep = CoherencePrep(
        fs=fs, x_spectrum=x_spectrum, y_spectrum=y_spectrum, Pxx=Pxx, Pyy=Pyy, Xf_t=Xf_t, denom=denom
    )
    result = coherence_from_prep(prep)
    tick = mark("real_matmul", tick)

    # Three durations, not one, because the weighted averages downstream could reasonably use any
    # of them and recomputing this file to change your mind is expensive:
    #   duration          -- the EEG recording's own length. What the analysis plan asks for.
    #   analysed_duration -- the overlap actually entering the estimate. x and y are truncated to a
    #                        common length, and the EEG and stimulus differ by up to ~30 s in this
    #                        dataset, so this is NOT the same number.
    #   n_frames          -- Welch windows averaged. Coherence variance goes as 1/n_frames, so
    #                        this is the statistically natural weight.
    duration = eeg_data.shape[1] / eeg_fs
    analysed_duration = min(eeg_data.shape[1], stimulus.n_samples) / fs

    # One contiguous copy of y, used by the null draw below. It replaces the 2.75 GB np.roll copy
    # the old path made per draw, and additionally gives the matmul a well-behaved operand.
    Yf_t = contiguous_for_matmul(y_spectrum.Xf)
    shifts = null_shifts(n_frames, fs, y_spectrum.hop, N_NULL, analysed_duration)
    null_Cxy = np.empty((N_NULL, *denom.shape), dtype=np.complex128)
    for null_idx, shift in enumerate(shifts):
        null_Cxy[null_idx] = shifted_cross_spectrum(Xf_t, Yf_t, int(shift), x_spectrum.scale, nperseg) / denom
    del Yf_t
    tick = mark("null_matmul", tick)

    n_subbands = stimulus.n_subbands
    tmp_path = output_path.with_name(f"{output_path.name}.{os.getpid()}.partial")
    with tmp_path.open("wb") as handle:
        np.savez(
            handle,
            audio_name=stimulus.audio_name,
            duration=duration,
            analysed_duration=analysed_duration,
            n_frames=n_frames,
            frequencies=result.freqs,
            subband_coherence=result.Cxy[:, :n_subbands, :].astype(COHERENCE_DTYPE),
            gammatone_coherence=result.Cxy[:, n_subbands:, :].astype(COHERENCE_DTYPE),
            eeg_psd=result.Pyy,
            null_subband_coherence=null_Cxy[:, :, :n_subbands, :].astype(COHERENCE_DTYPE),
            null_gammatone_coherence=null_Cxy[:, :, n_subbands:, :].astype(COHERENCE_DTYPE),
        )
    tmp_path.replace(output_path)
    mark("save", tick)
    return output_path


# ---------------------------------------------------------------------------------------------
# Worker pool
# ---------------------------------------------------------------------------------------------


def _init_worker(context: dict[str, Any]) -> None:
    """Pin this worker's BLAS to its share of the cores and give it a stimulus cache.

    The thread limit must outlive the call, so the controller is kept in ``_WORKER``: threadpoolctl
    restores the previous limits when it is garbage collected.
    """
    try:
        from threadpoolctl import threadpool_limits

        _WORKER["thread_limit"] = threadpool_limits(limits=context["blas_threads"])
    except ImportError:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, str(context["blas_threads"]))
    _WORKER.update(context)
    _WORKER["stimulus_cache"] = StimulusCache(
        context["audio_features_dir"], context["fs"], context["nperseg"], context["noverlap"]
    )
    _WORKER["n_done"] = 0


def _process_recording(task: dict[str, Any]) -> tuple[str, str | None]:
    try:
        stimulus = _WORKER["stimulus_cache"].get(task["audio_name"])
        save_stimulus_psd(stimulus, _WORKER["stimulus_psd_dir"], _WORKER["fs"])

        collect = TIMING and _WORKER["n_done"] < TIMING_FIRST_N
        timing: dict[str, float] | None = {} if collect else None
        compute_and_save_coherence(
            task["eeg_path"],
            stimulus,
            _WORKER["fs"],
            _WORKER["nperseg"],
            _WORKER["noverlap"],
            _WORKER["coherence_dir"],
            timing,
        )
        _WORKER["n_done"] += 1
        if timing:
            total = sum(timing.values())
            breakdown = "  ".join(f"{k}={v:.1f}s" for k, v in timing.items())
            tqdm.write(f"[timing] {task['eeg_path'].stem}: total={total:.1f}s  {breakdown}")
        return task["eeg_path"].stem, None
    except Exception as exc:  # one bad recording must not kill a multi-hour run
        return task["eeg_path"].stem, f"{type(exc).__name__}: {exc}"


def build_tasks(audio_features_dir: Path, eeg_dir: Path, coherence_dir: Path) -> list[dict[str, Any]]:
    """Every (stimulus, recording) pair still needing computation, in stimulus order.

    Stimulus order is what makes the per-worker ``StimulusCache`` hit: with a global pool and
    ordered dispatch, consecutive tasks handed to a worker usually share a stimulus.
    """
    npz_files = sorted(audio_features_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz feature files found in {audio_features_dir}")

    tasks: list[dict[str, Any]] = []
    for npz_path in npz_files:
        audio_name = npz_path.stem
        for eeg_path in find_matching_eeg_files(eeg_dir, audio_name):
            if not OVERWRITE and (coherence_dir / f"{eeg_path.stem}_coherence.npz").exists():
                continue
            tasks.append({"audio_name": audio_name, "eeg_path": eeg_path})
    return tasks


def run_tasks(tasks: list[dict[str, Any]], context: dict[str, Any], n_workers: int) -> list[tuple[str, str]]:
    """Process every task, returning (stem, error) for those that failed."""
    failures: list[tuple[str, str]] = []
    if n_workers <= 1:
        _init_worker(context)
        for task in tqdm(tasks, desc="recordings", unit="recording"):
            stem, error = _process_recording(task)
            if error:
                failures.append((stem, error))
                tqdm.write(f"FAILED {stem}: {error}")
        return failures

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_workers, initializer=_init_worker, initargs=(context,)) as pool:
        # imap, not imap_unordered: ordered dispatch is what keeps each worker on one stimulus.
        for stem, error in tqdm(
            pool.imap(_process_recording, tasks), total=len(tasks), desc="recordings", unit="recording"
        ):
            if error:
                failures.append((stem, error))
                tqdm.write(f"FAILED {stem}: {error}")
    return failures


def build_context(
    audio_features_dir: Path, stimulus_psd_dir: Path, coherence_dir: Path, fs: float, nperseg: int, noverlap: int
) -> dict[str, Any]:
    return {
        "audio_features_dir": audio_features_dir,
        "stimulus_psd_dir": stimulus_psd_dir,
        "coherence_dir": coherence_dir,
        "fs": fs,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "blas_threads": BLAS_THREADS_PER_WORKER,
    }


def main() -> int:
    config = load_config(ROOT / "src" / "package_data" / "config.json")
    audio_features_dir, eeg_dir, coherence_dir, stimulus_psd_dir = resolve_configured_inputs(config, ROOT)

    coherence_dir.mkdir(parents=True, exist_ok=True)
    stimulus_psd_dir.mkdir(parents=True, exist_ok=True)

    fs = STIMULUS_FS
    nperseg, noverlap = stimulus_window_sizes(fs)
    tasks = build_tasks(audio_features_dir, eeg_dir, coherence_dir)

    n_stimuli = len({task["audio_name"] for task in tasks})
    print(
        f"{len(tasks)} recording(s) over {n_stimuli} stimulus/stimuli, N_NULL={N_NULL}, "
        f"OVERWRITE={OVERWRITE}\n"
        f"{N_WORKERS} worker(s) x {BLAS_THREADS_PER_WORKER} BLAS thread(s) -> "
        f"{N_WORKERS * BLAS_THREADS_PER_WORKER} of {os.cpu_count()} cores"
    )
    if not tasks:
        print("nothing to do (set OVERWRITE = True to recompute existing outputs)")
        return 0

    context = build_context(audio_features_dir, stimulus_psd_dir, coherence_dir, fs, nperseg, noverlap)
    failures = run_tasks(tasks, context, N_WORKERS)

    print(f"Wrote coherence files to {coherence_dir} and stimulus PSDs to {stimulus_psd_dir}")
    if failures:
        print(f"{len(failures)} recording(s) failed; re-run to retry.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
