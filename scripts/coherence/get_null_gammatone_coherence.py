#!/usr/bin/env python3
"""Pooled null coherence between the broadband (gammatone) envelope and EEG.

Step 3 of the pipeline in ``.claude/coherence_pipeline_plan.md``. Draws ``N_NULL`` global-shift
null coherences per trial and duration-weights them into one complex profile per group per draw,
so each group ends up with an ``N_NULL``-sample null distribution to test its real coherence
against.

Two groupings are needed -- subjects pooled per stimulus, and stimuli pooled per subject -- and
this script produces **both in a single pass**. A trial's draw *i* is the same shift under either
grouping (the RNG is re-seeded from ``NULL_RNG_SEED`` per trial and the shift bounds depend only
on that trial), so the expensive part -- the 500 draws -- is computed once and folded into two
accumulators instead of being recomputed by a second run.

The cost of that is memory: a subject's trials are scattered across every stimulus they heard, so
no loop order keeps both groupings' accumulators contiguous, and all 69 + 81 of them (98.3 GB at
500 draws) have to be live at once. They are therefore **memmap-backed** on disk, and a per-trial
journal makes the whole thing resumable at trial granularity -- see ``accumulate_into_group`` for
the crash-safety protocol that replaces the old design's per-group "output file exists" check.

Three performance facts shape the inner loop; all three are measured, on this dataset, on this box
(24-core Xeon w5-2455X, 62 GB), and the first is worth more than the other two combined:

1. **The per-window FFT must be materialised C-contiguous before the draw loop.**
   ``windowed_fft`` returns ``Xf`` as ``(n_frames, F, Cy)`` over a buffer whose frequency axis has
   a 36 MB stride, so the batched matmul's 2561-element loop over frequency jumps 36 MB per step
   and misses cache on every one. Transposing to ``(F, n_frames, Cy)`` produces a *view*, which
   hoists the syntax out of the loop but not the cost -- every draw re-pays it. One
   ``astype(..., order="C")`` per trial costs ~2 s and takes a draw from **1.007 s to 0.152 s
   (6.6x)**; over 500 draws that is 430 s saved per trial for a 2 s outlay. Max relative error vs
   the strided path: 1.75e-15, i.e. float64 roundoff -- it is the same computation.

2. **The draw is computed in complex64** (``DRAW_DTYPE``), a further 1.8x (0.152 -> 0.083 s) and
   half the resident memory, which is what lets more workers fit. Max relative error vs float64 is
   5.4e-07 -- complex64 roundoff, against output that is stored as ``COHERENCE_DTYPE`` complex64
   regardless. The division by ``denom`` is still done in float64 and cast on assignment.

3. **The draw is single-threaded and memory-bandwidth-bound** (1.00x cores at any
   ``OMP_NUM_THREADS``; ~2.4 GB/s streaming one copy of ``Yf_t`` per draw), so the parallelism
   available is across *trials*, not within a draw. Measured multi-process scaling on contiguous
   arrays: 1.92x at 2 workers, 3.50x at 4, 5.75x at 8, 6.83x at 12 -- efficiency falls off because
   the workers contend for DRAM, not for cores. ``--workers 8`` is the default: past that you buy
   15% more throughput for 50% more memory, and memory is the binding constraint here.

Together: ~3.3 days per grouping (6.6 days for both) becomes **~2 h for both**.

    python scripts/coherence/get_null_gammatone_coherence.py --n-null 500

Weights come from the EEG recording, computed here from the recording being loaded anyway rather
than read from ``get_coherence.py``'s saved ``duration`` -- that field is only correct in files
written after the duration fix, and deriving it locally keeps this script independent of whether
those files have been regenerated yet.
"""

from __future__ import annotations

import argparse
import os

# The draw is single-threaded (fact 3 above) and the pool oversubscribes the box on its own, so
# pin every numeric library to one thread before numpy is imported anywhere below.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import multiprocessing as mp
import shutil
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (SRC, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import mne
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from tqdm import tqdm

from coherence.coherence import (
    WindowedSpectrum,
    _auto_power,
    _conj_transpose_for_matmul,
    contiguous_for_matmul,
    shifted_cross_spectrum,
    truncate_spectrum,
    windowed_fft,
)
from coherence.pipeline import (
    COHERENCE_DTYPE,
    DRAW_DTYPE,
    EXCLUDED_AUDIO_SUBSTRINGS,
    EXCLUDED_CONDITIONS,
    NULL_RNG_SEED,
    STIMULUS_FS,
    DEFAULT_REFERENCE,
    REFERENCES,
    build_trial_index,
    filter_trials_for_reference,
    fsync_path,
    load_referenced_eeg,
    null_shift_bounds,
    reference_channel_names,
    resolve_configured_inputs,
    resolve_reference_dirs,
    stimulus_window_sizes,
)
from transcription.config import load_config

mne.set_log_level("WARNING")

# Null draws per trial. Each unit costs every group one (F, 1, n_eeg) complex64 plane -- 1.31 MB --
# so a group's accumulator is N_NULL x 1.31 MB (655.6 MB at 500), and the run writes one such file
# per group (69 stimuli + 81 subjects = 98.3 GB of accumulator at 500 draws).
N_NULL = 500

# Default worker count. Chosen on measured scaling *and* memory: each worker holds a contiguous
# complex64 (F, n_frames, Cy) spectrum (~1.2 GB) plus its draw stack (655.6 MB at 500 draws), and
# transiently ~4.7 GB more inside windowed_fft, which FFT_CONCURRENCY caps.
DEFAULT_WORKERS = 8

# windowed_fft materialises a (n_frames, nperseg, n_eeg) float64 segment array *and* its complex128
# transform at the same time -- ~4.7 GB of transient per call, and the one place a pool of workers
# can collectively blow the 62 GB box. It is only ~5% of a trial's wall time, so capping how many
# workers may be inside it at once costs almost nothing and bounds the peak.
FFT_CONCURRENCY = 3

# grouping name -> the trial-index column whose distinct values define the groups.
GROUP_COLUMNS = {"stimulus": "audio_name", "subject": "sub"}
GROUPINGS = ("stimulus", "subject")

_WORKER: dict[str, Any] = {}


# --------------------------------------------------------------------------------------------
# Output locations
#
# The trial index, the exclusion rules and the null-shift convention now come from
# coherence.pipeline, so that this script and get_null_cochleogram_coherence.py cannot drift
# apart on them. Only the directory layout this script owns is defined here.
# --------------------------------------------------------------------------------------------


def resolve_output_dir(config: dict[str, Any], grouping: str, reference: str = DEFAULT_REFERENCE,
                       root: str | Path | None = None) -> Path:
    """Where a grouping's finished per-group files go, under that reference's ``null_coherence_dir``."""
    return resolve_reference_dirs(config, reference, root)[1] / f"gammatone_null_by_{grouping}"


# --------------------------------------------------------------------------------------------
# The draw itself
# --------------------------------------------------------------------------------------------


def iter_null_coherence(
    Xf_t: np.ndarray,
    Yf_t: np.ndarray,
    denom: np.ndarray,
    fs: float,
    hop: int,
    scale: float,
    nperseg: int,
    n_null: int,
    min_shift_seconds: float,
    max_shift_seconds: float,
) -> Iterator[np.ndarray]:
    """Yield ``n_null`` global-shift null complex coherencies, one (F, 1, Cy) draw at a time.

    Mirrors coherence.py's ``null_coherence_shuffled_windows_from_prep`` in mode="global_shift"
    -- same bounds, same fixed-seed RNG call sequence, so draw i is the same shift it would be
    there -- but routes each draw through ``shifted_cross_spectrum`` and takes the caller's cached
    Pxx/Pyy (already folded into ``denom``) rather than recomputing them.

    The division by ``denom`` stays in float64 and is cast to complex64 by the caller's assignment
    into the draw stack, so only the matmul runs at reduced precision.

    Re-seeded from NULL_RNG_SEED per trial, so a trial's draw i is independent of how many trials
    preceded it -- which is what makes a trial's contribution to its stimulus group and to its
    subject group the same draw, and what makes a resumed run identical to an uninterrupted one.
    """
    rng = np.random.default_rng(NULL_RNG_SEED)
    n_frames = Yf_t.shape[1]

    min_shift_frames = int(np.ceil(min_shift_seconds * fs / hop))
    if min_shift_frames * 2 >= n_frames:
        raise ValueError(
            f"min_shift_seconds={min_shift_seconds} needs {min_shift_frames} frames of clearance "
            f"in each direction, but only {n_frames} frames are available."
        )
    max_shift_frames = min(n_frames - min_shift_frames, int(np.floor(max_shift_seconds * fs / hop)))
    if max_shift_frames < min_shift_frames:
        raise ValueError(f"max_shift_seconds={max_shift_seconds} is smaller than min_shift_seconds={min_shift_seconds}")

    for _ in range(n_null):
        shift = int(rng.integers(min_shift_frames, max_shift_frames + 1))
        yield shifted_cross_spectrum(Xf_t, Yf_t, shift, scale, nperseg) / denom


class StimulusCache:
    """The gammatone envelope's windowed FFT and cached Pxx for one stimulus at a time.

    Kept per worker. The broadband envelope's FFT is only ~0.05 s (a (n_frames, F, 1) array,
    ~1/60th of the subband equivalent), so this is a minor saving rather than the load-bearing
    optimisation it would be for the subband script -- which is why trials can be dispatched to
    workers in whatever order the pool likes without worrying about cache locality.
    """

    def __init__(self, audio_features_dir: Path, stimulus_psd_dir: Path, fs: float, nperseg: int, noverlap: int):
        self.audio_features_dir = audio_features_dir
        self.stimulus_psd_dir = stimulus_psd_dir
        self.fs, self.nperseg, self.noverlap = fs, nperseg, noverlap
        self.audio_name: str | None = None
        self.entry: tuple[WindowedSpectrum, np.ndarray, int] | None = None

    def get(self, audio_name: str) -> tuple[WindowedSpectrum, np.ndarray, int]:
        """(windowed spectrum, cached Pxx, n_samples) for this stimulus's gammatone envelope."""
        if self.audio_name != audio_name:
            # Only the broadband envelope is read: .npz members load lazily, so the 60-channel
            # subband array sitting in the same file is never touched.
            envelope = np.load(self.audio_features_dir / f"{audio_name}.npz")["gammatone_envelope"]
            spectrum = windowed_fft(envelope, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap, axis=1)
            Pxx = np.load(self.stimulus_psd_dir / f"{audio_name}_stimulus_psd.npz")["gammatone_psd"]
            self.audio_name, self.entry = audio_name, (spectrum, Pxx, envelope.shape[1])
        return self.entry


def compute_trial_draws(
    audio_name: str,
    eeg_stem: str,
    reference: str,
    eeg_dir: Path,
    stimulus_cache: StimulusCache,
    fs: float,
    nperseg: int,
    n_null: int,
    fft_semaphore: Any = None,
) -> tuple[np.ndarray, np.ndarray, float, float, list[str]] | None:
    """All ``n_null`` null draws for one trial, plus its weights.

    ``None`` when the reference cannot be applied to this recording -- only the linked-mastoid
    case, for the 21 recordings carrying no A1/A2.

    Returns ``(stack, real, duration, analysed_duration, channel_names)`` where ``stack`` is ``(n_null, F, 1, Cy)`` in
    ``DRAW_DTYPE``, **already multiplied by ``duration``** so the caller can fold it into both
    groupings' accumulators with a plain ``+=`` and no temporary.

    ``Pyy`` and the real coherence are computed here rather than read from get_coherence.py's
    per-trial output. That output only exists for the average reference, and recomputing costs one
    extra matmul out of ``n_null + 1`` -- 0.2% at 500 draws -- against a per-trial file this
    script would otherwise have to have for every reference.
    """
    x_full, cached_Pxx, n_stim_samples = stimulus_cache.get(audio_name)

    loaded = load_referenced_eeg(eeg_dir / f"{eeg_stem}.fif", reference)
    if loaded is None:
        return None  # no A1/A2 under linked mastoids: the caller journals it as skipped
    eeg_data, eeg_fs, channel_names = loaded
    if eeg_fs != fs:
        raise ValueError(f"{eeg_stem}: EEG sample rate {eeg_fs} does not match stimulus rate {fs}")

    # windowed_fft is the memory peak of the whole script; bound how many workers are in it.
    if fft_semaphore is not None:
        fft_semaphore.acquire()
    try:
        y_full = windowed_fft(eeg_data, fs=fs, nperseg=nperseg, noverlap=stimulus_cache.noverlap, axis=1)
        # Reproduce get_coherence.py's truncation to a common length by dropping trailing windows.
        n_frames = min(x_full.n_frames, y_full.n_frames)
        hop = y_full.hop
        # Taken before the complex128 original is released: _auto_power wants the (n_frames, F, C)
        # spectrum, not the contiguous transpose that outlives it.
        Pyy = _auto_power(truncate_spectrum(y_full, n_frames))
        Yf_t = contiguous_for_matmul(y_full.Xf[:n_frames], DRAW_DTYPE)
    finally:
        if fft_semaphore is not None:
            fft_semaphore.release()
    del y_full  # release the 2.37 GB complex128 original now that the complex64 copy exists

    x_spectrum = truncate_spectrum(x_full, n_frames)
    Xf_t = _conj_transpose_for_matmul(x_spectrum).astype(DRAW_DTYPE, order="C")

    # The saved stimulus PSD covers the full-length envelope, so it is the right Pxx only when
    # no stimulus window was dropped. Recomputing is cheap at Cx=1, and rare in practice.
    Pxx = cached_Pxx if n_frames == x_full.n_frames else _auto_power(x_spectrum)
    denom = np.sqrt(Pxx[:, :, None] * Pyy[:, None, :])

    duration = eeg_data.shape[1] / eeg_fs
    analysed_duration = min(eeg_data.shape[1], n_stim_samples) / fs
    min_shift_seconds, max_shift_seconds = null_shift_bounds(analysed_duration)

    stack = np.empty((n_null, *denom.shape), dtype=DRAW_DTYPE)
    for null_idx, draw in enumerate(
        iter_null_coherence(
            Xf_t,
            Yf_t,
            denom,
            fs,
            hop,
            x_spectrum.scale,
            nperseg,
            n_null,
            min_shift_seconds,
            max_shift_seconds,
        )
    ):
        stack[null_idx] = draw  # complex128 -> complex64 on assignment

    # The real estimate is the shift-0 case of the same kernel, so it reuses both operands.
    real = (shifted_cross_spectrum(Xf_t, Yf_t, 0, x_spectrum.scale, nperseg) / denom).astype(COHERENCE_DTYPE)

    stack *= duration
    return stack, real, duration, analysed_duration, channel_names


# --------------------------------------------------------------------------------------------
# Accumulators: memmap-backed, journalled, resumable per trial
# --------------------------------------------------------------------------------------------


class GroupStore:
    """The on-disk working state for one group of one grouping.

    Four files, all under ``<work_dir>/<grouping>/``:

    ``<group>.null.npy``    (n_null, F, 1, Cy) complex64 -- the duration-weighted null sum
    ``<group>.real.npy``    (F, 1, Cy) complex64          -- the duration-weighted real sum
    ``<group>.journal``     one "stem<TAB>duration<TAB>analysed_duration" line per folded-in trial
    ``<group>.pending``     present only while a trial is mid-fold

    Sums are stored **unnormalised** and divided by the journal's total duration at finalize time,
    which is what makes the fold order-independent and therefore safe to resume and to parallelise.

    Crash safety. ``+=`` is not idempotent, so resuming has to know exactly which trials are in an
    accumulator, and the risk is a crash *between* the fold and the journal append. The pending
    marker closes that window: it is written before the fold and removed after the journal is
    fsynced, so a marker surviving startup means "this group may contain a partial or duplicated
    contribution" and the group is reset and recomputed from its trials. Everything happens under
    the group's lock, so at most one trial is ever mid-fold per group, and a reset costs only that
    group's handful of trials rather than the run.
    """

    def __init__(self, work_dir: Path, grouping: str, group_id: str):
        base = work_dir / grouping
        self.null_path = base / f"{group_id}.null.npy"
        self.real_path = base / f"{group_id}.real.npy"
        self.journal_path = base / f"{group_id}.journal"
        self.pending_path = base / f"{group_id}.pending"

    def create(self, null_shape: tuple[int, ...], real_shape: tuple[int, ...]) -> None:
        """Allocate zeroed accumulators. Sparse on write, so this is instant despite the size."""
        self.null_path.parent.mkdir(parents=True, exist_ok=True)
        open_memmap(self.null_path, mode="w+", dtype=COHERENCE_DTYPE, shape=null_shape)
        open_memmap(self.real_path, mode="w+", dtype=COHERENCE_DTYPE, shape=real_shape)
        self.journal_path.write_text("")

    def reset(self) -> None:
        for path in (self.null_path, self.real_path, self.journal_path, self.pending_path):
            path.unlink(missing_ok=True)

    def exists(self) -> bool:
        return self.null_path.exists() and self.real_path.exists() and self.journal_path.exists()

    def journal(self) -> list[tuple[str, float, float]]:
        if not self.journal_path.exists():
            return []
        entries = []
        for line in self.journal_path.read_text().splitlines():
            if not line.strip():
                continue
            stem, duration, analysed = line.split("\t")
            entries.append((stem, float(duration), float(analysed)))
        return entries

    def folded_stems(self) -> set[str]:
        return {stem for stem, _, _ in self.journal()}

    def fold_skipped(self, stem: str) -> None:
        """Record a trial the reference cannot be applied to: journalled, nothing accumulated.

        A zero duration is the marker. It keeps the group's membership check satisfied -- so the
        group can finalise rather than waiting forever for a trial that will never arrive -- while
        contributing nothing to either accumulator or to the denominator, which is exactly what
        skipping means. Durations are never legitimately zero: they come from a recording's length.

        This is why there is no pre-run scan for the 21 recordings without A1/A2. Reading 563 FIF
        headers over NFS measured 18 minutes, paid again on every resume, and the worker has to
        open the recording anyway.
        """
        with self.journal_path.open("a") as handle:
            handle.write(f"{stem}\t{0.0!r}\t{0.0!r}\n")
            handle.flush()
            os.fsync(handle.fileno())

    def fold(self, stack: np.ndarray, real: np.ndarray, stem: str, duration: float, analysed_duration: float) -> None:
        """Add one trial's contribution. Call under the group's lock.

        ``stack`` arrives already multiplied by ``duration`` (it is folded into two groupings, so
        weighting it once at the source saves a 655.6 MB temporary per side); ``real`` is the
        trial's unweighted coherence and is weighted here.
        """
        self.pending_path.write_text(stem)
        fsync_path(self.pending_path)

        null_acc = open_memmap(self.null_path, mode="r+")
        null_acc += stack
        null_acc.flush()
        del null_acc

        real_acc = open_memmap(self.real_path, mode="r+")
        real_acc += duration * real
        real_acc.flush()
        del real_acc

        with self.journal_path.open("a") as handle:
            handle.write(f"{stem}\t{duration!r}\t{analysed_duration!r}\n")
            handle.flush()
            os.fsync(handle.fileno())

        self.pending_path.unlink(missing_ok=True)


def entries_all(store: "GroupStore") -> list[tuple[str, float, float]]:
    """The group's journal, skipped entries included."""
    return store.journal()


def finalize_group(store: GroupStore, output_path: Path, group_id: str, grouping: str, n_null: int,
                   freqs: np.ndarray, reference: str, channel_names: list[str]) -> bool:
    """Normalise a completed group's sums by its total duration and write the final .npz.

    Done one group at a time and the working files removed immediately, so the transient headroom
    this needs is one group (655.6 MB) rather than a second copy of all 98.3 GB.
    """
    # Zero-duration entries are trials this reference could not be applied to (see fold_skipped):
    # listed in the output, but out of the trial set, the weights and the denominator.
    skipped = [stem for stem, duration, _ in entries_all(store) if duration == 0.0]
    entries = [entry for entry in entries_all(store) if entry[1] != 0.0]
    total_duration = sum(duration for _, duration, _ in entries)
    total_analysed = sum(analysed for _, _, analysed in entries)
    if not entries:
        # Every one of this group's recordings was skipped, so there is nothing to divide by.
        # Under linked mastoids this is real: 3 of the 81 subjects (052, 061, 064) carry no A1/A2
        # on any recording. Writing the file anyway would put an all-NaN average on disk that
        # looks exactly like a finished group.
        #
        # Reset so the group leaves nothing behind: its accumulators are sparse (never written to)
        # but they keep the working directory alive, and a run that produced every file it could
        # should finish with no working state. A re-run recreates the group, re-dispatches its
        # trials -- all of which are skips, costing one header read each -- and drops it again.
        store.reset()
        return False

    null_acc = np.array(open_memmap(store.null_path, mode="r"))
    real_acc = np.array(open_memmap(store.real_path, mode="r"))
    # Written to a temporary name and renamed, so a half-written .npz can never be mistaken for a
    # finished group by the next run's resume check. np.savez is handed an open file rather than a
    # path because it appends ".npz" to any path that lacks the suffix.
    tmp_path = output_path.with_name(output_path.name + ".partial")
    with tmp_path.open("wb") as handle:
        np.savez(
            handle,
            name=str(group_id),
            group_by=grouping,
            reference=reference,
            # Saved because it has to be: the channel axis is 63 wide under FCz and 64 otherwise,
            # so a topography cannot be plotted from these without knowing which channels it holds.
            channel_names=np.array(channel_names),
            # Recordings this reference could not be applied to, dataset-wide. Empty but for
            # linked mastoids, where 21 recordings carry no A1/A2 and never enter any group.
            skipped_stems=np.array(skipped),
            n_null=n_null,
            frequencies=freqs,
            null_gammatone_coherence=(null_acc / total_duration).astype(COHERENCE_DTYPE),
            gammatone_coherence=(real_acc / total_duration).astype(COHERENCE_DTYPE),
            n_trials=len(entries),
            total_duration=total_duration,
            total_analysed_duration=total_analysed,
            trial_stems=np.array([stem for stem, _, _ in entries]),
            trial_durations=np.array([duration for _, duration, _ in entries]),
        )
    tmp_path.replace(output_path)
    store.reset()
    return True


# --------------------------------------------------------------------------------------------
# Worker pool
# --------------------------------------------------------------------------------------------


def _pin_to_one_thread() -> Any:
    """Hold every numeric library to one thread in this process, whatever the import order was.

    The module-level ``os.environ.setdefault`` above only binds if this script is the first thing
    to import numpy, which is true when it is run as a script and false when it is imported (by a
    test, or a driver that already pulled numpy in). The distinction is not cosmetic: with a
    contiguous operand the batched matmul *does* engage a threaded BLAS, so an unpinned pool
    oversubscribes 24 cores several times over on an operation that is already
    memory-bandwidth-bound -- measured at ~6x CPU per worker and a hard collapse in throughput.

    The returned controller must stay referenced for the limit to hold, hence the caller stashing
    it. threadpoolctl is not a declared dependency, so a missing one falls back to the env vars.
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return None
    return threadpool_limits(limits=1)


def _init_worker(context: dict[str, Any], locks: dict[tuple[str, str], Any], fft_semaphore: Any) -> None:
    _WORKER["thread_limit"] = _pin_to_one_thread()
    _WORKER.update(context)
    _WORKER["locks"] = locks
    _WORKER["fft_semaphore"] = fft_semaphore
    _WORKER["stimulus_cache"] = StimulusCache(
        context["audio_features_dir"],
        context["stimulus_psd_dir"],
        context["fs"],
        context["nperseg"],
        context["noverlap"],
    )


def _process_trial(task: dict[str, Any]) -> tuple[str, str | None]:
    """Compute one trial's draws and fold them into every grouping that still needs it."""
    try:
        computed = compute_trial_draws(
            task["audio_name"],
            task["eeg_stem"],
            _WORKER["reference"],
            _WORKER["eeg_dir"],
            _WORKER["stimulus_cache"],
            _WORKER["fs"],
            _WORKER["nperseg"],
            _WORKER["n_null"],
            _WORKER["fft_semaphore"],
        )
        if computed is None:
            for grouping, group_id in task["sides"]:
                store = GroupStore(_WORKER["work_dir"], grouping, group_id)
                with _WORKER["locks"][(grouping, group_id)]:
                    store.fold_skipped(task["eeg_stem"])
            return task["eeg_stem"], None
        stack, real, duration, analysed_duration, channel_names = computed
        # A layout change mid-dataset would silently mis-label every topography drawn from the
        # average, so it is an error rather than something the channel axis absorbs.
        if channel_names != _WORKER["channel_names"]:
            raise ValueError(f"{task['eeg_stem']}: channel layout differs from the first recording")

        for grouping, group_id in task["sides"]:
            store = GroupStore(_WORKER["work_dir"], grouping, group_id)
            with _WORKER["locks"][(grouping, group_id)]:
                store.fold(stack, real, task["eeg_stem"], duration, analysed_duration)
        return task["eeg_stem"], None
    except Exception as exc:  # keep one bad trial from killing a multi-hour run
        return task["eeg_stem"], f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


def check_existing_outputs(output_dirs: dict[str, Path], n_null: int, reference: str) -> None:
    """Refuse to mix draw counts, or references, in one output directory.

    A finished .npz from a run at a different ``n_null`` is a valid result worth keeping (the 50
    draws of an earlier run are exactly the first 50 of a longer one, so it makes a good
    cross-check), but it cannot be pooled with, or resumed into, this run's files.

    The reference check is the one that matters when a config has no ``references`` block: every
    reference then resolves to the same directory, and the files are the same shape and the same
    names, so nothing downstream could tell that a directory held two references' results mixed
    together. Files written before the reference was recorded are treated as the average
    reference, which is what they are.
    """
    for output_dir in output_dirs.values():
        for existing in sorted(output_dir.glob("*_gammatone_null.npz")):
            with np.load(existing) as saved:
                found = int(saved["n_null"])
                found_reference = str(saved["reference"]) if "reference" in saved.files else DEFAULT_REFERENCE
            if found != n_null:
                raise SystemExit(
                    f"{output_dir} already holds n_null={found} results (e.g. {existing.name}) but this run "
                    f"is n_null={n_null}. Move the old run aside first, e.g.\n"
                    f"    mv {output_dir} {output_dir}_n{found}"
                )
            if found_reference != reference:
                raise SystemExit(
                    f"{output_dir} already holds {found_reference}-referenced results (e.g. {existing.name}) "
                    f"but this run is {reference}. Give each reference its own directory: add a "
                    f'"references" block to config.json (see resolve_reference_dirs), or move the old '
                    f"run aside first, e.g.\n    mv {output_dir} {output_dir}_{found_reference}"
                )
            break  # one file settles it; they are all written by the same run


def plan_run(
    df: pd.DataFrame,
    groupings: tuple[str, ...],
    work_dir: Path,
    output_dirs: dict[str, Path],
    null_shape: tuple[int, ...],
    real_shape: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, pd.DataFrame]], int]:
    """Resume state: which (grouping, group) still need which trials.

    Groups already finalised are skipped outright; groups with a surviving pending marker are
    reset and recomputed; the rest resume from their journal. A trial is dispatched once and
    folded into every side that still wants it, so a run interrupted after the stimulus side of a
    trial landed but before the subject side did costs one recomputation, not two.
    """
    groups_by_grouping: dict[str, dict[str, pd.DataFrame]] = {}
    sides_by_stem: dict[str, list[tuple[str, str]]] = {}
    n_reset = 0

    for grouping in groupings:
        column = GROUP_COLUMNS[grouping]
        groups_by_grouping[grouping] = {str(gid): rows for gid, rows in df.groupby(column)}
        for group_id, rows in groups_by_grouping[grouping].items():
            if (output_dirs[grouping] / f"{group_id}_gammatone_null.npz").exists():
                continue
            store = GroupStore(work_dir, grouping, group_id)
            if store.pending_path.exists():
                store.reset()
                n_reset += 1
            if not store.exists():
                store.create(null_shape, real_shape)
            done = store.folded_stems()
            for stem in rows["eeg_stem"]:
                if stem not in done:
                    sides_by_stem.setdefault(stem, []).append((grouping, group_id))

    by_stem = df.set_index("eeg_stem")
    tasks = [
        {
            "eeg_stem": stem,
            "audio_name": by_stem.at[stem, "audio_name"],
            "sides": sides,
        }
        for stem, sides in sides_by_stem.items()
    ]
    return tasks, groups_by_grouping, n_reset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-null", type=int, default=N_NULL, help=f"null draws per trial (default {N_NULL})")
    parser.add_argument(
        "--reference",
        choices=REFERENCES,
        default=DEFAULT_REFERENCE,
        help=f"EEG reference (default {DEFAULT_REFERENCE}); outputs go under that reference's "
             "configured null_coherence_dir",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"parallel trials (default {DEFAULT_WORKERS})")
    parser.add_argument(
        "--groupings",
        default=",".join(GROUPINGS),
        help=f"comma-separated subset of {','.join(GROUPINGS)} (default both, in one pass)",
    )
    parser.add_argument("--restart", action="store_true", help="discard any resumable working state and start clean")
    args = parser.parse_args(argv)

    groupings = tuple(g.strip() for g in args.groupings.split(",") if g.strip())
    unknown = set(groupings) - set(GROUPINGS)
    if unknown:
        raise SystemExit(f"unknown grouping(s): {', '.join(sorted(unknown))}")

    config = load_config(ROOT / "src" / "package_data" / "config.json")
    audio_features_dir, eeg_dir, _, stimulus_psd_dir = resolve_configured_inputs(config, ROOT)
    null_dir = resolve_reference_dirs(config, args.reference, ROOT)[1]
    if not eeg_dir.exists():
        raise FileNotFoundError(f"{eeg_dir} does not exist; it is the input this script pools over")

    output_dirs = {g: resolve_output_dir(config, g, args.reference, ROOT) for g in groupings}
    for output_dir in output_dirs.values():
        output_dir.mkdir(parents=True, exist_ok=True)
    check_existing_outputs(output_dirs, args.n_null, args.reference)

    work_dir = null_dir / "gammatone_null_work"
    if args.restart and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Derived from the recordings themselves, not from get_coherence.py's per-trial output: that
    # output exists only for the average reference, and this script needs a trial index for all
    # three. Trials a reference cannot be applied to are discovered by the worker that loads them
    # and journalled at zero weight (GroupStore.fold_skipped), not scanned for up front.
    df = build_trial_index(eeg_dir)
    fs = STIMULUS_FS
    nperseg, noverlap = stimulus_window_sizes(fs)

    # Shapes and frequencies are trial-independent -- they follow from fs/nperseg and the channel
    # count the reference produces -- so one recording's header settles them for every accumulator.
    channel_names = reference_channel_names(eeg_dir / f"{df.iloc[0]['eeg_stem']}.fif", args.reference)
    real_shape = (nperseg // 2 + 1, 1, len(channel_names))
    null_shape = (args.n_null, *real_shape)
    freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)

    tasks, groups_by_grouping, n_reset = plan_run(df, groupings, work_dir, output_dirs, null_shape, real_shape)

    accumulator_gb = sum(len(groups_by_grouping[g]) for g in groupings) * np.prod(null_shape) * 8 / 1e9
    print(
        f"reference={args.reference}, {len(df)} trials -> " + " + ".join(f"{len(groups_by_grouping[g])} {g}" for g in groupings) + " groups\n"
        f"n_null={args.n_null}, workers={args.workers}, accumulators ~{accumulator_gb:.1f} GB in {work_dir}"
    )
    if n_reset:
        print(f"reset {n_reset} group(s) that were mid-fold when the previous run stopped")
    if not tasks:
        print("nothing to compute; finalizing")

    context = {
        "reference": args.reference,
        "channel_names": channel_names,
        "eeg_dir": eeg_dir,
        "audio_features_dir": audio_features_dir,
        "stimulus_psd_dir": stimulus_psd_dir,
        "fs": fs,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "n_null": args.n_null,
        "work_dir": work_dir,
    }

    failures: list[tuple[str, str]] = []
    if tasks:
        ctx = mp.get_context("fork")
        manager = ctx.Manager()
        locks = {
            (grouping, group_id): manager.Lock()
            for grouping in groupings
            for group_id in groups_by_grouping[grouping]
        }
        fft_semaphore = manager.Semaphore(FFT_CONCURRENCY)
        with ctx.Pool(
            processes=args.workers, initializer=_init_worker, initargs=(context, locks, fft_semaphore)
        ) as pool:
            for stem, error in tqdm(
                pool.imap_unordered(_process_trial, tasks), total=len(tasks), desc="trials", unit="trial"
            ):
                if error:
                    failures.append((stem, error))
                    tqdm.write(f"FAILED {stem}: {error}")

    n_written = 0

    empty_groups: list[str] = []
    for grouping in groupings:
        for group_id, rows in groups_by_grouping[grouping].items():
            output_path = output_dirs[grouping] / f"{group_id}_gammatone_null.npz"
            if output_path.exists():
                continue
            store = GroupStore(work_dir, grouping, group_id)
            if not store.exists():
                continue
            if store.folded_stems() != set(rows["eeg_stem"]):
                continue  # incomplete: leave the accumulator for a resumed run
            if finalize_group(store, output_path, group_id, grouping, args.n_null, freqs,
                              args.reference, channel_names):
                n_written += 1
            else:
                empty_groups.append(f"{grouping}/{group_id}")

    if empty_groups:
        print(f"{len(empty_groups)} group(s) had no trial this reference could be applied to and "
              f"were not written: {', '.join(sorted(empty_groups))}")
    print(f"wrote {n_written} group file(s) to " + ", ".join(str(d) for d in output_dirs.values()))
    if failures:
        print(f"{len(failures)} trial(s) failed; their groups were left unfinalised. Re-run to retry.")
        return 1
    if not any(work_dir.rglob("*.npy")):
        shutil.rmtree(work_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
