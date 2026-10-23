#!/usr/bin/env python3
"""Pooled null coherence between the 60-band cochleogram and EEG, over all included trials.

Step 2 of the pipeline in ``.claude/coherence_pipeline_plan.md``. Draws ``N_NULL`` global-shift
null coherences per trial and duration-weights them into a **single** global profile per draw, so
the dataset ends up with one ``N_NULL``-sample null distribution to test its pooled real coherence
against. There is no per-trial output and no grouping switch: the result is one aggregation over
all 563 included trials.

    python scripts/coherence/get_null_cochleogram_coherence.py --n-null 500

Output is **one .npz per cochleogram channel** -- 60 subbands plus the broadband gammatone
cross-check -- each holding that channel's ``(n_null, F, 1, 64)`` null array, its pooled real
coherence, and the two frequency axes, which are different things and both needed:

    frequencies       (2561,) Hz -- the coherence/FFT axis, 0 to 512 Hz, shared by every channel
    centre_frequency  scalar  Hz -- this channel's gammatone filter centre, one of the 60
                                    ERB-spaced values (50 Hz to 8 kHz) in the feature files;
                                    NaN for the broadband channel, which has no single centre

Each file is 655.6 MB and self-contained, so analysing one subband downstream never touches the
other 39 GB.

**The per-channel files are the accumulators**, not a copy of them
------------------------------------------------------------------
Each channel gets its own ``(n_null, F, 1, 64)`` memmap from the start, and a draw is scattered
across the 61 of them as it is computed. An earlier revision accumulated into one combined
``(n_null, F, 61, 64)`` array and split it at the end, which needed the combined array *and* the
61 files live at once -- 80 GB instead of 40 GB, for a copy that buys nothing. Finalize now
rewrites one channel at a time into its .npz (normalising by total duration and attaching the
metadata) and unlinks the .npy as it goes, so peak disk is **39.99 GB plus one channel**.

The cost of scattering is a strided read of ``cross[:, k, :]`` per channel per draw. That reads 64
contiguous complex64 -- 512-byte runs -- so cache lines are fully used and the penalty is
prefetcher efficiency rather than wasted bandwidth. It is expected to be small against a ~128 ms
matmul but has *not* been measured; ``bench_cochleogram_null.py --scatter`` is the check.

Three facts shape the rest, and the first two invert script 3's conclusions
---------------------------------------------------------------------------
1. **``Cx`` is the arithmetic intensity, so the 61 channels stay in one matmul.**
   Per frequency the kernel is ``(Cx, n_frames) x (n_frames, Cy)``: ``8*Cx*nf*Cy`` FLOPs against
   ``nf*Cy*8`` bytes of ``Yf``, i.e. exactly ``Cx`` FLOP/byte at complex64. This box balances
   around 15, so splitting the channel axis walks off a cliff. Measured at 24 threads:

       Cx=61: 653.9 GF/s   Cx=30: 433.0   Cx=15: 331.1   Cx=1: 42.9

   Four groups of 15 channels would run the same 23.6 PFLOP at 331 GF/s -- **20 h instead of
   10 h**. The channel axis is therefore split only for *storage*, where it is nearly free.

2. **Threads, not processes.** numpy's matmul releases the GIL (measured 6.44x at 24 threads,
   Cx=61), so parallelism goes across *draws* inside one process: each thread owns a contiguous
   range of draws and writes disjoint rows of every channel accumulator, so no lock is needed
   anywhere. A process pool would instead duplicate ``Yf_t`` (1.37 GB) and ``Xf_t`` (1.31 GB) per
   worker -- ~32 GB at 12 workers -- and that memory is worth more as accumulator. This is the
   opposite of script 3, which pins each worker to one thread because its bandwidth-bound
   ``Cx = 1`` kernel collapses under an unpinned pool.

3. **Blocking is over draws, not trials.** A trial's full draw stack is 39.99 GB, so script 3's
   compute-then-fold is impossible; and streaming draws straight into the accumulators has every
   trial touch all 40 GB, which on a 62 GB box is ``563 x 2 x 40 GB`` = ~45 TB of real disk
   traffic. Instead the outer loop walks blocks of ``DRAW_BLOCK`` draws and the inner loop walks
   all 563 trials, so a pass only touches ``DRAW_BLOCK x 79.99 MB`` across the 61 files --
   resident for the whole pass, read in once, written back once. Accumulator traffic falls to
   **39.99 GB written once**. The price is that each trial's EEG load and FFT repeats once per
   block, which the prefetch hides behind the matmul.

**Measurement caveat.** The throughput figures above were taken on 2026-08-13 while another job
was running on this box, so they are if anything understated, and the *ratios* are more
trustworthy than the absolutes. Re-run ``scripts/coherence/bench_cochleogram_null.py`` on an idle
machine before trusting the wall clock, and validate end to end with ``--max-trials 8``.

Projected wall clock at 500 draws: **~10.0 h of matmul** (23.6 PFLOP at 653.9 GF/s) plus load/FFT
and finalize. Unlike script 3's figures this has *not* been validated on real data.

Weights come from the EEG recording, derived here rather than read from ``get_coherence.py``'s
saved ``duration``: that field is only correct in files written after the duration fix, and the
617 legacy files may not have been migrated yet (Q7).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
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
    _auto_power,
    auto_power_transposed,
    contiguous_conj_transpose,
    contiguous_for_matmul,
    shifted_cross_spectrum,
    truncate_spectrum,
    windowed_fft,
)
from coherence.pipeline import (
    COHERENCE_DTYPE,
    DRAW_DTYPE,
    GAMMATONE_FEATURE_KEY,
    STIMULUS_FS,
    SUBBAND_FEATURE_KEY,
    DEFAULT_REFERENCE,
    REFERENCES,
    build_trial_index,
    fsync_path,
    load_centre_frequencies,
    load_referenced_eeg,
    null_shifts,
    reference_channel_names,
    resolve_configured_inputs,
    resolve_reference_dirs,
    stimulus_window_sizes,
)
from transcription.config import load_config

mne.set_log_level("WARNING")

N_NULL = 500

# Draws resident per pass. The block spans DRAW_BLOCK x 79.99 MB across the 61 channel files and
# must stay in page cache for a whole pass over the trials, while ceil(n_null / DRAW_BLOCK) is how
# many times each trial's EEG load + FFT repeats (mostly hidden by the prefetch).
#   250 -> 20.0 GB resident, 2 passes   <- default; ~28 GB peak RSS, comfortable on 62 GB
#   500 -> 40.0 GB resident, 1 pass     (~48 GB peak; fits, but leaves nothing for anything else)
# The ~28 GB peak is the block plus: two live Yf_t while the prefetch overlaps (2 x 1.37 GB), one
# or two Xf_t across a stimulus change (1.31 GB each), 24 threads x one 79.99 MB cross-spectrum,
# and a ~5.5 GB transient inside whichever windowed_fft holds _FFT_LOCK.
DRAW_BLOCK = 250

# Python threads over the draw axis, each calling a single-threaded matmul. 24 = physical cores.
DEFAULT_THREADS = 24

# windowed_fft materialises a (n_frames, nperseg, n_channels) float64 segment array *and* its
# complex128 transform at once -- ~5.5 GB for the EEG, ~5.2 GB for the 61-channel stimulus. The
# prefetch thread can overlap with the main thread's FFT, so a lock keeps the peak to one.
_FFT_LOCK = threading.Lock()


# --------------------------------------------------------------------------------------------
# Shifts
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# Stimulus and trial preparation
# --------------------------------------------------------------------------------------------


class StimulusCache:
    """One stimulus's 61-feature spectrum, held in its matmul-ready form.

    The raw ``(n_frames, F, 61)`` complex128 spectrum is ~2.62 GB, so it is converted to the
    contiguous complex64 ``(F, 61, n_frames)`` operand (1.31 GB) and the original dropped at once.
    Unlike script 3 -- where the stimulus FFT is ~0.05 s and caching is a nicety -- this one is
    seconds and gigabytes, so trials are walked in stimulus order (563 trials over 69 stimuli) to
    make the cache hit.
    """

    def __init__(self, audio_features_dir: Path, stimulus_psd_dir: Path, fs: float, nperseg: int, noverlap: int):
        self.audio_features_dir = audio_features_dir
        self.stimulus_psd_dir = stimulus_psd_dir
        self.fs, self.nperseg, self.noverlap = fs, nperseg, noverlap
        self.audio_name: str | None = None
        self.entry: dict[str, Any] | None = None

    def get(self, audio_name: str) -> dict[str, Any]:
        if self.audio_name == audio_name:
            return self.entry

        self.audio_name, self.entry = None, None  # drop the old 1.31 GB before allocating the new
        data = np.load(self.audio_features_dir / f"{audio_name}.npz")
        subbands = data[SUBBAND_FEATURE_KEY]
        features = np.concatenate([subbands, data[GAMMATONE_FEATURE_KEY]], axis=0)

        with _FFT_LOCK:
            spectrum = windowed_fft(features, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap, axis=1)
            Xf_t = contiguous_conj_transpose(spectrum.Xf, DRAW_DTYPE)

        psd = np.load(self.stimulus_psd_dir / f"{audio_name}_stimulus_psd.npz")
        self.audio_name = audio_name
        self.entry = {
            "Xf_t": Xf_t,
            "n_frames": spectrum.n_frames,
            "scale": spectrum.scale,
            "hop": spectrum.hop,
            "Pxx": np.concatenate([psd["subband_psd"], psd["gammatone_psd"]], axis=1),
            "n_samples": features.shape[1],
            "n_subbands": subbands.shape[0],
        }
        return self.entry


def prepare_trial(
    audio_name: str,
    eeg_stem: str,
    reference: str,
    eeg_dir: Path,
    stimulus_cache: StimulusCache,
    fs: float,
    nperseg: int,
) -> dict[str, Any] | None:
    """Everything a trial's draws need, or ``None`` if the reference cannot be applied to it.

    This is the work draw-blocking repeats, so it is held to the irreducible set: the EEG's
    windowed FFT, the truncation to a common length, and the reciprocal coherence denominator.

    ``Pyy`` is computed here rather than read from get_coherence.py's per-trial ``eeg_psd``: that
    output exists only for the average reference, and the spectrum it needs is already in hand.
    """
    stim = stimulus_cache.get(audio_name)

    loaded = load_referenced_eeg(eeg_dir / f"{eeg_stem}.fif", reference)
    if loaded is None:
        return None  # no A1/A2 under linked mastoids: the caller journals it at zero weight
    eeg_data, eeg_fs, channel_names = loaded
    if eeg_fs != fs:
        raise ValueError(f"{eeg_stem}: EEG sample rate {eeg_fs} does not match stimulus rate {fs}")

    with _FFT_LOCK:
        y_full = windowed_fft(eeg_data, fs=fs, nperseg=nperseg, noverlap=stimulus_cache.noverlap, axis=1)
        # get_coherence.py truncates x and y to a common sample length before windowing; dropping
        # trailing windows is equivalent, since window k always covers [k*hop, k*hop+nperseg).
        n_frames = min(stim["n_frames"], y_full.n_frames)
        # Taken before the complex128 original is released: _auto_power wants the (n_frames, F, C)
        # spectrum, not the contiguous transpose that outlives it.
        Pyy = _auto_power(truncate_spectrum(y_full, n_frames))
        Yf_t = contiguous_for_matmul(y_full.Xf[:n_frames], DRAW_DTYPE)
    del y_full

    scale = stim["scale"] / stim["n_frames"] * n_frames
    if n_frames == stim["n_frames"]:
        Xf_t, Pxx = stim["Xf_t"], stim["Pxx"]
    else:
        # Rare: the EEG is shorter than the stimulus by more than one hop, so the cached PSD (which
        # covers the full-length envelope) is the wrong Pxx and has to be recomputed.
        Xf_t = np.ascontiguousarray(stim["Xf_t"][:, :, :n_frames])
        Pxx = auto_power_transposed(Xf_t, scale, nperseg)

    duration = eeg_data.shape[1] / eeg_fs

    # The duration weight is folded into the *reciprocal* denominator, computed in float64 and
    # rounded once to float32. That lets the hot loop apply both with one in-place complex64
    # multiply; a float64 denominator would promote each 79.99 MB draw to complex128 in buffered
    # chunks on every multiply. Script 3 divides in float64 and casts on assignment, so this adds
    # one rounding of ~6e-8 relative -- the same order as the complex64 roundoff both accept.
    inv_denom = (duration / np.sqrt(Pxx[:, :, None] * Pyy[:, None, :])).astype(np.float32)

    return {
        "Xf_t": Xf_t,
        "Yf_t": Yf_t,
        "inv_denom": inv_denom,
        "scale": scale,
        "hop": stim["hop"],
        "n_frames": n_frames,
        "n_subbands": stim["n_subbands"],
        "duration": duration,
        "analysed_duration": min(eeg_data.shape[1], stim["n_samples"]) / fs,
        "channel_names": channel_names,
    }


def _safe_prepare(prepare: Any, task: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """``prepare(task)`` with its exception captured rather than raised.

    Preparation runs on the prefetch thread, so a raised exception would surface from
    ``Future.result()`` when the driver's ``for`` advances -- *outside* the per-trial ``try`` that
    is supposed to keep one bad recording from killing a multi-hour run. Returning the error
    instead puts it back under the driver's control.
    """
    try:
        return prepare(task), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def iter_prepared_pairs(
    tasks: list[dict[str, Any]], prepare: Any, prefetch: bool
) -> Iterator[tuple[dict[str, Any], dict[str, Any] | None, str | None]]:
    """``(task, prep, error)`` per task, preparing trial *t+1* while *t* is being folded.

    The per-trial load+FFT is NFS-bound and largely single-threaded against a 24-thread matmul, so
    overlapping them costs one core and removes almost all of what draw-blocking would repeat.
    """
    if not prefetch or len(tasks) < 2:
        for task in tasks:
            prep, error = _safe_prepare(prepare, task)
            yield task, prep, error
        return

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(_safe_prepare, prepare, tasks[0])
        for index, task in enumerate(tasks):
            prep, error = pending.result()
            pending = pool.submit(_safe_prepare, prepare, tasks[index + 1]) if index + 1 < len(tasks) else None
            yield task, prep, error


# --------------------------------------------------------------------------------------------
# Per-channel accumulators: the working state is already the output shape
# --------------------------------------------------------------------------------------------


class ChannelAccumulators:
    """One ``(n_null, F, 1, 64)`` memmap per cochleogram channel, under ``<work_dir>/``.

    61 files of 655.6 MB. Because each is already the shape and dtype of its finished .npz, there
    is never a combined copy: finalize normalises one file, writes its .npz, and unlinks it, so
    peak disk is the 39.99 GB working set plus a single channel.

    That unlink is why the *output* path is part of this class's state. A channel is either an
    accumulator or an output, never both, so "the .npy is missing" is not the same question as
    "the .npy has to be created": a channel finalised by an earlier run has no accumulator left,
    and recreating it -- zeroed -- would send finalize back over a finished channel with an array
    of zeros, which is the right shape and dtype and entirely finite, so nothing downstream would
    catch it. ``create``, ``open`` and ``finalize_channels`` therefore all skip finalised channels.
    """

    def __init__(
        self,
        work_dir: Path,
        output_dir: Path,
        n_channels: int,
        n_subbands: int,
        n_null: int,
        n_freqs: int,
        n_eeg: int,
    ):
        self.work_dir = work_dir
        self.output_dir = output_dir
        self.n_channels = n_channels
        self.shape = (n_null, n_freqs, 1, n_eeg)
        self.paths = [work_dir / f"channel_{k:02d}.npy" for k in range(n_channels)]
        self.output_paths = [output_dir / channel_filename(k, n_subbands) for k in range(n_channels)]

    def finalized_channels(self) -> list[int]:
        """Channels whose .npz is already written, and whose accumulator is therefore gone."""
        return [k for k, path in enumerate(self.output_paths) if path.exists()]

    def create(self) -> None:
        """Allocate any missing accumulators, zeroed. Sparse on write, so this is instant."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        for path, output_path in zip(self.paths, self.output_paths):
            if path.exists() or output_path.exists():
                continue
            open_memmap(path, mode="w+", dtype=COHERENCE_DTYPE, shape=self.shape)

    def open(self) -> list[np.ndarray | None]:
        """One handle per channel, ``None`` where the channel has already been finalised."""
        handles: list[np.ndarray | None] = []
        for path, output_path in zip(self.paths, self.output_paths):
            if not path.exists() and output_path.exists():
                handles.append(None)
                continue
            handle = open_memmap(path, mode="r+")
            if handle.shape != self.shape or handle.dtype != np.dtype(COHERENCE_DTYPE):
                raise SystemExit(
                    f"{path} is {handle.shape} {handle.dtype} but this run needs "
                    f"{self.shape} {np.dtype(COHERENCE_DTYPE)}. Move it aside or pass --restart."
                )
            handles.append(handle)
        return handles

    def allocated_bytes(self) -> int:
        """Real blocks already consumed on disk, which a resumed run must not be charged for.

        A finalised channel counts as allocated: its accumulator is gone and will not be
        recreated, so those bytes are already on disk in their final form.
        """
        total = 0
        for npy_path, npz_path in zip(self.paths, self.output_paths):
            path = npy_path if npy_path.exists() else npz_path
            if path.exists():
                total += os.stat(path).st_blocks * 512
        return total


def block_ranges(n_null: int, draw_block: int) -> list[tuple[int, int]]:
    return [(start, min(start + draw_block, n_null)) for start in range(0, n_null, draw_block)]


def check_disk_space(work_dir: Path, output_dir: Path, accumulators: ChannelAccumulators, total_bytes: int) -> None:
    """Refuse to start a ~10 h run that cannot land its output.

    Because the accumulators *are* the output arrays, finalize swaps one .npy for one .npz of the
    same size at a time rather than doubling: the requirement is the 39.99 GB working set plus one
    channel of headroom, and whatever is already allocated by a resumed run is discounted.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    one_channel = total_bytes // max(1, accumulators.n_channels)
    still_needed = max(0, total_bytes - accumulators.allocated_bytes())
    same_fs = os.stat(work_dir).st_dev == os.stat(output_dir).st_dev

    needs = {work_dir: still_needed + one_channel} if same_fs else {work_dir: still_needed, output_dir: total_bytes}
    for path, required in needs.items():
        free = shutil.disk_usage(path).free
        if free < required:
            raise SystemExit(
                f"{path} has {free / 1e9:.1f} GB free but this run needs {required / 1e9:.1f} GB.\n"
                f"Point --output-dir at a larger volume (/data_storage), or settle Q7 and shrink "
                f"get_coherence.py's 297 GB first."
            )


# --------------------------------------------------------------------------------------------
# The draw loop
# --------------------------------------------------------------------------------------------


def fold_trial(
    accumulators: list[np.ndarray],
    prep: dict[str, Any],
    shifts: np.ndarray,
    start: int,
    stop: int,
    nperseg: int,
    threads: int,
) -> None:
    """Add one trial's duration-weighted draws ``[start, stop)`` into every channel accumulator.

    Threads partition the draw range, so each thread writes a disjoint set of rows in every one of
    the 61 files and no lock is needed anywhere -- the reason the channel axis never has to be
    split for parallelism. Each thread holds one 79.99 MB cross-spectrum at a time and the
    ``*=``/``+=`` are in place, so nothing else is allocated inside the loop.

    The per-channel scatter reads ``cross[:, k, :]`` -- strided, but in 512-byte contiguous runs,
    so cache lines are fully used; see the note in the module docstring.
    """
    Xf_t, Yf_t, inv_denom, scale = prep["Xf_t"], prep["Yf_t"], prep["inv_denom"], prep["scale"]

    def run(draw_indices: range) -> None:
        for draw_idx in draw_indices:
            cross = shifted_cross_spectrum(Xf_t, Yf_t, int(shifts[draw_idx]), scale, nperseg)
            cross *= inv_denom  # applies 1/denom and the duration weight, in place, complex64
            for channel, accumulator in enumerate(accumulators):
                accumulator[draw_idx, :, 0, :] += cross[:, channel, :]

    if threads <= 1:
        run(range(start, stop))
        return

    bounds = np.linspace(start, stop, threads + 1).astype(int)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(lambda i: run(range(bounds[i], bounds[i + 1])), range(threads)))


# --------------------------------------------------------------------------------------------
# Journal
# --------------------------------------------------------------------------------------------


class BlockJournal:
    """Which trials are folded into one draw block, plus a marker for one mid-fold.

    A block is finished simply by listing every trial -- the arrays are slices of the per-channel
    accumulators, so there is nothing to concatenate at the end.

    ``+=`` is not idempotent, so the pending marker is written before a fold and cleared after the
    journal is fsynced. A marker surviving startup means the block may hold a partial or duplicated
    contribution, and that block -- one pass over the trials, not the run -- is zeroed and redone.
    """

    def __init__(self, work_dir: Path, block_idx: int):
        self.journal_path = work_dir / f"block_{block_idx:03d}.journal"
        self.pending_path = work_dir / f"block_{block_idx:03d}.pending"

    def exists(self) -> bool:
        return self.journal_path.exists()

    def create(self) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.journal_path.write_text("")

    def reset(self) -> None:
        self.journal_path.unlink(missing_ok=True)
        self.pending_path.unlink(missing_ok=True)

    def entries(self) -> list[tuple[str, float, float]]:
        if not self.journal_path.exists():
            return []
        rows = []
        for line in self.journal_path.read_text().splitlines():
            if not line.strip():
                continue
            stem, duration, analysed = line.split("\t")
            rows.append((stem, float(duration), float(analysed)))
        return rows

    def folded_stems(self) -> set[str]:
        return {stem for stem, _, _ in self.entries()}

    def mark_pending(self, stem: str) -> None:
        self.pending_path.write_text(stem)
        fsync_path(self.pending_path)

    def commit(self, stem: str, duration: float, analysed_duration: float) -> None:
        with self.journal_path.open("a") as handle:
            handle.write(f"{stem}\t{duration!r}\t{analysed_duration!r}\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.pending_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------------------------
# Finalize
# --------------------------------------------------------------------------------------------


def channel_filename(channel_idx: int, n_subbands: int) -> str:
    return "gammatone_null.npz" if channel_idx == n_subbands else f"subband_{channel_idx:02d}_null.npz"


def fold_real(real_handle: np.ndarray, prep: dict[str, Any], nperseg: int) -> None:
    """Add one trial's duration-weighted real coherence to the running sum.

    The real estimate is the shift-0 case of the draw kernel, so it reuses both operands and
    costs one matmul out of ``n_null + 1``. ``inv_denom`` already carries the duration weight,
    which is what makes this a weighted sum rather than a plain one.

    Folded during the FIRST draw block only: every block walks all 563 trials, so doing it in
    each would count every trial once per block. Block 0's journal is therefore also the real
    sum's journal -- a resumed run skips the trials already in it, whose contribution is already
    in the memmap, and a block-0 reset zeroes this alongside the draws.
    """
    cross = shifted_cross_spectrum(prep["Xf_t"], prep["Yf_t"], 0, prep["scale"], nperseg)
    cross *= prep["inv_denom"]
    real_handle += cross


def finalize_channels(
    accumulators: ChannelAccumulators,
    real_sum: np.ndarray,
    output_dir: Path,
    freqs: np.ndarray,
    centre_freqs: np.ndarray,
    n_subbands: int,
    total_duration: float,
    total_analysed: float,
    stems: list[str],
    durations: list[float],
    n_null: int,
    reference: str,
    channel_names: list[str],
    skipped_stems: list[str],
) -> int:
    """Normalise each channel accumulator by total duration and write its .npz.

    One channel at a time, unlinking each .npy once its .npz is in place, so the transient this
    needs is one 655.6 MB channel rather than a second copy of all 39.99 GB.

    This stage is the one that is not journalled, and it does not need to be: a channel's .npz is
    renamed into place before its .npy is unlinked, so the pair of files *is* the record of what
    is done. A rerun after a crash here resumes by skipping every channel that already has one --
    the alternative, rewriting it from the zeroed accumulator ``create`` would otherwise have put
    back, is silent corruption rather than a lost hour.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    already = accumulators.finalized_channels()
    if already:
        print(f"{len(already)} channel(s) already finalised by an earlier run; writing the other {accumulators.n_channels - len(already)}")

    for channel, path in enumerate(tqdm(accumulators.paths, desc="finalize", unit="channel")):
        # Written to a temporary name and renamed, so a half-written .npz can never be mistaken
        # for a finished channel. np.savez is handed an open file rather than a path because it
        # appends ".npz" to any path lacking the suffix.
        output_path = accumulators.output_paths[channel]
        if output_path.exists() and not path.exists():
            continue

        is_gammatone = channel == n_subbands
        null_coherence = np.array(open_memmap(path, mode="r")) / total_duration
        tmp_path = output_path.with_name(output_path.name + ".partial")
        with tmp_path.open("wb") as handle:
            np.savez(
                handle,
                null_coherence=null_coherence.astype(COHERENCE_DTYPE),
                coherence=(real_sum[:, channel : channel + 1, :] / total_duration).astype(COHERENCE_DTYPE),
                # Two different axes: 2561 coherence bins, one gammatone filter centre.
                frequencies=freqs,
                reference=reference,
                # The channel axis is 63 wide under FCz and 64 otherwise, so a topography cannot
                # be plotted from these without knowing which channels it holds.
                channel_names=np.array(channel_names),
                # Recordings this reference could not be applied to; empty but for linked mastoids.
                skipped_stems=np.array(skipped_stems),
                centre_frequency=np.float64(np.nan if is_gammatone else centre_freqs[channel]),
                channel_index=channel,
                channel_kind="gammatone" if is_gammatone else "subband",
                n_null=n_null,
                n_trials=len(stems),
                total_duration=total_duration,
                total_analysed_duration=total_analysed,
                trial_stems=np.array(stems),
                trial_durations=np.array(durations),
            )
        tmp_path.replace(output_path)
        del null_coherence
        path.unlink()
        written += 1

    return written


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


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
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help=f"draw threads (default {DEFAULT_THREADS})")
    parser.add_argument("--draw-block", type=int, default=DRAW_BLOCK, help=f"draws resident per pass (default {DRAW_BLOCK})")
    parser.add_argument("--output-dir", type=Path, default=None, help="where the per-channel files go (default under null_coherence_dir)")
    parser.add_argument("--max-trials", type=int, default=None, help="benchmark on the first N trials only")
    parser.add_argument("--no-prefetch", action="store_true", help="disable the one-deep trial prefetch")
    parser.add_argument("--restart", action="store_true", help="discard any resumable working state and start clean")
    args = parser.parse_args(argv)

    # Each draw thread calls a single-threaded matmul; nested BLAS threads would oversubscribe 24
    # cores several times over. The controller has to stay referenced for the limit to hold -- if
    # it is garbage collected the previous limits are restored -- hence binding it to a local that
    # outlives the draw loop.
    blas_limit = None
    try:
        from threadpoolctl import threadpool_limits

        blas_limit = threadpool_limits(limits=1)
    except ImportError:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")

    config = load_config(ROOT / "src" / "package_data" / "config.json")
    audio_features_dir, eeg_dir, _, stimulus_psd_dir = resolve_configured_inputs(config, ROOT)
    if not eeg_dir.exists():
        raise FileNotFoundError(f"{eeg_dir} does not exist; it is the input this script pools over")

    null_dir = resolve_reference_dirs(config, args.reference, ROOT)[1]
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else null_dir / "cochleogram_null"
    work_dir = null_dir / "cochleogram_null_work"
    if args.restart and work_dir.exists():
        shutil.rmtree(work_dir)

    # Derived from the recordings themselves: get_coherence.py's per-trial output exists only for
    # the average reference. Trials a reference cannot be applied to are discovered by the worker
    # that loads them and journalled at zero weight, not scanned for up front -- reading 563 FIF
    # headers over NFS measured 18 minutes, and a resumed run would pay it again.
    df = build_trial_index(eeg_dir)
    if args.max_trials is not None:
        df = df.head(args.max_trials).reset_index(drop=True)
    # Stimulus order, so the 1.31 GB StimulusCache actually hits across a pass.
    df = df.sort_values(["audio_name", "eeg_stem"]).reset_index(drop=True)

    fs = STIMULUS_FS
    nperseg, noverlap = stimulus_window_sizes(fs)
    freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)

    # Shapes follow from fs/nperseg, the feature file's channel count and the channel axis this
    # reference produces -- no per-trial coherence needed.
    channel_names = reference_channel_names(eeg_dir / f"{df.iloc[0]['eeg_stem']}.fif", args.reference)
    n_freqs, n_eeg = len(freqs), len(channel_names)
    n_subbands = int(np.load(audio_features_dir / f"{df.iloc[0]['audio_name']}.npz")[SUBBAND_FEATURE_KEY].shape[0])
    n_channels = n_subbands + 1

    # Read straight from a feature file rather than picking it up inside the trial loop: a resumed
    # run whose blocks are all complete skips that loop entirely and would otherwise reach finalize
    # with nothing -- the one path where the failure costs a whole run's compute. The 60 ERB-spaced
    # centres are identical across stimuli, so any one file settles them.
    centre_freqs = load_centre_frequencies(audio_features_dir / f"{df.iloc[0]['audio_name']}.npz")
    if centre_freqs.shape != (n_subbands,):
        raise SystemExit(f"expected {n_subbands} gammatone centre frequencies, found {centre_freqs.shape}")

    accumulators = ChannelAccumulators(
        work_dir, output_dir, n_channels, n_subbands, args.n_null, n_freqs, n_eeg
    )
    total_bytes = n_channels * args.n_null * n_freqs * n_eeg * np.dtype(COHERENCE_DTYPE).itemsize
    check_disk_space(work_dir, output_dir, accumulators, total_bytes)
    accumulators.create()
    handles = accumulators.open()

    blocks = block_ranges(args.n_null, args.draw_block)
    all_stems = set(df["eeg_stem"])

    # A finalised channel has no accumulator to fold into, and unlike a missing block it cannot be
    # rebuilt -- the draws that made it are gone. That only matters if there is folding left to do,
    # which the journals decide; refusing here is what lets the loop below assume all 61 handles
    # are live. Reached only if a run died inside finalize and then lost its journals or its .npy
    # files, since finalize does not start until every block is complete.
    finalized = accumulators.finalized_channels()
    if finalized and any(
        BlockJournal(work_dir, i).pending_path.exists() or BlockJournal(work_dir, i).folded_stems() != all_stems
        for i in range(len(blocks))
    ):
        raise SystemExit(
            f"{len(finalized)} channel(s) in {output_dir} are already finalised, but the draw "
            f"blocks in {work_dir} are incomplete and the finalised channels' accumulators cannot "
            f"be rebuilt. Move those .npz files aside to recompute them, or pass --restart to "
            f"redo the whole run."
        )

    print(
        f"{len(df)} trials, n_null={args.n_null}, {len(blocks)} block(s) of <={args.draw_block} draws\n"
        f"threads={args.threads}, block resident ~{args.draw_block * total_bytes / args.n_null / 1e9:.1f} GB, "
        f"{n_channels} channel accumulators ~{total_bytes / 1e9:.1f} GB in {work_dir}\n"
        f"output: {n_channels} files of ~{total_bytes / n_channels / 1e6:.0f} MB in {output_dir}"
    )

    stimulus_cache = StimulusCache(audio_features_dir, stimulus_psd_dir, fs, nperseg, noverlap)
    failures: list[tuple[str, str]] = []

    def prepare(task: dict[str, Any]) -> dict[str, Any]:
        return prepare_trial(
            task["audio_name"],
            task["eeg_stem"],
            args.reference,
            eeg_dir,
            stimulus_cache,
            fs,
            nperseg,
        )

    # The real coherence accumulates here during block 0 (see fold_real). Memmapped like the
    # draws, so a resumed run picks up the contributions already folded in.
    real_path = work_dir / "real_sum.npy"
    real_handle = (open_memmap(real_path, mode="r+") if real_path.exists() else
                   open_memmap(real_path, mode="w+", dtype=np.complex128,
                               shape=(n_freqs, n_channels, n_eeg)))

    for block_idx, (start, stop) in enumerate(blocks):
        journal = BlockJournal(work_dir, block_idx)
        if journal.pending_path.exists():
            print(f"block {block_idx}: mid-fold when the previous run stopped; zeroing and recomputing")
            for handle in handles:
                handle[start:stop] = 0
                handle.flush()
            if block_idx == 0:  # the real sum is folded in this block, so it resets with it
                real_handle[:] = 0
                real_handle.flush()
            journal.reset()
        if not journal.exists():
            journal.create()
        done = journal.folded_stems()
        if done == all_stems:
            continue

        tasks = [
            {"eeg_stem": row["eeg_stem"], "audio_name": row["audio_name"]}
            for _, row in df.iterrows()
            if row["eeg_stem"] not in done
        ]

        progress = tqdm(total=len(tasks), desc=f"draws {start}-{stop}", unit="trial")
        for task, prep, error in iter_prepared_pairs(tasks, prepare, not args.no_prefetch):
            try:
                if error is not None:  # raised on the prefetch thread; re-surfaced here
                    raise RuntimeError(error)
                if prep is None:
                    # The reference does not apply to this recording. Journalled at zero weight so
                    # the block can complete, contributing to no accumulator and no denominator.
                    journal.commit(task["eeg_stem"], 0.0, 0.0)
                    progress.update(1)
                    continue
                shifts = null_shifts(prep["n_frames"], fs, prep["hop"], args.n_null, prep["analysed_duration"])
                if prep["channel_names"] != channel_names:
                    raise ValueError(f"{task['eeg_stem']}: channel layout differs from the first recording")
                journal.mark_pending(task["eeg_stem"])
                fold_trial(handles, prep, shifts, start, stop, nperseg, args.threads)
                if block_idx == 0:
                    fold_real(real_handle, prep, nperseg)
                journal.commit(task["eeg_stem"], prep["duration"], prep["analysed_duration"])
            except Exception as exc:  # keep one bad trial from killing a multi-hour run
                failures.append((task["eeg_stem"], f"{type(exc).__name__}: {exc}"))
                tqdm.write(f"FAILED {task['eeg_stem']}: {type(exc).__name__}: {exc}")
            progress.update(1)
        progress.close()
        for handle in handles:
            handle.flush()
        if block_idx == 0:
            real_handle.flush()

    if failures:
        print(f"{len(failures)} trial(s) failed; nothing finalised. Re-run to retry.")
        return 1

    incomplete = [i for i in range(len(blocks)) if BlockJournal(work_dir, i).folded_stems() != all_stems]
    if incomplete:
        print(f"{len(incomplete)} of {len(blocks)} block(s) incomplete; working state kept for a resumed run")
        return 1

    # Zero-duration entries are recordings this reference could not be applied to: listed in the
    # output, but out of the trial set, the weights and the denominator.
    all_entries = BlockJournal(work_dir, 0).entries()
    skipped_stems = [stem for stem, duration, _ in all_entries if duration == 0.0]
    entries = [entry for entry in all_entries if entry[1] != 0.0]
    if not entries:
        raise SystemExit(f"no recording could be given the {args.reference} reference")
    total_duration = sum(duration for _, duration, _ in entries)
    total_analysed = sum(analysed for _, _, analysed in entries)

    real_sum = np.array(real_handle)  # duration-weighted already: inv_denom carries the weight
    del handles, real_handle  # release the memmaps before finalize reopens and unlinks them
    written = finalize_channels(
        accumulators,
        real_sum,
        output_dir,
        freqs,
        centre_freqs,
        n_subbands,
        total_duration,
        total_analysed,
        [stem for stem, _, _ in entries],
        [duration for _, duration, _ in entries],
        args.n_null,
        args.reference,
        channel_names,
        skipped_stems,
    )

    shutil.rmtree(work_dir, ignore_errors=True)
    if blas_limit is not None:
        blas_limit.unregister()  # also what keeps the controller referenced across the draw loop
    print(f"wrote {written} per-channel file(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
