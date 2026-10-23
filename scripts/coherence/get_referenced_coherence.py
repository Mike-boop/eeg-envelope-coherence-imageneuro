#!/usr/bin/env python3
"""Grand-average broadband and subband coherence under an alternative EEG reference, for the supplement.

The main pipeline (``get_coherence.py`` and the two ``get_null_*`` scripts) is average-referenced
throughout. This script answers the reviewer-facing question "does the result survive a different
reference?" for the two alternatives the original ``neural_tracking_across_spectrum`` scripts
supported -- linked mastoids and FCz -- **without** reproducing the pipeline underneath them.

    python scripts/coherence/get_referenced_coherence.py
    python scripts/coherence/get_referenced_coherence.py --reference avg --max-trials 8

Two things are deliberately not here, and both are what keep it cheap:

1. **No per-trial output.** Each trial's coherence is folded into a duration-weighted accumulator
   and dropped. ``get_coherence.py`` writes 161 MB per recording (99.5 GB across the dataset)
   because the per-trial coherence feeds everything downstream; a supplementary figure needs one
   grand average, so the output here is **one ~160 MB .npz per reference**.
2. **One null draw.** The null is a single pooled sample, not a distribution (see below), so the
   ~40 GB of a 500-draw subband null is not repeated per reference.

Both the broadband envelope and the 60 cochleogram subbands are used, stacked into one ``Cx = 61``
stimulus exactly as ``get_null_cochleogram_coherence.py`` stacks them -- subbands first, broadband
last -- so one matmul yields both, and the broadband output is the same computation it was when
this script used the broadband envelope alone. (Until 2026-09-16 it did: ``Cx = 1``, ~4 MB per
reference. The broadband keys and shapes are unchanged, so readers of those files are unaffected.)

There is consequently nothing on disk to resume from, so there is no resume path: a re-run is a
re-run. Each output is still written under a temporary name and renamed, so an interrupted run
never replaces a complete file from an earlier one.

Output
------
One file per reference, in a subdirectory of the configured ``coherence_dir`` so the alternative
references never mix with the average-referenced per-trial files in that directory:

    <coherence_dir>/reference_average/average_coherence_<reference>.npz

This is the same separation the original scripts got from their sibling ``coherences_FCz/``
directories, collapsed to one file because one file is all that is produced.

The broadband keys are ``gammatone_coherence`` ``(F, 1, Cy)``, ``null_gammatone_coherence``
``(n_null, F, 1, Cy)``, ``frequencies`` and ``total_duration`` -- deliberately the names
``get_null_gammatone_coherence.py`` writes, so the loading loop in
``analysis/average_coherence.ipynb`` reads these files unchanged. The subband keys follow
``get_coherence.py``'s names: ``subband_coherence`` ``(F, 60, Cy)`` and ``null_subband_coherence``
``(n_null, F, 60, Cy)``, with ``centre_frequencies`` ``(60,)`` giving each subband's gammatone
filter centre (a different axis from ``frequencies``, the 2561 coherence bins).

A file from before the subbands were added counts as *not done*: it is recomputed rather than
skipped, and replaced only once the new average is complete.

``channel_names`` is saved because it has to be: the EEG channel axis is no longer the 64 channels
of the raw files for every reference (FCz is dropped after being referenced to, leaving 63), so a
topography cannot be plotted from these without knowing which channels the axis holds. It is
checked against every trial as they are folded in, so a layout change mid-dataset is an error
rather than a silently mis-labelled average.

The averaging convention is the pipeline's, not a new one
---------------------------------------------------------
``coherence = Σ_trials duration · Cxy / Σ_trials duration``, complex, exactly as
``get_null_gammatone_coherence.py`` finalizes its groups and as ``analysis/average_coherence.ipynb``
combines them -- the weight is the EEG recording's own duration, and the average is taken over the
*complex* coherency, with ``|·|²`` left to the plotting code. The trial set is the same one the
pooled scripts use (``load_trial_index``, which drops the sin/av conditions and the
shifted/artefact stimuli), so the supplementary average is comparable to the main one trial for
trial rather than merely similar.

The null is one global-shift draw per trial, averaged the same way, with shifts from
``pipeline.null_shifts`` -- the per-recording seeded sequence, so a trial's draw here is the same
shift it gets in every other script. It is a single pooled null *sample*, not a distribution: it
says where zero sits on the same axis as the real average, which is what a supplementary
"the alternative reference looks like this" panel needs. Percentile statements still require the
~500-draw runs of ``get_null_gammatone_coherence.py``, which are deliberately not repeated here.

References
----------
``A1``/``A2`` are ``misc``-typed in these files, so they are outside ``pick("eeg")`` and cannot
leak into an average reference or into the output's channel axis. Verified 2026-09-15 that MNE's
``set_eeg_reference`` still subtracts them from the EEG channels when they are ``misc``
(bit-exact against a manual subtraction), which is what makes the linked-mastoid case work at all:

    "avg"             average of the 64 scalp electrodes                     -> 64 channels
    "linked_mastoids" mean of A1/A2; recordings missing either are skipped   -> 64 channels
    "FCz"             the FCz electrode, which is then dropped as all-zero   -> 63 channels

**21 of the 617 recordings have no A1/A2** (measured 2026-09-15; every file has FCz, and the EEG
channel layout is otherwise identical across all 617). Those trials are skipped for
``linked_mastoids`` only -- their duration never enters the denominator -- so that average is over
a slightly smaller trial set than the others. The count and the stems are printed and saved, and
the difference is worth stating wherever the panels appear side by side.

``avg`` is available so the comparison can be made against a grand average produced by *this* code
rather than by the main pipeline -- same trials, weights and windowing, differing only in the
reference. It is also the script's validation: it reproduces a duration-weighted average of
``get_coherence.py``'s own per-trial files -- broadband *and* subband, real and null draw 0 -- to
complex64 roundoff.

Cost
----
Per trial: one EEG load, one 64-channel FFT, and ``1 + n_null`` matmuls at ``Cx = 61``. Per
stimulus, per worker: one 61-channel FFT, cached for that stimulus's following trials.

The operands are materialised C-contiguous at complex64 (``DRAW_DTYPE``), the working precision of
the pooled scripts: at ``Cx = 61`` the stimulus spectrum is 2.6 GB in complex128 and 1.3 GB in
complex64, and memory per worker is what limits the pool. Accumulators stay complex128, since a
sum over 563 trials in complex64 would lose precision the stored output cannot get back.

Every numeric library is pinned to one thread before numpy is imported, and the parallelism is
across trials. Trials are dispatched individually, in stimulus order, so the per-worker stimulus
cache still hits under ordered dispatch and a failure costs one trial rather than a stimulus.
"""

from __future__ import annotations

import argparse
import os

# Parallelism is across trials, and an unpinned pool oversubscribes the box several times over, so
# pin every numeric library to one thread before numpy is imported anywhere below.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import multiprocessing as mp
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mne
import numpy as np
from tqdm import tqdm

from coherence.coherence import (
    _auto_power,
    auto_power_transposed,
    contiguous_for_matmul,
    shifted_cross_spectrum,
    truncate_spectrum,
    windowed_fft,
)
from coherence.pipeline import (
    COHERENCE_DTYPE,
    DRAW_DTYPE,
    STIMULUS_FS,
    load_centre_frequencies,
    load_stimulus_features,
    load_trial_index,
    null_shifts,
    resolve_configured_inputs,
    stimulus_window_sizes,
)
from transcription.config import load_config

mne.set_log_level("WARNING")

REFERENCES = ("avg", "linked_mastoids", "FCz")
DEFAULT_REFERENCES = ("avg", "linked_mastoids", "FCz")

MASTOID_CHANNELS = ("A1", "A2")
FCZ_CHANNEL = "FCz"

# Accumulators are complex128: a grand average over 563 trials summed in complex64 would lose
# precision the stored complex64 output cannot get back. At Cx = 61 that is 0.32 GB per copy, held
# only in the parent process.
ACCUMULATOR_DTYPE = np.complex128

# The subband outputs, whose presence marks a file as written by this version of the script.
SUBBAND_KEYS = ("subband_coherence", "null_subband_coherence", "centre_frequencies")

_WORKER: dict[str, Any] = {}


# ---------------------------------------------------------------------------------------------
# Referencing
# ---------------------------------------------------------------------------------------------


def load_referenced_eeg(fif_path: Path, reference: str) -> tuple[np.ndarray, float, list[str]] | None:
    """EEG data, sample rate and channel names under ``reference``; ``None`` if it cannot be applied.

    ``None`` means "skip this recording", which only the linked-mastoid case can return -- a
    recording without both mastoids has no linked-mastoid reference to apply, and the original
    scripts skipped it rather than falling back to something else. Skipped trials are excluded
    from the weighted average entirely (their duration never enters the denominator) and are
    listed in the output.
    """
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)

    if reference == "avg":
        # pick first: A1/A2 are misc-typed and so are already outside the average, but picking
        # makes that explicit rather than dependent on the channel types in the file.
        raw.pick("eeg")
        raw.set_eeg_reference("average", verbose=False)
    elif reference == "linked_mastoids":
        if not set(MASTOID_CHANNELS) <= set(raw.ch_names):
            return None
        raw.set_eeg_reference(list(MASTOID_CHANNELS), verbose=False)
        raw.pick("eeg")  # drops the mastoids; they are misc, and are not wanted in the output
    elif reference == "FCz":
        raw.set_eeg_reference([FCZ_CHANNEL], verbose=False)
        raw.pick("eeg")
        raw.drop_channels([FCZ_CHANNEL])  # identically zero after referencing to itself
    else:
        raise ValueError(f"Unknown reference {reference!r}, expected one of {sorted(REFERENCES)}")

    return raw.get_data(), float(raw.info["sfreq"]), list(raw.ch_names)


# ---------------------------------------------------------------------------------------------
# Per-trial coherence
# ---------------------------------------------------------------------------------------------


@dataclass
class CachedStimulus:
    """One stimulus's 61-channel spectrum -- 60 subbands, then the broadband envelope -- per worker.

    ~1.3 GB at ``Cx = 61`` in complex64, and a 61-channel FFT, so this is the load-bearing cache it
    is in ``get_coherence.py``: the median stimulus has 6 trials, and ordered dispatch hands a
    worker runs of the same stimulus. Only the contiguous operand is kept, not the complex128
    ``WindowedSpectrum`` it came from, which would double the worker's resident memory.
    """

    audio_name: str
    Xf_t: np.ndarray  # (F, 61, n_frames) conjugated, C-contiguous, DRAW_DTYPE
    Pxx: np.ndarray  # (F, 61) float64, over the full length
    freqs: np.ndarray
    n_frames: int
    hop: int
    scale: float
    n_samples: int


class StimulusCache:
    def __init__(self, audio_features_dir: Path, fs: float, nperseg: int, noverlap: int):
        self.audio_features_dir = audio_features_dir
        self.fs, self.nperseg, self.noverlap = fs, nperseg, noverlap
        self.cached: CachedStimulus | None = None

    def get(self, audio_name: str) -> CachedStimulus:
        if self.cached is not None and self.cached.audio_name == audio_name:
            return self.cached
        self.cached = None  # drop the previous stimulus's 1.3 GB before allocating the next

        features = load_stimulus_features(self.audio_features_dir / f"{audio_name}.npz")  # broadband last
        spectrum = windowed_fft(features, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap, axis=1)
        Pxx = _auto_power(spectrum)
        # Conjugated in place: the spectrum is this function's own and is dropped below, and a
        # conjugated copy would be a second 2.6 GB alongside it at the worker's peak.
        np.conj(spectrum.Xf, out=spectrum.Xf)
        self.cached = CachedStimulus(
            audio_name=audio_name,
            Xf_t=spectrum.Xf.transpose(1, 2, 0).astype(DRAW_DTYPE, order="C"),
            Pxx=Pxx,
            freqs=spectrum.freqs,
            n_frames=spectrum.n_frames,
            hop=spectrum.hop,
            scale=spectrum.scale,
            n_samples=features.shape[1],
        )
        return self.cached


def compute_trial(
    eeg_path: Path,
    stimulus: CachedStimulus,
    reference: str,
    fs: float,
    nperseg: int,
    noverlap: int,
    n_null: int,
) -> dict[str, Any] | None:
    """One trial's real and null coherence, ``(F, 61, Cy)`` with broadband last, or ``None`` if the
    reference cannot apply. Returned in ``COHERENCE_DTYPE``: the matmul runs at complex64, so a
    complex128 copy would carry no extra precision and double the 0.16 GB sent back per trial."""
    loaded = load_referenced_eeg(eeg_path, reference)
    if loaded is None:
        return None
    eeg_data, eeg_fs, channel_names = loaded
    if eeg_fs != fs:
        raise ValueError(f"{eeg_path.name}: EEG sample rate {eeg_fs} does not match stimulus rate {fs}")

    y_full = windowed_fft(eeg_data, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1)

    # Reproduce the common-length truncation by dropping trailing windows from whichever side has
    # more; the surviving windows are bit-identical either way. (Measured on 60 sampled trials:
    # the EEG is the shorter side in none of them, so the stimulus branch below is near-dead.)
    n_frames = min(stimulus.n_frames, y_full.n_frames)
    y_spectrum = truncate_spectrum(y_full, n_frames)
    Pyy = _auto_power(y_spectrum)
    # Materialised once and reused by every matmul: the real estimate is the shift-0 case of the
    # same kernel, so there is no second operand layout to prepare.
    Yf_t = contiguous_for_matmul(y_spectrum.Xf, DRAW_DTYPE)
    del y_full, y_spectrum

    if n_frames == stimulus.n_frames:
        Xf_t, Pxx, scale = stimulus.Xf_t, stimulus.Pxx, stimulus.scale
    else:
        # Dropping trailing windows only rescales `scale` (see truncate_spectrum); Pxx has to be
        # recomputed, since the cached one covers windows this trial does not have.
        scale = stimulus.scale / stimulus.n_frames * n_frames
        Xf_t = np.ascontiguousarray(stimulus.Xf_t[:, :, :n_frames])
        Pxx = auto_power_transposed(Xf_t, scale, nperseg)
    denom = np.sqrt(Pxx[:, :, None] * Pyy[:, None, :])

    # Three durations, as in get_coherence.py: `duration` is the EEG recording's own length and is
    # the weight used here; `analysed_duration` is the overlap actually entering the estimate, and
    # is what the null bounds are derived from.
    duration = eeg_data.shape[1] / eeg_fs
    analysed_duration = min(eeg_data.shape[1], stimulus.n_samples) / fs

    real = (shifted_cross_spectrum(Xf_t, Yf_t, 0, scale, nperseg) / denom).astype(COHERENCE_DTYPE)
    null = np.empty((n_null, *denom.shape), dtype=COHERENCE_DTYPE)
    for draw, shift in enumerate(null_shifts(n_frames, fs, stimulus.hop, n_null, analysed_duration)):
        null[draw] = shifted_cross_spectrum(Xf_t, Yf_t, int(shift), scale, nperseg) / denom

    return {
        "real": real,
        "null": null,
        "psd": Pyy,
        "duration": duration,
        "analysed_duration": analysed_duration,
        "channel_names": channel_names,
        "frequencies": stimulus.freqs,
    }


# ---------------------------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------------------------


@dataclass
class Accumulator:
    """Duration-weighted sums over trials, allocated on the first trial folded in.

    Lazily shaped because the EEG channel count depends on the reference (63 under FCz, 64
    otherwise) and is not worth hard-coding; the first trial settles it, and every later trial is
    checked against it via ``channel_names``.
    """

    n_null: int
    real: np.ndarray | None = None  # (F, 61, Cy), broadband last
    null: np.ndarray | None = None  # (n_null, F, 61, Cy)
    psd: np.ndarray | None = None  # (F, Cy) float64
    frequencies: np.ndarray | None = None
    total_duration: float = 0.0
    total_analysed_duration: float = 0.0
    channel_names: list[str] | None = None
    stems: list[str] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def fold(self, trial: dict[str, Any], stem: str) -> None:
        if self.real is None:
            self.real = np.zeros(trial["real"].shape, dtype=ACCUMULATOR_DTYPE)
            self.null = np.zeros((self.n_null, *trial["real"].shape), dtype=ACCUMULATOR_DTYPE)
            self.psd = np.zeros(trial["psd"].shape, dtype=np.float64)
            self.channel_names = trial["channel_names"]
            self.frequencies = trial["frequencies"]
        elif self.channel_names != trial["channel_names"]:
            raise ValueError(f"{stem}: channel layout differs from the preceding recordings")

        duration = trial["duration"]
        # Widened before weighting: `duration * complex64` would round each term to complex64
        # before it reaches the complex128 sum.
        self.real += duration * trial["real"].astype(ACCUMULATOR_DTYPE)
        self.null += duration * trial["null"].astype(ACCUMULATOR_DTYPE)
        self.psd += duration * trial["psd"]
        self.total_duration += duration
        self.total_analysed_duration += trial["analysed_duration"]
        self.stems.append(stem)
        self.durations.append(duration)


# ---------------------------------------------------------------------------------------------
# Worker pool -- one task per trial
# ---------------------------------------------------------------------------------------------


def _init_worker(context: dict[str, Any]) -> None:
    _WORKER.update(context)
    _WORKER["stimulus_cache"] = StimulusCache(
        context["audio_features_dir"], context["fs"], context["nperseg"], context["noverlap"]
    )


def _process_trial(task: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
    try:
        stimulus = _WORKER["stimulus_cache"].get(task["audio_name"])
        trial = compute_trial(
            _WORKER["eeg_dir"] / f"{task['eeg_stem']}.fif",
            stimulus,
            _WORKER["reference"],
            _WORKER["fs"],
            _WORKER["nperseg"],
            _WORKER["noverlap"],
            _WORKER["n_null"],
        )
        return task["eeg_stem"], trial, None
    except Exception as exc:  # one bad recording must not kill the run
        return task["eeg_stem"], None, f"{type(exc).__name__}: {exc}"


def run_tasks(
    tasks: list[dict[str, Any]], context: dict[str, Any], n_workers: int
) -> tuple[Accumulator, list[tuple[str, str]]]:
    """Fold every trial into one accumulator, in whatever order the workers finish.

    Failures are returned rather than swallowed. Unlike ``get_coherence.py``, where a missing
    output file is visibly missing and the run is resumable, a lost trial here would be invisible:
    the average would simply be normalised by a smaller total duration and look perfectly
    well-formed. The caller refuses to write an output file when this list is non-empty.
    """
    total = Accumulator(n_null=context["n_null"])
    failures: list[tuple[str, str]] = []

    def collect(result: tuple[str, dict[str, Any] | None, str | None]) -> None:
        stem, trial, error = result
        if error is not None:
            failures.append((stem, error))
            tqdm.write(f"FAILED {stem}: {error}")
        elif trial is None:
            total.skipped.append(stem)
        else:
            total.fold(trial, stem)

    desc = f"{context['reference']} trials"
    if n_workers <= 1:
        _init_worker(context)
        for task in tqdm(tasks, desc=desc, unit="trial"):
            collect(_process_trial(task))
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_workers, initializer=_init_worker, initargs=(context,)) as pool:
            # Ordered: consecutive tasks share a stimulus (they are sorted by audio_name), so an
            # ordered dispatch is what lets the per-worker StimulusCache hit at all.
            for result in tqdm(pool.imap(_process_trial, tasks), total=len(tasks), desc=desc, unit="trial"):
                collect(result)
    return total, failures


# ---------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------


def has_subbands(path: Path) -> bool:
    """Whether an existing output was written with the subband arrays, i.e. by this version."""
    try:
        with zipfile.ZipFile(path) as archive:
            members = set(archive.namelist())
    except zipfile.BadZipFile:
        return False
    return all(f"{key}.npy" in members for key in SUBBAND_KEYS)


def save_average(accumulator: Accumulator, reference: str, output_path: Path, centre_frequencies: np.ndarray) -> None:
    """Normalise the weighted sums by the total duration and write the one output file.

    Written under a temporary name and renamed, so an interrupted write can never replace a
    complete file from an earlier run with a truncated one. ``np.savez`` is handed an open file
    because it appends ".npz" to any path lacking the suffix.
    """
    if accumulator.real is None:
        raise RuntimeError(f"{reference}: no trials were folded in, nothing to save")
    n_subbands = accumulator.real.shape[1] - 1
    if centre_frequencies.shape != (n_subbands,):
        raise ValueError(f"{n_subbands} subbands but {centre_frequencies.shape[0]} centre frequencies")

    real = accumulator.real / accumulator.total_duration
    null = accumulator.null / accumulator.total_duration
    tmp_path = output_path.with_name(output_path.name + ".partial")
    with tmp_path.open("wb") as handle:
        np.savez(
            handle,
            reference=reference,
            channel_names=np.array(accumulator.channel_names),
            frequencies=accumulator.frequencies,
            gammatone_coherence=real[:, -1:, :].astype(COHERENCE_DTYPE),
            null_gammatone_coherence=null[:, :, -1:, :].astype(COHERENCE_DTYPE),
            subband_coherence=real[:, :-1, :].astype(COHERENCE_DTYPE),
            null_subband_coherence=null[:, :, :-1, :].astype(COHERENCE_DTYPE),
            centre_frequencies=centre_frequencies,
            eeg_psd=accumulator.psd / accumulator.total_duration,
            n_trials=len(accumulator.stems),
            total_duration=accumulator.total_duration,
            total_analysed_duration=accumulator.total_analysed_duration,
            trial_stems=np.array(accumulator.stems),
            trial_durations=np.array(accumulator.durations),
            skipped_stems=np.array(accumulator.skipped),
        )
    tmp_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--reference",
        action="append",
        choices=REFERENCES,
        help=f"EEG reference; repeatable. Default: {' '.join(DEFAULT_REFERENCES)}",
    )
    parser.add_argument("--n-null", type=int, default=1, help="Null draws per trial (default 1)")
    parser.add_argument(
        "--n-workers",
        type=int,
        default=6,
        help="Parallel trials, one thread each (default 6; ~4 GB per worker, measured at Cx = 61)",
    )
    parser.add_argument("--max-trials", type=int, default=None, help="Smoke-test on the first N trials only")
    parser.add_argument("--overwrite", action="store_true", help="Recompute references whose output exists")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    references = tuple(dict.fromkeys(args.reference or DEFAULT_REFERENCES))

    config = load_config(ROOT / "src" / "package_data" / "config.json")
    audio_features_dir, eeg_dir, coherence_dir, _ = resolve_configured_inputs(config, ROOT)
    output_dir = coherence_dir / "reference_average"
    output_dir.mkdir(parents=True, exist_ok=True)

    fs = STIMULUS_FS
    nperseg, noverlap = stimulus_window_sizes(fs)

    trials = load_trial_index(coherence_dir)
    if args.max_trials is not None:
        trials = trials.head(args.max_trials)
    if trials.empty:
        raise RuntimeError(f"No included trials found under {coherence_dir}")
    # Stimulus order is what makes the per-worker StimulusCache hit under ordered dispatch.
    trials = trials.sort_values(["audio_name", "eeg_stem"])
    tasks = [{"audio_name": row.audio_name, "eeg_stem": row.eeg_stem} for row in trials.itertuples()]
    # Identical across stimuli, so any one feature file settles them.
    centre_frequencies = load_centre_frequencies(audio_features_dir / f"{tasks[0]['audio_name']}.npz")

    exit_code = 0
    print(
        f"{len(tasks)} trial(s) over {trials['audio_name'].nunique()} stimulus/stimuli, "
        f"references={', '.join(references)}, n_null={args.n_null}, "
        f"{args.n_workers} single-threaded worker(s) of {os.cpu_count()} cores"
    )

    for reference in references:
        # A smoke test writes to its own name: a partial average under the real one would be
        # indistinguishable from a finished run, and would make the next full run skip it.
        suffix = "" if args.max_trials is None else f"_first{args.max_trials}"
        output_path = output_dir / f"average_coherence_{reference}{suffix}.npz"
        if output_path.exists() and not args.overwrite:
            if has_subbands(output_path):
                print(f"{reference}: {output_path} exists, skipping (--overwrite to recompute)")
                continue
            print(f"{reference}: {output_path.name} predates the subband arrays; recomputing, "
                  "and replacing it only once the new average is complete")

        context = {
            "audio_features_dir": audio_features_dir,
            "eeg_dir": eeg_dir,
            "reference": reference,
            "fs": fs,
            "nperseg": nperseg,
            "noverlap": noverlap,
            "n_null": args.n_null,
        }
        accumulator, failures = run_tasks(tasks, context, args.n_workers)
        if failures:
            print(
                f"{reference}: {len(failures)} trial(s) failed, so the average would be over an "
                f"incomplete trial set -- not writing {output_path.name}. Re-run to retry."
            )
            exit_code = 1
            continue
        save_average(accumulator, reference, output_path, centre_frequencies)

        skipped = f", {len(accumulator.skipped)} skipped (no mastoids)" if accumulator.skipped else ""
        print(
            f"{reference}: {len(accumulator.stems)} trial(s){skipped}, "
            f"{accumulator.total_duration / 3600:.2f} h weighted, "
            f"{len(accumulator.channel_names)} channels -> {output_path}"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
