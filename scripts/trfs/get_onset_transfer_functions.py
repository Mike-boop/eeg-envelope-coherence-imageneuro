"""Per-trial spectra for onset-envelope transfer functions, swept over the low-pass cutoff.

The same estimator as ``get_transfer_functions.py`` and the same three arrays saved undivided --

    cross_spectrum_onset_envelope   S_xy   (F, n_cutoffs, Cy) complex
    stimulus_psd_onset_envelope     S_xx   (F, n_cutoffs)
    eeg_psd[_<reference>]           S_yy   (F, Cy)

-- but against the **onset envelope** rather than the envelope itself: the broadband gammatone
envelope low-passed, differentiated and half-wave rectified, so only the rising edges survive
(``peakRate``). See :mod:`trfs.onset_envelope` for why the low-pass has to be zero-phase.

The low-pass cutoff is the free parameter, and rather than one run per value the whole bank is
computed in a single pass. Everything downstream of the feature is a batched operation over a
channel axis, so 17 cutoffs cost one windowed FFT of a 17-row array instead of 17 runs of a
1-row one; the EEG side, which dominates, is done once either way. The cutoff axis is saved with
the arrays, so the analysis picks a cutoff by value rather than by which directory it is in.

Outputs::

    <onset_trf_dir>/window-5s_overlap-0.80/<eeg_stem>.npz

carrying, besides the three arrays: ``cutoffs`` (the axis ``S_xy``'s middle dimension runs
over), ``filter_order``, ``frequencies``, ``channel_names`` (66 where the recording has A1/A2),
``analysed_duration`` and ``duration``, ``stimulus``, ``subject``, ``reference``, and the window
parameters. As in the envelope script a PSD is written for every reachable reference, because
``S_yy`` is ``|Y|**2`` and squaring does not survive a change of reference the way the
cross-spectrum does.

About 27 MB a trial at 17 cutoffs, so ~15 GB for the 563.

Examples
--------
    python scripts/trfs/get_onset_transfer_functions.py
    python scripts/trfs/get_onset_transfer_functions.py --cutoffs 4 8 12
    python scripts/trfs/get_onset_transfer_functions.py --cutoff-range 4 20 1 --workers 4
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


def _cap_blas_threads() -> None:
    """Divide the machine's cores between the workers, before numpy picks a thread count.

    The same trap the envelope script documents: BLAS defaults to one thread per core *per
    process*, and at four workers that is four times oversubscribed. Measured there at 33.0
    s/trial uncapped against 1.67 s/trial capped. Read from argv because it has to happen before
    numpy is imported, which is long before argparse runs.
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


DEFAULT_WORKERS = 4
_cap_blas_threads()

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
    GAMMATONE_FEATURE_KEY,
    OVERLAP_FRACTION,
    REFERENCES,
    STIMULUS_FS,
    WINDOW_SECONDS,
    build_trial_index,
    fsync_path,
    load_referenced_eeg,
    resolve_configured_inputs,
)
from transcription.config import load_config
from trfs.eeg import (
    REFERENCE_DROPS,
    REFERENCE_SUBTRACTIONS,
    load_avg_referenced_eeg,
    reference_weights,
    rereferenced_auto_power,
)
from trfs.onset_envelope import DEFAULT_ORDER, ONSET_FEATURE_KEY, onset_envelope_bank
from trfs.onset_envelope import resolve_onset_trf_dir

mne.set_log_level("WARNING")

# The cutoffs the reviewer's question is about: 4 Hz is about the slowest rise that still counts
# as an onset rather than a swell, and by 20 Hz the derivative starts following individual
# glottal pulses instead of syllabic edges.
DEFAULT_CUTOFFS = np.arange(4.0, 21.0, 1.0)

_WORKER: dict[str, Any] = {}


# ---------------------------------------------------------------------------------------------
# Per-stimulus windowing, cached
# ---------------------------------------------------------------------------------------------


class StimulusCache:
    """One stimulus's windowed onset-envelope bank per worker.

    Trials are dispatched in stimulus order, so a worker sees runs of the same stimulus and this
    hits for all but the first of each run -- which matters more here than in the envelope
    script, since the cached object is ``n_cutoffs`` rows rather than one.
    """

    def __init__(self, audio_features_dir: Path, fs: float, nperseg: int, noverlap: int,
                 cutoffs: np.ndarray, order: int):
        self.audio_features_dir = audio_features_dir
        self.fs, self.nperseg, self.noverlap = fs, nperseg, noverlap
        self.cutoffs, self.order = cutoffs, order
        self.cached: dict[str, Any] | None = None

    def get(self, audio_name: str) -> dict[str, Any]:
        if self.cached is not None and self.cached["audio_name"] == audio_name:
            return self.cached
        self.cached = None  # drop the previous stimulus before allocating the next

        # Only the broadband row is needed; the subbands in the same file are not read at all.
        with np.load(self.audio_features_dir / f"{audio_name}.npz") as saved:
            envelope = np.asarray(saved[GAMMATONE_FEATURE_KEY], float).reshape(-1)
        features = onset_envelope_bank(envelope, self.fs, self.cutoffs, self.order)

        spectrum = windowed_fft(features, fs=self.fs, nperseg=self.nperseg,
                                noverlap=self.noverlap, axis=1)
        Pxx = _auto_power(spectrum)
        np.conj(spectrum.Xf, out=spectrum.Xf)  # in place: this copy is ours and is dropped below
        self.cached = {
            "audio_name": audio_name,
            "Xf_t": spectrum.Xf.transpose(1, 2, 0).astype(DRAW_DTYPE, order="C"),
            "Pxx": Pxx,
            "freqs": spectrum.freqs,
            "n_frames": spectrum.n_frames,
            "scale": spectrum.scale,
            "n_samples": features.shape[1],
        }
        return self.cached


# ---------------------------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------------------------


def compute_trial(
    eeg_path: Path,
    stimulus: dict[str, Any],
    reference: str,
    fs: float,
    nperseg: int,
    noverlap: int,
) -> dict[str, Any] | None:
    """One trial's ``(S_xy, S_xx, S_yy)``, or ``None`` if the reference cannot apply.

    Identical to the envelope script's, line for line, except that the stimulus side carries a
    cutoff axis instead of a single channel -- which is the point: the two feature families are
    compared, so they must not differ in the estimator.
    """
    if reference == "avg":
        # Keeps A1/A2 where the recording has them, and never skips a recording: the average is
        # formed from the 64 standard channels but applied to the mastoids too.
        eeg_data, eeg_fs, channel_names = load_avg_referenced_eeg(eeg_path)
    else:
        loaded = load_referenced_eeg(eeg_path, reference)
        if loaded is None:
            return None
        eeg_data, eeg_fs, channel_names = loaded
    if eeg_fs != fs:
        raise ValueError(f"{eeg_path.name}: EEG rate {eeg_fs} != stimulus rate {fs}")

    y_full = windowed_fft(eeg_data, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1)

    # Drop trailing windows from whichever side has more. Window k always covers samples
    # [k*hop, k*hop+nperseg), so the surviving windows are bit-identical to a truncation first.
    n_frames = min(stimulus["n_frames"], y_full.n_frames)
    y_spectrum = truncate_spectrum(y_full, n_frames)
    Pyy = _auto_power(y_spectrum)
    Yf_t = contiguous_for_matmul(y_spectrum.Xf, DRAW_DTYPE)
    del y_full, y_spectrum

    if n_frames == stimulus["n_frames"]:
        Xf_t, Pxx, scale = stimulus["Xf_t"], stimulus["Pxx"], stimulus["scale"]
    else:
        # Measured across this dataset the branch never fires -- no trial is shorter than its
        # stimulus -- but a short recording must not silently use the wrong denominator.
        scale = stimulus["scale"] / stimulus["n_frames"] * n_frames
        Xf_t = np.ascontiguousarray(stimulus["Xf_t"][:, :, :n_frames])
        Pxx = auto_power_transposed(Xf_t, scale, nperseg)

    Sxy = shifted_cross_spectrum(Xf_t, Yf_t, 0, scale, nperseg)  # (F, n_cutoffs, Cy)

    # The EEG PSD under each of the other references, computed here because it cannot be
    # computed later: S_yy is |Y|**2 and squaring is not linear. No second FFT -- see
    # `rereferenced_auto_power`. Only under "avg", the fit that keeps the channels the other
    # references are defined from.
    referenced_psd: dict[str, tuple[np.ndarray, list[str]]] = {}
    if reference == "avg":
        for name in REFERENCE_SUBTRACTIONS:
            weights = reference_weights(channel_names, name)
            if weights is None:
                continue  # this recording lacks the channels that reference is defined from
            power = rereferenced_auto_power(Yf_t, Pyy, weights, scale, nperseg)
            keep = [i for i, ch in enumerate(channel_names) if ch not in REFERENCE_DROPS[name]]
            referenced_psd[name] = (power[:, keep], [channel_names[i] for i in keep])

    return {
        "Sxy": Sxy.astype(COHERENCE_DTYPE),
        "Pxx": Pxx,
        "Pyy": Pyy,
        "duration": eeg_data.shape[1] / eeg_fs,
        "analysed_duration": min(eeg_data.shape[1], stimulus["n_samples"]) / fs,
        "channel_names": channel_names,
        "referenced_psd": referenced_psd,
        "frequencies": stimulus["freqs"],
    }


def save_trial(
    trial: dict[str, Any],
    task: dict[str, Any],
    output_path: Path,
    context: dict[str, Any],
) -> None:
    """Write one trial, via a temporary file renamed into place.

    The rename is atomic on POSIX, so an interrupted run leaves either a complete file or none --
    never a truncated one that a resumed run would skip over as though it were finished.
    """
    payload: dict[str, Any] = {
        f"cross_spectrum_{ONSET_FEATURE_KEY}": trial["Sxy"],
        f"stimulus_psd_{ONSET_FEATURE_KEY}": trial["Pxx"],
        # The axis the middle dimension of the cross-spectrum runs over. Saved with the data so
        # a cutoff is chosen by value rather than by remembering the order of a command line.
        "cutoffs": np.asarray(context["cutoffs"], float),
        "filter_order": int(context["order"]),
        "eeg_psd": trial["Pyy"],
        "frequencies": trial["frequencies"],
        **{f"eeg_psd_{name}": power for name, (power, _) in trial["referenced_psd"].items()},
        **{f"channel_names_{name}": np.array(names)
           for name, (_, names) in trial["referenced_psd"].items()},
        "channel_names": np.array(trial["channel_names"]),
        "analysed_duration": float(trial["analysed_duration"]),
        "duration": float(trial["duration"]),
        "stimulus": task["audio_name"],
        "subject": task["sub"],
        "eeg_stem": task["eeg_stem"],
        "reference": context["reference"],
        "window_seconds": float(context["window_seconds"]),
        "overlap_fraction": float(context["overlap"]),
        "nperseg": context["nperseg"],
        "noverlap": context["noverlap"],
        "fs": float(context["fs"]),
    }

    temporary = output_path.with_suffix(".npz.tmp")
    with open(temporary, "wb") as handle:
        np.savez(handle, **payload)
        handle.flush()
    fsync_path(temporary)
    temporary.rename(output_path)


def _init_worker(context: dict[str, Any]) -> None:
    _WORKER.update(context)
    _WORKER["stimulus_cache"] = StimulusCache(
        context["audio_features_dir"], context["fs"], context["nperseg"],
        context["noverlap"], context["cutoffs"], context["order"],
    )


def _process_trial(task: dict[str, Any]) -> tuple[str, str, str | None]:
    """``(stem, status, error)``. The worker writes its own file; nothing large comes back."""
    try:
        stimulus = _WORKER["stimulus_cache"].get(task["audio_name"])
        trial = compute_trial(
            _WORKER["eeg_dir"] / f"{task['eeg_stem']}.fif",
            stimulus,
            _WORKER["reference"],
            _WORKER["fs"],
            _WORKER["nperseg"],
            _WORKER["noverlap"],
        )
        if trial is None:
            return task["eeg_stem"], "skipped", None
        save_trial(trial, task, _WORKER["output_dir"] / f"{task['eeg_stem']}.npz", _WORKER)
        return task["eeg_stem"], "written", None
    except Exception as exc:  # one bad recording must not kill the run
        return task["eeg_stem"], "failed", f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reference", choices=REFERENCES, default="avg",
                        help="EEG reference to fit under (default avg, which keeps A1/A2 and so "
                             "can be moved to either other reference afterwards)")
    parser.add_argument("--cutoffs", type=float, nargs="+", default=None,
                        help="Low-pass cutoffs in Hz (default "
                             f"{DEFAULT_CUTOFFS[0]:g}-{DEFAULT_CUTOFFS[-1]:g} in 1 Hz steps)")
    parser.add_argument("--cutoff-range", type=float, nargs=3, metavar=("LOW", "HIGH", "STEP"),
                        default=None, help="Cutoffs as a range, inclusive of HIGH")
    parser.add_argument("--order", type=int, default=DEFAULT_ORDER,
                        help=f"Butterworth order before the zero-phase pass (default "
                             f"{DEFAULT_ORDER}; filtfilt doubles it in effect)")
    parser.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS,
                        help=f"Welch window length (default {WINDOW_SECONDS:g})")
    parser.add_argument("--overlap", type=float, default=OVERLAP_FRACTION,
                        help=f"Welch overlap fraction (default {OVERLAP_FRACTION:g})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Worker processes (default {DEFAULT_WORKERS}); each peaks around "
                             "4 GB on the EEG spectrum")
    parser.add_argument("--max-trials", type=int, default=None,
                        help="Smoke-test on the first N trials only")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute trials whose output already exists")
    return parser.parse_args()


def resolve_cutoffs(args: argparse.Namespace) -> np.ndarray:
    """The cutoff bank, from either flag or the default."""
    if args.cutoffs is not None and args.cutoff_range is not None:
        raise SystemExit("give --cutoffs or --cutoff-range, not both")
    if args.cutoffs is not None:
        cutoffs = np.asarray(args.cutoffs, float)
    elif args.cutoff_range is not None:
        low, high, step = args.cutoff_range
        if step <= 0:
            raise SystemExit(f"--cutoff-range STEP must be positive, got {step}")
        cutoffs = np.arange(low, high + step / 2, step)
    else:
        cutoffs = DEFAULT_CUTOFFS.copy()
    if cutoffs.size == 0:
        raise SystemExit("no cutoffs")
    if np.any(cutoffs <= 0) or np.any(cutoffs >= STIMULUS_FS / 2):
        raise SystemExit(f"cutoffs must be in (0, {STIMULUS_FS / 2:g}) Hz")
    return cutoffs


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.overlap < 1.0:
        raise SystemExit(f"--overlap must be in [0, 1), got {args.overlap}")
    cutoffs = resolve_cutoffs(args)

    config = load_config()
    audio_features_dir, eeg_dir, _, _ = resolve_configured_inputs(config)
    output_dir = resolve_onset_trf_dir(
        config, args.reference, args.window_seconds, args.overlap, ROOT
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    nperseg = int(round(args.window_seconds * STIMULUS_FS))
    noverlap = int(round(args.overlap * nperseg))

    trials = build_trial_index(eeg_dir)
    # Stimulus order, so consecutive tasks share a stimulus and the per-worker cache hits.
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

    print(f"reference {args.reference}; onset envelope (peakRate), order {args.order}")
    print(f"{len(cutoffs)} cutoffs: "
          + ", ".join(f"{cutoff:g}" for cutoff in cutoffs) + " Hz")
    print(f"window {args.window_seconds:g} s, overlap {args.overlap:g} "
          f"(nperseg {nperseg}, noverlap {noverlap})")
    print(f"{len(trials)} trials over {trials['sub'].nunique()} participants")
    print(f"-> {output_dir}")

    context = {
        "eeg_dir": eeg_dir,
        "audio_features_dir": audio_features_dir,
        "output_dir": output_dir,
        "reference": args.reference,
        "fs": STIMULUS_FS,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "window_seconds": args.window_seconds,
        "overlap": args.overlap,
        "cutoffs": cutoffs,
        "order": args.order,
    }
    tasks = trials.to_dict("records")

    counts = {"written": 0, "skipped": 0, "failed": 0}
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
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=args.workers, initializer=_init_worker,
                      initargs=(context,)) as pool:
            for result in tqdm(pool.imap(_process_trial, tasks), total=len(tasks),
                               desc="trials", unit="trial"):
                collect(result)

    # A lost trial is visible here -- its file is simply absent, and a rerun picks it up -- so a
    # failure is reported rather than fatal to the whole run.
    print(f"\nwrote {counts['written']} trials"
          + (f", {counts['skipped']} skipped by the reference" if counts["skipped"] else "")
          + (f", {counts['failed']} FAILED" if counts["failed"] else ""))
    for stem, error in failures[:5]:
        print(f"  {stem}: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
