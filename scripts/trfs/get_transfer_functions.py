"""Per-trial spectra for frequency-domain transfer functions, envelope to EEG.

For each trial the three arrays a transfer function is made of are written, and the division is
**not** done:

    cross_spectrum_<feature>   S_xy   (F, Cy) complex   the numerator
    stimulus_psd_<feature>     S_xx   (F,)              the denominator
    eeg_psd                    S_yy   (F, Cy)           for normalising, not for dividing
    eeg_psd_<reference>        S_yy   (F, Cy')          the same, under each other reference

so that

    H(f) = S_xy(f) / (S_xx(f) + ridge * mean_f S_xx(f))

is formed at analysis time. That leaves three things open which a saved quotient would have
closed: the **regularisation** ``ridge``, any **normalisation** by band power (``S_yy`` is what
makes a topography readable -- raw ``|H|`` maps EEG amplitude, not tracking), and the **EEG
reference**, since ``S_xy`` is linear in the EEG and the fit keeps A1/A2.

One file per trial, 2.7 MB each, about 1.5 GB for the 563 -- against the hundreds of gigabytes
the old repo's per-trial arrays took, because only one number per frequency is kept rather than
one per frequency per lag. ``trfs.transfer_function.pool_transfer_functions`` assembles the
per-participant average from them.

Outputs::

    <trf_dir>/window-5s_overlap-0.80/<eeg_stem>.npz

each carrying, besides the three arrays: ``frequencies``, ``channel_names`` (66 where the
recording has A1/A2, 64 where it does not), ``analysed_duration`` and ``duration``, ``stimulus``,
``subject``, ``reference``, and the window parameters. ``S_yy`` is a squared quantity and so
does not re-reference the way ``S_xy`` does, which is why a PSD is written for each reference
rather than one for the fit's own -- ``eeg_psd_linked_mastoids`` (64 channels) and
``eeg_psd_FCz`` (63), each with its own ``channel_names_<reference>``. That roughly doubles a
file, to 5.3 MB. With ``--subbands``, also
``cross_spectrum_subband_envelopes`` (F, 60, Cy) and ``stimulus_psd_subband_envelopes`` (F, 60).

Examples
--------
    python scripts/trfs/get_transfer_functions.py
    python scripts/trfs/get_transfer_functions.py --reference linked_mastoids
    python scripts/trfs/get_transfer_functions.py --subbands --workers 2
    python scripts/trfs/get_transfer_functions.py --window-seconds 10 --overlap 0.5
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

    Each trial is several batched matmuls, and BLAS defaults to one thread per core *per
    process*. With N workers that is N times oversubscribed, and the cost is not a mild slowdown:
    measured on this 24-core box at 4 workers, uncapped ran at **33.0 s/trial** against
    **1.67 s/trial** capped -- a 20x loss to context switching, with the load average above 75
    and no I/O wait at all. It only appeared once the per-reference PSDs tripled the number of
    matmuls per trial, which is exactly the kind of change that makes it look like the new work
    is slow rather than the threading.

    Read from argv rather than the parsed arguments because it has to happen before numpy is
    imported, and argparse runs long after that. An explicit OMP_NUM_THREADS is left alone.
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


# Defined here so _cap_blas_threads can read it before numpy is imported.
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
    SUBBAND_FEATURE_KEY,
    WINDOW_SECONDS,
    build_trial_index,
    fsync_path,
    load_centre_frequencies,
    load_referenced_eeg,
    load_stimulus_features,
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
from trfs.transfer_function import resolve_trf_dir

mne.set_log_level("WARNING")

# DEFAULT_WORKERS is set above, before numpy is imported, so _cap_blas_threads can use it.
# The per-worker peak is the EEG's windowed FFT: (n_frames, F, Cy) in complex128 is ~2.8 GB at a
# typical trial length, plus the ~1.4 GB complex64 contiguous copy that outlives it. Four workers
# is ~13 GB of the 62 GB on this machine, leaving room for the page cache the NFS reads depend on.
# Each worker writes its own output, so nothing large goes back through the pipe.

_WORKER: dict[str, Any] = {}


# ---------------------------------------------------------------------------------------------
# Per-stimulus windowing, cached
# ---------------------------------------------------------------------------------------------


class StimulusCache:
    """One stimulus's windowed feature spectrum per worker.

    Trials are dispatched in stimulus order, so a worker sees runs of the same stimulus and this
    hits for all but the first of each run. It matters most under ``--subbands``, where the
    operand is 61 channels rather than 1.
    """

    def __init__(self, audio_features_dir: Path, fs: float, nperseg: int, noverlap: int,
                 subbands: bool):
        self.audio_features_dir = audio_features_dir
        self.fs, self.nperseg, self.noverlap = fs, nperseg, noverlap
        self.subbands = subbands
        self.cached: dict[str, Any] | None = None

    def get(self, audio_name: str) -> dict[str, Any]:
        if self.cached is not None and self.cached["audio_name"] == audio_name:
            return self.cached
        self.cached = None  # drop the previous stimulus before allocating the next

        features = load_stimulus_features(self.audio_features_dir / f"{audio_name}.npz")
        # Broadband is the last row. Without --subbands only that row is transformed, which takes
        # the stimulus-side FFT from 61 channels to 1 -- the largest single saving in the
        # broadband-only run, and the reason it is the default.
        if not self.subbands:
            features = features[-1:]

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

    The cross-spectrum comes back in ``COHERENCE_DTYPE``: the matmul runs at complex64, so a
    complex128 copy would carry no extra precision. The auto-spectra stay float64 -- they are
    small, and they are what a later division is most sensitive to.
    """
    if reference == "avg":
        # Keeps A1/A2 where the recording has them, and never skips a recording: the average is
        # formed from the 64 standard channels but applied to the mastoids too, which is what
        # leaves every other reference reachable from this one fit.
        eeg_data, eeg_fs, channel_names = load_avg_referenced_eeg(eeg_path)
    else:
        # The pipeline's own loader, for a direct fit under one of the other references -- the
        # cross-check that re-referencing the average fit lands in the same place.
        loaded = load_referenced_eeg(eeg_path, reference)
        if loaded is None:
            return None
        eeg_data, eeg_fs, channel_names = loaded
    if eeg_fs != fs:
        raise ValueError(f"{eeg_path.name}: EEG rate {eeg_fs} != stimulus rate {fs}")

    y_full = windowed_fft(eeg_data, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1)

    # Drop trailing windows from whichever side has more. Window k always covers samples
    # [k*hop, k*hop+nperseg), so this is the same estimate a common-length truncation before
    # windowing would have given, with the surviving windows bit-identical.
    n_frames = min(stimulus["n_frames"], y_full.n_frames)
    y_spectrum = truncate_spectrum(y_full, n_frames)
    Pyy = _auto_power(y_spectrum)
    Yf_t = contiguous_for_matmul(y_spectrum.Xf, DRAW_DTYPE)
    del y_full, y_spectrum

    if n_frames == stimulus["n_frames"]:
        Xf_t, Pxx, scale = stimulus["Xf_t"], stimulus["Pxx"], stimulus["scale"]
    else:
        # Truncation only rescales `scale`; Pxx covers windows this trial does not have and is
        # recomputed. Measured across the dataset this branch never fires -- no trial is shorter
        # than its stimulus -- but a short recording must not silently use the wrong denominator.
        scale = stimulus["scale"] / stimulus["n_frames"] * n_frames
        Xf_t = np.ascontiguousarray(stimulus["Xf_t"][:, :, :n_frames])
        Pxx = auto_power_transposed(Xf_t, scale, nperseg)

    Sxy = shifted_cross_spectrum(Xf_t, Yf_t, 0, scale, nperseg)  # (F, Cx, Cy), broadband last

    # The EEG PSD under each of the other references, computed here because it cannot be computed
    # later: S_yy is |Y|**2 and squaring is not linear, so unlike the cross-spectrum it does not
    # survive a change of reference. Without these, `channel_sd` normalisation would only ever be
    # available under the reference the fit was written with. No second FFT -- see
    # `rereferenced_auto_power`. Only under "avg", which is the fit that keeps the channels the
    # other references are defined from.
    referenced_psd: dict[str, tuple[np.ndarray, list[str]]] = {}
    if reference == "avg":
        for name in REFERENCE_SUBTRACTIONS:
            weights = reference_weights(channel_names, name)
            if weights is None:
                continue  # this recording lacks the channels that reference is defined from
            power = rereferenced_auto_power(Yf_t, Pyy, weights, scale, nperseg)
            keep = [i for i, ch in enumerate(channel_names) if ch not in REFERENCE_DROPS[name]]
            referenced_psd[name] = (power[:, keep], [channel_names[i] for i in keep])

    # `duration` is the EEG recording's own length; `analysed_duration` is the overlap that
    # actually entered the estimate, and is what should weight a pooled average. They are equal
    # for every trial in this dataset -- no recording is shorter than its stimulus.
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
    Sxy, Pxx = trial["Sxy"], trial["Pxx"]
    payload: dict[str, Any] = {
        # The broadband envelope is the last stimulus channel, as in the feature files.
        f"cross_spectrum_{GAMMATONE_FEATURE_KEY}": Sxy[:, -1, :],
        f"stimulus_psd_{GAMMATONE_FEATURE_KEY}": Pxx[:, -1],
        "eeg_psd": trial["Pyy"],
        "frequencies": trial["frequencies"],
        # One PSD per reachable reference, each on that reference's own channel axis -- 64 under
        # linked mastoids, 63 under FCz -- so it lines up with what
        # `rereference_transfer_function` returns for the cross-spectrum.
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
    if context["subbands"]:
        payload[f"cross_spectrum_{SUBBAND_FEATURE_KEY}"] = Sxy[:, :-1, :]
        payload[f"stimulus_psd_{SUBBAND_FEATURE_KEY}"] = Pxx[:, :-1]
        if context["centre_frequencies"] is not None:
            payload["centre_frequencies"] = context["centre_frequencies"]

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
        context["noverlap"], context["subbands"],
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
    parser.add_argument("--subbands", action="store_true",
                        help="Also keep the 60 subband cross-spectra (61x the stimulus-side work "
                             "and ~47 GB of output; broadband only by default)")
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


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.overlap < 1.0:
        raise SystemExit(f"--overlap must be in [0, 1), got {args.overlap}")

    config = load_config()
    audio_features_dir, eeg_dir, _, _ = resolve_configured_inputs(config)
    output_dir = resolve_trf_dir(config, args.reference, args.window_seconds, args.overlap, ROOT)
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

    feature = "broadband + 60 subbands" if args.subbands else "broadband envelope"
    print(f"reference {args.reference}; {feature}")
    print(f"window {args.window_seconds:g} s, overlap {args.overlap:g} "
          f"(nperseg {nperseg}, noverlap {noverlap})")
    print(f"{len(trials)} trials over {trials['sub'].nunique()} participants")
    print(f"-> {output_dir}")

    centre_frequencies = None
    if args.subbands:
        centre_frequencies = load_centre_frequencies(
            audio_features_dir / f"{trials['audio_name'].iloc[0]}.npz"
        )

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
        "subbands": args.subbands,
        "centre_frequencies": centre_frequencies,
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

    # Unlike the pooled scripts, a lost trial here is visible -- its file is simply absent, and a
    # rerun picks it up -- so a failure is reported rather than fatal to the whole run.
    print(f"\nwrote {counts['written']} trials"
          + (f", {counts['skipped']} skipped by the reference" if counts["skipped"] else "")
          + (f", {counts['failed']} FAILED" if counts["failed"] else ""))
    for stem, error in failures[:5]:
        print(f"  {stem}: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
