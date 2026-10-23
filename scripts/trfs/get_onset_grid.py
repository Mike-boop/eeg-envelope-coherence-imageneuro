"""Transfer functions for a grid of band-limited onset features, pooled in the script.

For every ``(low, high)`` cell of a modulation-frequency grid the envelope is band-limited,
differentiated and half-wave rectified, and a transfer function is fitted against the EEG. The
question is whether the response cares *which band* an acoustic edge lives in -- a rise in the
2-4 Hz band is a different event from one in the 100-200 Hz band, and only the first is syllabic.

Unlike the other TRF scripts this one **pools inside the script** and writes a single file: the
per-trial parts would be 630 MB a trial at the default grid, 355 GB for the run, against under a
gigabyte pooled. The cost is that ridge and the EEG-SD normalisation would have to be decided
here -- so neither is applied. What is written is the plain quotient ``S_xy / S_xx`` in the EEG's
own units, with the feature's SD and the EEG's SD beside it so either normalisation can be
applied afterwards.

The two auto-spectra are written as well -- ``feature_power`` is the ``S_xx`` the quotient
divided by, ``eeg_power`` is ``S_yy`` per channel -- weighted per half exactly as the fits are.
They cost 28 MB against the file's 1.8 GB and they are what turns a transfer function into an
answerable question: ``|H|**2 * S_xx`` is the EEG power the band's feature predicts at each
frequency, and over ``S_yy`` it is the fraction of the EEG there that it accounts for. Without
them a cell can only be read as an amplitude in arbitrary units, through a display filter whose
passband has to be chosen before the answer is known.

Three things make the default 3570-band grid affordable, each worth about a factor of ten:

* **Bands as differences of low-passes** (:mod:`trfs.onset_grid`): 85 filter passes per stimulus
  rather than 3570. Measured, 3 s a stimulus against 51 s.
* **Stimuli outermost.** The stimulus-side FFT is 127 s for the whole grid and would dominate if
  it ran per trial; holding one stimulus's trials together runs it 69 times, not 563.
* **The frequency axis truncated before the cross-spectrum.** Keeping 0-100 Hz rather than
  0-512 Hz takes the matmul from 17.8 s a trial to 3.6 s.

The Welch density scaling and the one-sided fold are applied to **neither** side, because they
are identical in ``S_xy`` and ``S_xx`` and cancel exactly in the quotient. That also sidesteps a
real trap: ``_onesided_fold`` treats the last bin of its input as Nyquist, which is true of a
full spectrum and false of a truncated one.

Two halves of the participants are pooled separately, so a band can be asked how *reliable* its
response is and not only how large.

With ``--null`` each stimulus's features are paired with a different stimulus's EEG -- real data
on both sides, wrong pairing. That is the floor a band has to clear, and it is a separate run of
the same cost.

Outputs::

    <onset_grid_dir>/window-5s_overlap-0.80/onset_grid[_null].npz

Keys: ``fits`` (2, n_bands, F, C), ``feature_power`` (2, n_bands, F), ``eeg_power`` (2, F, C),
``half_weight`` (2) -- what all three are weighted by, one per half in a whole run -- the band
edges, the frequency axis, and the run's parameters. The fraction of EEG power a band's feature
accounts for at each frequency is ``|fits/w|**2 * (feature_power/w) / (eeg_power/w)``.

Examples
--------
    python scripts/trfs/get_onset_grid.py
    python scripts/trfs/get_onset_grid.py --null
    python scripts/trfs/get_onset_grid.py --max-frequency 30 --workers 6
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _cap_blas_threads() -> None:
    """Divide the cores between workers before numpy picks a thread count.

    The same trap the envelope script documents: BLAS defaults to one thread per core *per
    process*, measured at a 20x penalty when four workers each took all 24.
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

from coherence.coherence import windowed_fft
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
    resolve_configured_inputs,
)
from transcription.config import load_config
from trfs.eeg import load_avg_referenced_eeg
from trfs.onset_grid import (
    DEFAULT_EDGES,
    DEFAULT_ORDER,
    band_features,
    band_pairs,
    edge_derivatives,
    resolve_onset_grid_dir,
)
from trfs.transfer_function import rereference_transfer_function

mne.set_log_level("WARNING")

_WORKER: dict[str, Any] = {}


# ---------------------------------------------------------------------------------------------
# One stimulus: its trials' EEG, then the grid streamed past them
# ---------------------------------------------------------------------------------------------


def load_eeg_spectra(paths, reference, fs, nperseg, noverlap, keep):
    """`(spectra, channels, sds)` for a set of trials, truncated to `keep` bins at complex64.

    Re-referenced in the time domain rather than after the transform, which costs nothing and
    gives the per-channel SD exactly, from the trace itself rather than from a scaled integral
    of a truncated spectrum.

    Truncating before anything is stored is what makes holding a whole stimulus's trials
    affordable: the full-resolution spectrum is 2.8 GB a trial and this is 246 MB.
    """
    spectra, sds, channels = [], [], None
    for path in paths:
        eeg_data, eeg_fs, names = load_avg_referenced_eeg(path)
        if eeg_fs != fs:
            raise ValueError(f"{path.name}: EEG rate {eeg_fs} != stimulus rate {fs}")
        referenced, kept = rereference_transfer_function(eeg_data.T, names, reference)
        channels = channels or kept
        sds.append(referenced.std(axis=0))
        spectrum = windowed_fft(referenced.T, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1)
        # (n_frames, F, C) -> (keep, n_frames, C), contiguous, for the batched matmul.
        spectra.append(np.ascontiguousarray(spectrum.Xf[:, :keep, :].transpose(1, 0, 2),
                                            dtype=DRAW_DTYPE))
        del spectrum, eeg_data, referenced
    return spectra, channels, sds


def process_stimulus(task: dict[str, Any]) -> tuple[str, int, float]:
    """Accumulate every trial paired with this stimulus into the worker's running totals."""
    started = time.perf_counter()
    context = _WORKER
    fs, nperseg, noverlap = context["fs"], context["nperseg"], context["noverlap"]
    keep, pairs, edges = context["keep"], context["pairs"], context["edges"]

    with np.load(context["audio_features_dir"] / f"{task['audio_name']}.npz") as saved:
        envelope = np.asarray(saved[GAMMATONE_FEATURE_KEY], float).reshape(-1)
    derivatives = edge_derivatives(envelope, fs, edges, context["order"])

    spectra, channels, sds = load_eeg_spectra(
        [context["eeg_dir"] / f"{stem}.fif" for stem in task["eeg_stems"]],
        context["reference"], fs, nperseg, noverlap, keep,
    )
    if context["totals"] is None:
        context["channels"] = channels
        context["totals"] = np.zeros((2, len(pairs), keep, len(channels)), COHERENCE_DTYPE)
        context["feature_sd"] = np.zeros(len(pairs))
        context["eeg_sd"] = np.zeros(len(channels))
        # Per half, and in float64: these are divided into each other downstream, and the
        # fits they are divided into are accumulated per half too.
        context["feature_power"] = np.zeros((2, len(pairs), keep))
        context["eeg_power"] = np.zeros((2, keep, len(channels)))
        # What each half's accumulators are weighted by. One in a whole run, by construction
        # -- every participant contributes their listening and the halves are equal counts --
        # but not under --max-stimuli, and a file that carries it can be divided without
        # reconstructing the trial table to find out.
        context["half_weight"] = np.zeros(2)
    assert channels == context["channels"], f"{task['audio_name']}: channel axis changed"

    # The stimulus side counts too: a recording longer than its stimulus would otherwise leave
    # `Xf[:n_frames]` short and the matmul contracting two different lengths.
    hop = context["nperseg"] - context["noverlap"]
    stimulus_frames = (derivatives.shape[1] - context["noverlap"]) // hop
    n_frames = min(stimulus_frames, min(spectrum.shape[1] for spectrum in spectra))
    buffer = np.empty((context["chunk"], derivatives.shape[1]), np.float32)
    total_share = float(np.sum(task["shares"]))
    # A stimulus's features are the same for every trial that heard it, so its S_xx enters
    # once per half with that half's share of the listening rather than once per trial.
    half_share = [float(sum(share for share, half in zip(task["shares"], task["halves"])
                            if half == index)) for index in (0, 1)]

    for start in range(0, len(pairs), context["chunk"]):
        block = pairs[start:start + context["chunk"]]
        features = band_features(derivatives, block, buffer[:len(block)])
        context["feature_sd"][start:start + len(block)] += total_share * features.std(axis=1)

        spectrum = windowed_fft(features, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1)
        Xf = spectrum.Xf[:n_frames, :keep, :]                        # (frames, keep, n_block)
        # Raw, unfolded and unscaled: the density scaling and the one-sided fold are the same
        # factor in numerator and denominator and cancel in S_xy / S_xx.
        Pxx = (np.abs(Xf) ** 2).sum(0)                               # (keep, n_block)
        # Kept, not only divided out. Same convention as S_xy -- unscaled and unfolded -- so
        # the Welch factors cancel in |S_xy|**2 / (S_xx S_yy) exactly as they do in S_xy / S_xx.
        for index, weight in enumerate(half_share):
            if weight:
                context["feature_power"][index, start:start + len(block)] += weight * Pxx.T
        Xf_t = np.ascontiguousarray(np.conj(Xf).transpose(1, 2, 0), dtype=DRAW_DTYPE)
        del spectrum, Xf

        for spec, share, half in zip(spectra, task["shares"], task["halves"]):
            Sxy = np.matmul(Xf_t, np.ascontiguousarray(spec[:, :n_frames, :]))
            context["totals"][half, start:start + len(block)] += (
                share * (Sxy / Pxx[:, :, None]).transpose(1, 0, 2)).astype(COHERENCE_DTYPE)
            del Sxy
        del Xf_t

    # Outside the band loop: the EEG does not depend on which bands are being fitted, and
    # inside it this would be added once per chunk of the grid rather than once per trial.
    for spec, sd, share, half in zip(spectra, sds, task["shares"], task["halves"]):
        context["eeg_sd"] += share * sd
        context["half_weight"][half] += share
        context["eeg_power"][half] += share * (
            np.abs(spec[:, :n_frames, :]) ** 2).sum(1)
    return task["audio_name"], len(spectra), time.perf_counter() - started


def _init_worker(context: dict[str, Any]) -> None:
    _WORKER.update(context)
    _WORKER.update(channels=None, totals=None, feature_sd=None, eeg_sd=None)


def process_block(block: list[dict[str, Any]]) -> str | None:
    """Run a worker's whole share of the stimuli, then write its partial totals.

    One task per worker rather than one per stimulus, and deliberately so: with dynamic
    dispatch there is no point at which a worker knows it is finished, and the obvious way to
    collect the accumulators afterwards -- mapping a flush over ``range(workers)`` -- does not
    guarantee one call per process. A worker could take two flushes and another none, silently
    dropping a whole accumulator. A static partition has no such hole.
    """
    for position, task in enumerate(block, 1):
        name, count, seconds = process_stimulus(task)
        print(f"  [{os.getpid()}] {position}/{len(block)} {name}: {count} trials, "
              f"{seconds:.0f} s", flush=True)
    if _WORKER["totals"] is None:
        return None
    path = Path(_WORKER["partial_dir"]) / f"partial_{os.getpid()}.npz"
    np.savez(path, totals=_WORKER["totals"], feature_sd=_WORKER["feature_sd"],
             eeg_sd=_WORKER["eeg_sd"], feature_power=_WORKER["feature_power"],
             eeg_power=_WORKER["eeg_power"], half_weight=_WORKER["half_weight"],
             channel_names=np.array(_WORKER["channels"]))
    return str(path)


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reference", choices=REFERENCES, default="avg")
    parser.add_argument("--edges", type=float, nargs="+", default=None,
                        help="Band edges in Hz (default: 1 Hz steps to 30, then 5 Hz to 300)")
    parser.add_argument("--max-frequency", type=float, default=100.0,
                        help="Keep the EEG spectrum to this frequency (default 100 Hz); the "
                             "matmul and the output both scale with it")
    parser.add_argument("--order", type=int, default=DEFAULT_ORDER)
    parser.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
    parser.add_argument("--overlap", type=float, default=OVERLAP_FRACTION)
    parser.add_argument("--chunk", type=int, default=64,
                        help="Bands transformed at once (default 64); the peak memory term")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-stimuli", type=int, default=None, help="Smoke test")
    parser.add_argument("--null", action="store_true",
                        help="Pair each stimulus's features with a DIFFERENT stimulus's EEG: "
                             "real data both sides, wrong pairing, which is the SNR floor")
    parser.add_argument("--null-rotation", type=int, default=7,
                        help="How far to rotate the stimulus list for --null (default 7)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    edges = np.asarray(args.edges if args.edges is not None else DEFAULT_EDGES, float)
    edges = np.unique(edges)
    pairs = band_pairs(edges)

    config = load_config()
    audio_features_dir, eeg_dir, _, _ = resolve_configured_inputs(config)
    output_dir = resolve_onset_grid_dir(
        config, args.reference, args.window_seconds, args.overlap, ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = output_dir / f".partial_{'null' if args.null else 'real'}"
    partial_dir.mkdir(parents=True, exist_ok=True)
    for stale in partial_dir.glob("partial_*.npz"):
        stale.unlink()

    nperseg = int(round(args.window_seconds * STIMULUS_FS))
    noverlap = int(round(args.overlap * nperseg))
    frequencies = np.fft.rfftfreq(nperseg, 1 / STIMULUS_FS)
    keep = int(np.searchsorted(frequencies, args.max_frequency, side="right"))

    trials = build_trial_index(eeg_dir).sort_values(["audio_name", "sub"]).reset_index(drop=True)
    # Each trial's weight: duration within participant, participants equally. The durations
    # come from the FIF headers, which is 563 small reads over NFS at 2.7 s each -- 25 minutes
    # serially. Read on threads (the wait is latency, not Python) and cached beside the output,
    # keyed on the stem list so a changed dataset misses rather than silently reusing lengths.
    duration_cache = output_dir.parent / "trial_durations.npz"
    stems = np.array(list(trials["eeg_stem"]))
    durations = None
    if duration_cache.exists():
        with np.load(duration_cache) as saved:
            if np.array_equal(saved["eeg_stem"], stems):
                durations = saved["duration"]
                print(f"trial durations from {duration_cache.name}")
    if durations is None:
        print(f"reading {len(stems)} FIF headers for trial durations "
              f"(~{len(stems) * 2.7 / 60:.0f} min serial, threaded here)...", flush=True)

        def _duration(stem):
            raw = mne.io.read_raw_fif(eeg_dir / f"{stem}.fif", preload=False, verbose="ERROR")
            return raw.n_times / raw.info["sfreq"]

        with ThreadPoolExecutor(max_workers=16) as pool:
            durations = np.array(list(pool.map(_duration, stems)))
        duration_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(duration_cache, eeg_stem=stems, duration=durations)
        print(f"  cached to {duration_cache}", flush=True)
    trials["duration"] = durations
    subject_weight = trials.groupby("sub")["duration"].sum()
    people = sorted(subject_weight.index)
    # Alternate participants between halves, so the two are matched in size and in listening
    # time rather than split by whatever order the subject ids happen to be in.
    half_of = {person: index % 2 for index, person in enumerate(people)}
    half_size = [sum(1 for p in people if half_of[p] == h) for h in (0, 1)]
    trials["half"] = [half_of[s] for s in trials["sub"]]
    trials["share"] = [
        row.duration / subject_weight[row.sub] / half_size[half_of[row.sub]]
        for row in trials.itertuples()
    ]

    names = list(dict.fromkeys(trials["audio_name"]))
    if args.max_stimuli is not None:
        names = names[: args.max_stimuli]
        trials = trials[trials["audio_name"].isin(names)]
    grouped = {name: block for name, block in trials.groupby("audio_name")}

    tasks = []
    for index, name in enumerate(names):
        # In null mode the features come from `name` and the EEG from another stimulus, so the
        # pairing is broken while both sides stay real data with their own weights.
        partner = names[(index + args.null_rotation) % len(names)] if args.null else name
        block = grouped[partner]
        tasks.append({
            "audio_name": name,
            "eeg_stems": list(block["eeg_stem"]),
            "shares": list(block["share"]),
            "halves": list(block["half"]),
        })

    print(f"reference {args.reference}; {'NULL (mismatched pairing)' if args.null else 'real'}")
    print(f"{len(edges)} edges {edges[0]:g}-{edges[-1]:g} Hz -> {len(pairs)} bands, "
          f"chunk {args.chunk}")
    print(f"EEG kept to {frequencies[keep - 1]:g} Hz ({keep} of {len(frequencies)} bins)")
    print(f"{len(trials)} trials, {len(people)} participants "
          f"({half_size[0]} + {half_size[1]}), {trials['duration'].sum() / 3600:.1f} h")
    print(f"output ~{2 * len(pairs) * keep * 64 * 8 / 1e9:.2f} GB -> {output_dir}")

    context = {
        "eeg_dir": eeg_dir, "audio_features_dir": audio_features_dir,
        "reference": args.reference, "fs": STIMULUS_FS, "nperseg": nperseg,
        "noverlap": noverlap, "keep": keep, "pairs": pairs, "edges": edges,
        "order": args.order, "chunk": args.chunk, "partial_dir": str(partial_dir),
    }

    # Round-robin rather than contiguous blocks: stimuli vary in length, and dealing them out
    # keeps the workers' shares closer in total duration than slicing the list would.
    blocks = [tasks[start::args.workers] for start in range(min(args.workers, len(tasks)))]
    blocks = [block for block in blocks if block]

    started = time.perf_counter()
    if args.workers <= 1 or len(blocks) == 1:
        _init_worker(context)
        partials = [process_block(tasks)]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=len(blocks), initializer=_init_worker,
                      initargs=(context,)) as pool:
            partials = pool.map(process_block, blocks, chunksize=1)

    partials = [p for p in dict.fromkeys(partials) if p]
    totals, feature_sd, eeg_sd, channels = None, None, None, None
    feature_power = eeg_power = half_weight = None
    for path in partials:
        with np.load(path) as saved:
            totals = saved["totals"] if totals is None else totals + saved["totals"]
            feature_sd = (saved["feature_sd"] if feature_sd is None
                          else feature_sd + saved["feature_sd"])
            eeg_sd = saved["eeg_sd"] if eeg_sd is None else eeg_sd + saved["eeg_sd"]
            feature_power = (saved["feature_power"] if feature_power is None
                             else feature_power + saved["feature_power"])
            eeg_power = (saved["eeg_power"] if eeg_power is None
                         else eeg_power + saved["eeg_power"])
            half_weight = (saved["half_weight"] if half_weight is None
                           else half_weight + saved["half_weight"])
            channels = [str(c) for c in saved["channel_names"]]
        Path(path).unlink()
    partial_dir.rmdir()

    stem = "onset_grid_null" if args.null else "onset_grid"
    output_path = output_dir / f"{stem}.npz"
    temporary = output_path.with_suffix(".npz.tmp")
    with open(temporary, "wb") as handle:
        np.savez(handle, fits=totals, feature_sd=feature_sd, eeg_sd=eeg_sd,
                 feature_power=feature_power, eeg_power=eeg_power, half_weight=half_weight,
                 band_low=edges[pairs[:, 0]], band_high=edges[pairs[:, 1]], edges=edges,
                 frequencies=frequencies[:keep], channel_names=np.array(channels),
                 half_sizes=np.array(half_size), n_trials=len(trials),
                 hours=trials["duration"].sum() / 3600, reference=args.reference,
                 window_seconds=float(args.window_seconds),
                 overlap_fraction=float(args.overlap), nperseg=nperseg, noverlap=noverlap,
                 fs=float(STIMULUS_FS), filter_order=args.order, is_null=bool(args.null))
        handle.flush()
    fsync_path(temporary)
    temporary.rename(output_path)

    print(f"\nwrote {output_path} ({output_path.stat().st_size / 1e9:.2f} GB) "
          f"in {(time.perf_counter() - started) / 60:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
