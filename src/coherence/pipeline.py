"""Dataset-level plumbing shared by the three coherence scripts.

``coherence.py`` is the signal-processing library and stays pure numpy. This module is the layer
above it: the windowing parameters, dtypes, null-shift convention, directory resolution and file
naming that ``scripts/coherence/get_coherence.py``,
``scripts/coherence/get_null_cochleogram_coherence.py`` and
``scripts/coherence/get_null_gammatone_coherence.py`` must agree on.

It exists because those three used to agree by importing each other -- script 2 pulled
``load_trial_index`` and ``shifted_cross_spectrum`` from script 3, and both pulled ``STIMULUS_FS``
and ``load_eeg`` from script 1. That made the *first* script in the pipeline a library for the
later ones, so a change to how step 1 is invoked could silently alter steps 2 and 3, and it left
the null-shift convention -- the thing every cross-check between them depends on -- duplicated in
three places.

Anything here is, by definition, a decision that has to be identical across the pipeline. Anything
specific to one script's loop structure or output layout belongs in that script.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from concurrent.futures import ProcessPoolExecutor

import mne
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------------------------
# Signal and windowing parameters
# --------------------------------------------------------------------------------------------

# The sample rate the audio features and the EEG recordings share.
STIMULUS_FS = 1024.0

# Welch windowing, shared between the stimulus PSDs and every coherence calculation.
WINDOW_SECONDS = 5.0
OVERLAP_FRACTION = 0.8

# Coherence is computed in complex128 (numpy's default) but stored as complex64: the subband
# arrays are by far the largest on disk, and coherence magnitudes/phases do not need double
# precision once the FFT and averaging are done.
COHERENCE_DTYPE = np.complex64

# Working precision for the null draw in the pooled scripts. Measured 1.8x faster and half the
# memory of complex128 at 5.4e-07 relative error, against output stored as COHERENCE_DTYPE anyway.
DRAW_DTYPE = np.complex64

# Keys in audio_features_dir/*.npz, produced by scripts/feature_extraction/extract_features.py.
SUBBAND_FEATURE_KEY = "subband_envelopes"
GAMMATONE_FEATURE_KEY = "gammatone_envelope"
CENTRE_FREQS_KEY = "gammatone_center_freqs"


def stimulus_window_sizes(fs: float) -> tuple[int, int]:
    """(nperseg, noverlap) for the shared Welch windowing at sample rate ``fs``."""
    nperseg = int(round(WINDOW_SECONDS * fs))
    noverlap = int(round(OVERLAP_FRACTION * nperseg))
    return nperseg, noverlap


# --------------------------------------------------------------------------------------------
# The null-shift convention
# --------------------------------------------------------------------------------------------

# The global-shift null circularly shifts the EEG's window order rather than fully shuffling it.
# A full shuffle destroys the window-to-window correlation that Welch overlap gives real data too,
# making that null unrealistically low; a shift preserves it while still mismatching audio and EEG.
# NULL_MIN_SHIFT_SECONDS is the requested lower bound; the upper bound is each recording's analysed
# duration minus the same margin. NULL_RNG_SEED is fixed so reruns reproduce the same draws.
NULL_MIN_SHIFT_SECONDS = 60.0
NULL_RNG_SEED = 0


def null_shift_bounds(duration_seconds: float) -> tuple[float, float]:
    """(min_shift_seconds, max_shift_seconds), scaled to a recording's own analysed duration.

    Max is the duration minus the same margin as the minimum, so shifts stay clear of wrapping
    back near a 0-lag pairing. For the rare recording too short to leave NULL_MIN_SHIFT_SECONDS of
    clearance on both sides (available frames span roughly ``duration - WINDOW_SECONDS`` seconds),
    both bounds shrink to a quarter of what is available.
    """
    available = duration_seconds - WINDOW_SECONDS
    min_shift = NULL_MIN_SHIFT_SECONDS
    if 2.0 * min_shift >= available:
        min_shift = available / 4.0
    max_shift = available - min_shift
    return min_shift, max_shift


def null_shifts(n_frames: int, fs: float, hop: int, n_null: int, analysed_duration: float) -> np.ndarray:
    """The ``n_null`` global-shift values for one recording, in draw order.

    Reproduces ``coherence.null_coherence_shuffled_windows_from_prep`` in ``mode="global_shift"``:
    same bounds, same fixed-seed generator, one ``rng.integers`` per draw. Draw *i* is therefore
    the same shift that function would produce.

    Returning the whole sequence rather than yielding it is what lets a caller compute draws in
    blocks -- slicing ``[start:stop]`` gives exactly the shifts draws ``start..stop`` would get in
    an uninterrupted run, which is what makes blocked and resumed runs identical to a single pass.

    The generator is re-seeded per recording, so a recording's draw *i* never depends on how many
    recordings preceded it -- the property that lets one trial's draw be folded into several
    different pooled averages and still be the same draw.
    """
    min_shift_seconds, max_shift_seconds = null_shift_bounds(analysed_duration)

    min_shift_frames = int(np.ceil(min_shift_seconds * fs / hop))
    if min_shift_frames * 2 >= n_frames:
        raise ValueError(
            f"min_shift_seconds={min_shift_seconds} needs {min_shift_frames} frames of clearance "
            f"in each direction, but only {n_frames} frames are available."
        )
    max_shift_frames = min(n_frames - min_shift_frames, int(np.floor(max_shift_seconds * fs / hop)))
    if max_shift_frames < min_shift_frames:
        raise ValueError(
            f"max_shift_seconds={max_shift_seconds} is smaller than min_shift_seconds={min_shift_seconds}"
        )

    rng = np.random.default_rng(NULL_RNG_SEED)
    return rng.integers(min_shift_frames, max_shift_frames + 1, size=n_null)


# --------------------------------------------------------------------------------------------
# Directories and file naming
# --------------------------------------------------------------------------------------------

# EEG filenames look like
# "sub-043_-_ses-001_-_trial-003_-_cond-natural_-_audio01-podcast_14_-_eeg.fif"; this glob matches
# every recording of a given stimulus across subjects/sessions/trials/conditions.
EEG_GLOB_TEMPLATE = "*_-_audio01-{audio_name}_-_eeg.fif"


def resolve_configured_inputs(
    config: dict[str, Any], root: str | Path | None = None
) -> tuple[Path, Path, Path, Path]:
    """Resolve the audio-feature, EEG, coherence-output and stimulus-PSD directories."""
    root_path = Path(root or ROOT).expanduser().resolve()

    def _resolve(key: str, default: str) -> Path:
        path = Path(config.get(key, default)).expanduser()
        return path if path.is_absolute() else (root_path / path).resolve()

    return (
        _resolve("audio_features_dir", "data/audio_features"),
        _resolve("eeg_dir", "data/eeg"),
        _resolve("coherence_dir", "data/coherence"),
        _resolve("stimulus_psd_dir", "data/stimulus_psd"),
    )


def resolve_null_coherence_dir(config: dict[str, Any], root: str | Path | None = None) -> Path:
    """The directory the pooled-null scripts own, under which each keeps its own subdirectory."""
    root_path = Path(root or ROOT).expanduser().resolve()
    path = Path(config.get("null_coherence_dir", "data/null_coherence")).expanduser()
    return path if path.is_absolute() else (root_path / path).resolve()


def find_matching_eeg_files(eeg_dir: Path, audio_name: str) -> list[Path]:
    """Every EEG recording of a given audio stimulus, across subjects/sessions/trials."""
    return sorted(eeg_dir.glob(EEG_GLOB_TEMPLATE.format(audio_name=audio_name)))


def parse_coherence_filename(coherence_file: Path) -> dict[str, Any]:
    """Parse a get_coherence.py output filename into its components.

    e.g. "sub-084_-_ses-001_-_trial-002_-_cond-natural_-_audio01-audiobook_14_2_-_eeg_coherence.npz"
    """
    parts = coherence_file.stem.split("_-_")
    assert len(parts) == 6, f"Unexpected coherence filename format: {coherence_file}"
    return {
        "sub": parts[0].removeprefix("sub-"),
        "ses": parts[1].removeprefix("ses-"),
        "trial": parts[2].removeprefix("trial-"),
        "cond": parts[3].removeprefix("cond-"),
        "audio_name": parts[4].removeprefix("audio01-"),
        "coherence_file": coherence_file,
        "eeg_stem": coherence_file.stem.removesuffix("_coherence"),
    }


# --------------------------------------------------------------------------------------------
# EEG references
# --------------------------------------------------------------------------------------------

# The three schemes the original neural_tracking_across_spectrum scripts supported. Every pooled
# script takes one and writes under that reference's own directories, so a reference's outputs can
# never be mistaken for another's -- the arrays are the same shape and the same names.
REFERENCES = ("avg", "linked_mastoids", "FCz")
DEFAULT_REFERENCE = "avg"

MASTOID_CHANNELS = ("A1", "A2")
FCZ_CHANNEL = "FCz"

# Concurrency for the pre-run header scan. Processes, not threads: the wait is NFS latency, but
# FIF header parsing is pure Python and holds the GIL, so 16 threads measured no faster than one
# (1082 s for 563 recordings). Processes overlap the waiting properly.
HEADER_SCAN_WORKERS = 12


def resolve_reference_dirs(
    config: dict[str, Any], reference: str = DEFAULT_REFERENCE, root: str | Path | None = None
) -> tuple[Path, Path]:
    """That reference's ``(coherence_dir, null_coherence_dir)``.

    Read from ``config["references"][reference]`` when it is there, falling back to the top-level
    ``coherence_dir``/``null_coherence_dir`` otherwise -- which is what makes a config written
    before references existed still resolve, and keeps "avg" pointing at the directories its
    outputs are already in.
    """
    if reference not in REFERENCES:
        raise ValueError(f"Unknown reference {reference!r}, expected one of {sorted(REFERENCES)}")
    root_path = Path(root or ROOT).expanduser().resolve()
    entry = config.get("references", {}).get(reference, {})

    def _resolve(key: str, default: str) -> Path:
        path = Path(entry.get(key) or config.get(key, default)).expanduser()
        return path if path.is_absolute() else (root_path / path).resolve()

    coherence_dir = _resolve("coherence_dir", "data/coherence")
    null_dir = _resolve("null_coherence_dir", str(coherence_dir / "null"))
    return coherence_dir, null_dir


def reference_channel_names(fif_path: Path, reference: str = DEFAULT_REFERENCE) -> list[str] | None:
    """The channel axis ``reference`` produces for this recording, from its header alone.

    Reads no sample data, so it is cheap enough to run over every recording before a long job --
    which is what settles the accumulator shapes and which recordings a reference has to skip,
    without loading a single trial. ``None`` mirrors ``load_referenced_eeg``: the reference cannot
    be applied here.
    """
    if reference not in REFERENCES:
        raise ValueError(f"Unknown reference {reference!r}, expected one of {sorted(REFERENCES)}")
    info = mne.io.read_raw_fif(fif_path, preload=False, verbose=False).info
    eeg_names = [name for name, kind in zip(info["ch_names"], info.get_channel_types())
                 if kind == "eeg"]
    if reference == "linked_mastoids":
        return eeg_names if set(MASTOID_CHANNELS) <= set(info["ch_names"]) else None
    if reference == "FCz":
        return [name for name in eeg_names if name != FCZ_CHANNEL]
    return eeg_names


def _reference_applies(task: tuple[Path, str]) -> bool:
    """Worker for the header scan; module-level so it can be sent to a process pool."""
    fif_path, reference = task
    return reference_channel_names(fif_path, reference) is not None


def filter_trials_for_reference(
    df: pd.DataFrame, eeg_dir: Path, reference: str = DEFAULT_REFERENCE
) -> tuple[pd.DataFrame, list[str]]:
    """``(trials the reference applies to, stems it does not)``.

    Done up front rather than when a trial is reached, because the trial set decides group
    membership: a recording skipped mid-run would leave its groups' journals waiting for a trial
    that never arrives, and its duration must never enter a denominator. Only the linked-mastoid
    case drops anything -- 21 of the 617 recordings carry no A1/A2.
    """
    if reference != "linked_mastoids":
        return df.reset_index(drop=True), []

    # ~2 s per header over NFS, so 563 of them serially is ~18 minutes before the run starts.
    stems = list(df["eeg_stem"])
    with ProcessPoolExecutor(max_workers=HEADER_SCAN_WORKERS) as pool:
        applicable = list(pool.map(
            _reference_applies, ((eeg_dir / f"{stem}.fif", reference) for stem in stems),
            chunksize=8,
        ))
    skipped = [stem for stem, found in zip(stems, applicable) if not found]
    return df[~df["eeg_stem"].isin(skipped)].reset_index(drop=True), skipped


# Excluded from every pooled average: "sin"/"av" are different listening conditions, and a
# "shifted"/"artefact" audio name marks a shifted-audio calibration variant or a flagged-bad trial.
# get_coherence.py still computes per-trial coherence for these; they just never enter an average.
EXCLUDED_CONDITIONS = ("sin", "av")
EXCLUDED_AUDIO_SUBSTRINGS = ("shifted", "artefact")


def parse_eeg_filename(eeg_file: Path) -> dict[str, Any]:
    """Parse an EEG recording's filename into the same fields ``parse_coherence_filename`` gives.

    e.g. "sub-084_-_ses-001_-_trial-002_-_cond-natural_-_audio01-audiobook_14_2_-_eeg.fif"

    There is no ``coherence_file`` here: that is the point. The pooled scripts derive their trial
    set from the recordings themselves, so a reference that has no per-trial coherence on disk --
    which is every reference but "avg" -- still has a trial index.
    """
    parts = eeg_file.stem.split("_-_")
    assert len(parts) == 6, f"Unexpected EEG filename format: {eeg_file}"
    return {
        "sub": parts[0].removeprefix("sub-"),
        "ses": parts[1].removeprefix("ses-"),
        "trial": parts[2].removeprefix("trial-"),
        "cond": parts[3].removeprefix("cond-"),
        "audio_name": parts[4].removeprefix("audio01-"),
        "eeg_stem": eeg_file.stem,
    }


def _apply_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    keep = ~df["cond"].isin(EXCLUDED_CONDITIONS)
    for substring in EXCLUDED_AUDIO_SUBSTRINGS:
        keep &= ~df["audio_name"].str.contains(substring, regex=False)
    return df[keep].reset_index(drop=True)


def build_trial_index(eeg_dir: Path) -> pd.DataFrame:
    """One row per EEG recording, minus the trials excluded from pooled averages.

    The reference-independent counterpart of ``load_trial_index``: same columns but for
    ``coherence_file``, same exclusions, and the same row order (both sort by stem), so the two
    agree trial for trial wherever per-trial coherence exists for every recording.
    """
    rows = [parse_eeg_filename(f) for f in sorted(eeg_dir.glob("*.fif"))]
    if not rows:
        raise FileNotFoundError(f"No EEG recordings found under {eeg_dir}")
    return _apply_exclusions(pd.DataFrame(rows))


def load_trial_index(coherence_dir: Path) -> pd.DataFrame:
    """One row per get_coherence.py output, minus the trials excluded from pooled averages."""
    rows = [parse_coherence_filename(f) for f in sorted(coherence_dir.glob("*.npz"))]
    return _apply_exclusions(pd.DataFrame(rows))


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------


def load_referenced_eeg(
    fif_path: Path, reference: str = DEFAULT_REFERENCE
) -> tuple[np.ndarray, float, list[str]] | None:
    """EEG data, sample rate and channel names under ``reference``; ``None`` to skip the recording.

    The channel axis is not the same width for every reference, which is why the names come back
    with the data and every script that writes an average saves them:

        "avg"             average of the 64 scalp electrodes                     -> 64 channels
        "linked_mastoids" mean of A1/A2                                          -> 64 channels
        "FCz"             the FCz electrode, which is then dropped as all-zero   -> 63 channels

    ``None`` means the reference cannot be applied to this recording, which only the linked-mastoid
    case returns: 21 of the 617 recordings carry no A1/A2, and the original scripts skipped them
    rather than falling back to something else. A skipped trial must be left out of the weighted
    average entirely -- its duration never entering the denominator -- and recorded in the output,
    since it makes that reference's trial set smaller than the others'.

    A1/A2 are "misc"-typed in these files, so they are outside ``pick("eeg")`` and cannot leak into
    an average reference or into the output's channel axis. Verified 2026-09-15 that MNE's
    ``set_eeg_reference`` still subtracts them from the EEG channels when they are "misc"
    (bit-exact against a manual subtraction), which is what makes the linked-mastoid case work.
    """
    if reference not in REFERENCES:
        raise ValueError(f"Unknown reference {reference!r}, expected one of {sorted(REFERENCES)}")

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

    return raw.get_data(), float(raw.info["sfreq"]), list(raw.ch_names)


def load_eeg(fif_path: Path) -> tuple[np.ndarray, float]:
    """Average-referenced EEG data (eeg-type channels only) and the recording's sample rate.

    The average-reference case of ``load_referenced_eeg``, kept as its own entry point because it
    is what ``get_coherence.py`` wants and it cannot return ``None``.
    """
    loaded = load_referenced_eeg(fif_path, "avg")
    assert loaded is not None, "the average reference applies to every recording"
    eeg_data, fs, _ = loaded
    return eeg_data, fs


def load_stimulus_features(npz_path: Path) -> np.ndarray:
    """Subband envelopes and the gammatone envelope as one (n_features, n_samples) array.

    The broadband channel is last, so ``features[:-1]`` is the cochleogram and ``features[-1:]`` is
    the broadband envelope -- which is bit-exactly their sum.
    """
    data = np.load(npz_path)
    return np.concatenate([data[SUBBAND_FEATURE_KEY], data[GAMMATONE_FEATURE_KEY]], axis=0)


def load_centre_frequencies(npz_path: Path) -> np.ndarray:
    """The (n_subbands,) ERB-spaced gammatone filter centres, 50 Hz to 8 kHz.

    Distinct from the coherence frequency axis (``np.fft.rfftfreq(nperseg, 1/fs)``, 2561 bins):
    this indexes the *stimulus* channel axis, that one the FFT bins. Identical across stimuli, so
    any one feature file settles them.
    """
    return np.load(npz_path)[CENTRE_FREQS_KEY]


def fsync_path(path: Path) -> None:
    """Flush a written file all the way to disk, for the journal protocols in the pooled scripts."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
