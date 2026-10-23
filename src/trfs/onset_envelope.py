"""The onset envelope (``peakRate``) of the broadband amplitude envelope.

The reviewer's feature: low-pass the broadband gammatone envelope, differentiate it, and
half-wave rectify what is left, so that only the rising edges survive. It is the standard
acoustic-onset regressor -- Oganian & Chang's ``peakRate`` -- and it asks a different question
from the envelope itself: not *how loud* the stimulus is at each moment but *how sharply it is
getting louder*, which is where syllable onsets live.

Three choices in it are not free, and each is made here rather than left implicit:

**The low-pass has to be zero-phase.** These transfer functions are read for their latency, and
a causal filter's group delay would be added to every lag -- at a 4 Hz cutoff a third-order
Butterworth's delay is tens of milliseconds, which is the size of the effect being measured.
``sosfiltfilt`` runs the filter forwards and backwards, so the phase cancels exactly and the
delay is zero. The price is that the magnitude response is squared, which is why the order here
is low; see :func:`lowpass_envelope`.

**The cutoff is the parameter of interest.** It sets what counts as an onset: below about 4 Hz
only the slowest swells survive, and above about 20 Hz the derivative starts to follow
individual glottal pulses rather than syllabic rises. The script sweeps it.

**The derivative keeps its length and its units.** ``np.diff`` returns one sample fewer, so the
first sample is prepended back; and the difference is scaled by the sample rate, so the feature
is an amplitude rate per second rather than per sample. The scale does not change a transfer
function's shape -- ``H = S_xy / S_xx`` is homogeneous of degree -1 in the feature, so a
constant factor only rescales ``H`` -- but it makes the saved spectra comparable with the
envelope's own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

from trfs.transfer_function import window_directory_name

#: Default filter order. Low on purpose: `sosfiltfilt` applies the response twice, so this is an
#: effective order of 6, and a steeper filter at a 4 Hz cutoff rings for long enough to smear
#: the very onsets the feature is meant to isolate.
DEFAULT_ORDER = 3

#: The key the cutoff axis is saved under, and the stem of the two spectrum arrays.
ONSET_FEATURE_KEY = "onset_envelope"


def lowpass_envelope(
    envelope: np.ndarray, fs: float, cutoff: float, order: int = DEFAULT_ORDER
) -> np.ndarray:
    """Zero-phase low-pass along the last axis, at ``cutoff`` Hz.

    Butterworth in second-order sections rather than transfer-function form: at a 4 Hz cutoff
    and 1024 Hz sampling the poles sit very close to the unit circle, and the ``(b, a)`` form
    loses accuracy there badly enough to matter.
    """
    if not 0.0 < cutoff < fs / 2:
        raise ValueError(f"cutoff must be in (0, {fs / 2:g}) Hz, got {cutoff}")
    sos = butter(order, cutoff, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(envelope, float), axis=-1)


def onset_envelope(
    envelope: np.ndarray, fs: float, cutoff: float, order: int = DEFAULT_ORDER
) -> np.ndarray:
    """``max(d/dt lowpass(envelope), 0)`` along the last axis, same shape as the input.

    Half-wave rectified, so a fall in amplitude contributes nothing: an onset is a rise, and
    keeping the falls would make the feature a (signed) rate of change rather than an onset
    marker. That rectification is also the only nonlinearity in the chain, and it is what makes
    this feature more than a filtered copy of the envelope.
    """
    smoothed = lowpass_envelope(envelope, fs, cutoff, order)
    # `prepend` keeps the length, and prepending the first sample makes that sample's
    # derivative zero rather than inventing a step at the start of the recording.
    derivative = np.diff(smoothed, axis=-1, prepend=smoothed[..., :1]) * fs
    return np.maximum(derivative, 0.0)


def onset_envelope_bank(
    envelope: np.ndarray, fs: float, cutoffs: np.ndarray, order: int = DEFAULT_ORDER
) -> np.ndarray:
    """One onset envelope per cutoff, stacked as ``(n_cutoffs, n_samples)``.

    Stacked rather than computed one at a time because everything downstream -- the windowed
    FFT, the cross-spectrum against the EEG -- is a batched operation over a channel axis, so a
    bank of cutoffs costs one pass instead of one pass each.
    """
    envelope = np.asarray(envelope, float).reshape(-1)
    return np.stack([onset_envelope(envelope, fs, float(cutoff), order) for cutoff in cutoffs])


def resolve_onset_trf_dir(
    config: dict[str, Any],
    reference: str,
    window_seconds: float,
    overlap_fraction: float,
    root: str | Path | None = None,
) -> Path:
    """Where this reference's onset transfer functions for this window setting live.

    ``config["references"][reference]["onset_trf_dir"]`` when it is there, falling back to the
    top-level ``onset_trf_dir`` -- the same resolution order as every other configured path, so
    a config written before this feature existed still resolves to something sensible.
    """
    entry = config.get("references", {}).get(reference, {})
    configured = entry.get("onset_trf_dir") or config.get("onset_trf_dir")
    if configured is None:
        raise KeyError(
            f"No onset_trf_dir configured for reference {reference!r}; add one to config.json"
        )
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = (Path(root or Path(__file__).resolve().parents[2]) / path).resolve()
    return path / window_directory_name(window_seconds, overlap_fraction)
