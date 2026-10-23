"""A grid of band-limited onset features: which modulation band's edges drive the response.

:mod:`trfs.onset_envelope` asks *how sharp* an edge has to be to count, by sweeping a low-pass
cutoff. This asks a different question: an onset is a rise in the envelope, but the envelope
rises on many timescales at once, and a rise in the 2-4 Hz band is a different acoustic event
from a rise in the 100-200 Hz band. Band-limiting before differentiating isolates them::

    x_{[a,b]}(t) = max( d/dt bandpass_{[a,b]}[ env(t) ], 0 )

Two things make a grid of these affordable, and both are worth stating because the obvious
implementation of either is intractable.

**Bands as differences of low-passes.** Designing a bandpass per cell of an 85-edge grid is 3570
filter designs per stimulus. Instead only the 85 *edges* are filtered, and a band is the
difference of the two low-passed signals either side of it. For Butterworth filters of equal
order the magnitude response is monotone in ``f / f_c``, so ``L_high(f) >= L_low(f)`` at every
frequency and the difference is a genuine, non-negative bandpass response. Measured, that takes
a stimulus from 51 s to 3 s -- and it buys a property a bank of separately designed bandpasses
would not have: the bands are **exactly additive**, so ``[a,b] + [b,c]`` is ``[a,c]`` to the last
bit, and a grid built from them is internally consistent.

Differentiation is linear and commutes with the low-pass, so it too is done once per edge rather
than once per band. Only the rectification -- the one nonlinearity, and the thing that makes this
an onset feature rather than a filtered copy of the envelope -- is per band.

**The lowest edge is zero.** ``[0, b]`` means "no high-pass", which is exactly the feature
:mod:`trfs.onset_envelope` produces at cutoff ``b``. So the first row of the grid reproduces the
cutoff sweep, and the two analyses are nested rather than merely similar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

from trfs.transfer_function import window_directory_name

#: Filter order before the zero-phase pass, as in :mod:`trfs.onset_envelope`.
DEFAULT_ORDER = 3

#: The default grid: 1 Hz steps to 30 Hz, then 5 Hz steps to 300 Hz.
DEFAULT_EDGES = np.concatenate([np.arange(0.0, 31.0, 1.0), np.arange(35.0, 301.0, 5.0)])

ONSET_GRID_KEY = "onset_grid"


def band_pairs(edges: np.ndarray) -> np.ndarray:
    """Every ``(low, high)`` index pair of the edge list, as ``(n_bands, 2)``.

    Lower-triangular and in a fixed order, so the band axis of everything downstream means the
    same thing without having to be relabelled.
    """
    return np.array([(i, j) for i in range(len(edges)) for j in range(i + 1, len(edges))],
                    dtype=int)


def edge_derivatives(
    envelope: np.ndarray, fs: float, edges: np.ndarray, order: int = DEFAULT_ORDER,
    dtype=np.float32,
) -> np.ndarray:
    """``d/dt lowpass_e(envelope)`` for every edge, as ``(n_edges, n_samples)``.

    Zero-phase, because these transfer functions are read for their latency and a causal
    filter's group delay would be added to every lag -- at a 4 Hz cutoff that is 82 ms, which is
    larger than the response being measured.

    An edge at 0 Hz is the zero array: it stands for "no high-pass", so that ``[0, b]`` is the
    plain onset envelope at cutoff ``b``.
    """
    envelope = np.asarray(envelope, float).reshape(-1)
    out = np.zeros((len(edges), envelope.size), dtype)
    for index, edge in enumerate(edges):
        if edge <= 0:
            continue
        if edge >= fs / 2:
            raise ValueError(f"edge {edge} is at or above Nyquist ({fs / 2:g} Hz)")
        smoothed = sosfiltfilt(butter(order, edge, btype="low", fs=fs, output="sos"), envelope)
        out[index] = np.diff(smoothed, prepend=smoothed[:1]) * fs
    return out


def band_features(derivatives: np.ndarray, pairs: np.ndarray, out: np.ndarray | None = None
                  ) -> np.ndarray:
    """The rectified band features for `pairs`, as ``(len(pairs), n_samples)``.

    ``out`` lets a caller reuse one buffer across chunks, which is what keeps a grid of
    thousands of bands inside a few gigabytes.
    """
    if out is None:
        out = np.empty((len(pairs), derivatives.shape[1]), derivatives.dtype)
    for row, (low, high) in enumerate(pairs):
        np.subtract(derivatives[high], derivatives[low], out=out[row])
        np.maximum(out[row], 0.0, out=out[row])
    return out


def resolve_onset_grid_dir(
    config: dict[str, Any],
    reference: str,
    window_seconds: float,
    overlap_fraction: float,
    root: str | Path | None = None,
) -> Path:
    """Where this reference's onset-grid results live, by the usual resolution order."""
    entry = config.get("references", {}).get(reference, {})
    configured = entry.get("onset_grid_dir") or config.get("onset_grid_dir")
    if configured is None:
        raise KeyError(
            f"No onset_grid_dir configured for reference {reference!r}; add one to config.json"
        )
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = (Path(root or Path(__file__).resolve().parents[2]) / path).resolve()
    return path / window_directory_name(window_seconds, overlap_fraction)
