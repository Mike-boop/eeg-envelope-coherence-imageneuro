"""Welch-style spectral coherence between two (possibly multichannel) real signals.

Ported from ``ntracking.coherence.csd_multivariate`` (neural_tracking_across_spectrum), split
so the windowing+FFT pass can be computed once and reused across the true coherence estimate
and every null-distribution draw, and given an actual ``coherence()`` entry point (the old
module only computed cross/auto spectral density; every caller inlined its own coherence
formula, and its one named helper mislabeled complex coherency as magnitude-squared).

``coherence()`` and the ``null_coherence_*`` functions each call ``prepare_coherence()``
internally, so calling more than one of them on the same x, y redoes that windowing+FFT pass
every time. When you need both the true coherence and null draws for the same x, y (e.g. a
significance test), call ``prepare_coherence()`` once yourself and pass the result to
``coherence_from_prep()`` / ``null_coherence_shuffled_windows_from_prep()`` /
``null_coherence_random_phase_from_prep()`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.fft import rfft, rfftfreq


def _hann_periodic(m: int) -> np.ndarray:
    n = np.arange(m, dtype=float)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / m)


def _as_channels(signal: np.ndarray, axis: int) -> np.ndarray:
    """Move the sample axis to the front and ensure a trailing channel axis."""
    signal = np.asarray(signal, dtype=float)
    if axis != 0:
        signal = np.moveaxis(signal, axis, 0)
    if signal.ndim == 1:
        signal = signal[:, None]
    return signal


def _onesided_fold(spectrum: np.ndarray, nperseg: int) -> np.ndarray:
    """Double all bins except DC (and Nyquist, for even nperseg) for a one-sided density PSD."""
    folded = spectrum.copy()
    idx = slice(1, None) if nperseg % 2 else slice(1, -1)
    folded[idx] *= 2.0
    return folded


@dataclass
class WindowedSpectrum:
    """Per-window FFT coefficients for one signal, shared across coherence and every null draw."""

    freqs: np.ndarray  # (F,)
    Xf: np.ndarray  # (n_frames, F, C) complex
    hop: int
    n_frames: int
    nperseg: int
    scale: float  # density-scaling denominator (fs * window energy * n_frames)


def windowed_fft(
    signal: np.ndarray,
    fs: float,
    nperseg: int = 1024,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: str | bool = "constant",
    axis: int = 0,
) -> WindowedSpectrum:
    """Segment a (possibly multichannel) signal into overlapping windows, detrend, taper, rfft."""
    signal = _as_channels(signal, axis)
    n, n_channels = signal.shape

    if noverlap is None:
        noverlap = nperseg // 2
    hop = nperseg - noverlap
    n_frames = 1 + (n - nperseg) // hop
    if n_frames <= 0:
        raise ValueError("Signal shorter than nperseg")

    if isinstance(window, str):
        if window.lower() != "hann":
            raise ValueError("Only 'hann' supported.")
        win = _hann_periodic(nperseg)
    else:
        win = np.asarray(window, dtype=float)
    window_energy = np.sum(win**2)

    shape = (n_frames, nperseg, n_channels)
    strides = (signal.strides[0] * hop, signal.strides[0], signal.strides[1])
    segments = np.array(
        np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides, writeable=False),
        dtype=float,
        copy=True,
    )

    if detrend == "constant":
        segments -= segments.mean(axis=1, keepdims=True)
    elif detrend is not False:
        raise ValueError("detrend must be 'constant' or False")

    segments *= win[None, :, None]

    Xf = rfft(segments, axis=1)
    freqs = rfftfreq(nperseg, 1.0 / fs)
    scale = fs * window_energy * n_frames

    return WindowedSpectrum(freqs=freqs, Xf=Xf, hop=hop, n_frames=n_frames, nperseg=nperseg, scale=scale)


def _matched_windowed_ffts(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int | None,
    window: str,
    detrend: str | bool,
    axis: int,
) -> tuple[WindowedSpectrum, WindowedSpectrum]:
    """Truncate x and y to a common length, then window+FFT each independently."""
    x = _as_channels(x, axis)
    y = _as_channels(y, axis)
    n = min(x.shape[0], y.shape[0])
    x_spectrum = windowed_fft(x[:n], fs, nperseg=nperseg, noverlap=noverlap, window=window, detrend=detrend)
    y_spectrum = windowed_fft(y[:n], fs, nperseg=nperseg, noverlap=noverlap, window=window, detrend=detrend)
    return x_spectrum, y_spectrum


def _auto_power(spectrum: WindowedSpectrum) -> np.ndarray:
    """Per-channel auto-power spectral density (F, C), one-sided density scaling applied."""
    power = np.sum(np.abs(spectrum.Xf) ** 2, axis=0) / spectrum.scale
    return _onesided_fold(power, spectrum.nperseg)


def _conj_transpose_for_matmul(spectrum: WindowedSpectrum) -> np.ndarray:
    """x's conjugated per-window FFT, reshaped to (F, Cx, n_frames) for a batched matmul.

    Hoisted out of the cross-spectrum computation so it can be computed once and reused across
    the true coherence estimate and every null draw (only y's side changes between those)."""
    return np.conj(spectrum.Xf).transpose(1, 2, 0)


def _cross_spectrum(Xf_t: np.ndarray, Yf: np.ndarray, scale: float, nperseg: int) -> np.ndarray:
    """Cross-spectral density (F, Cx, Cy) from x's pre-transposed conjugate FFT (F, Cx,
    n_frames) and a (possibly reordered or phase-rotated) copy of y's per-window FFT
    (n_frames, F, Cy), one-sided density scaling applied.

    Uses a batched matmul rather than np.einsum's default (non-BLAS) contraction path, which is
    ~20x slower at realistic channel/frame counts — the difference matters here because this
    runs once per null draw.
    """
    Yf_t = Yf.transpose(1, 0, 2)  # (F, n_frames, Cy)
    cross = np.matmul(Xf_t, Yf_t) / scale  # (F, Cx, Cy)
    return _onesided_fold(cross, nperseg)


def truncate_spectrum(spectrum: WindowedSpectrum, n_frames: int) -> WindowedSpectrum:
    """The first ``n_frames`` windows of an already-computed spectrum.

    ``_matched_windowed_ffts`` truncates x and y to a common sample length *before* windowing.
    Window k always covers samples ``[k*hop, k*hop+nperseg)``, so dropping trailing samples only
    drops trailing windows -- the surviving ones are bit-identical. That is what lets one signal's
    windowed FFT be computed once and reused against many partners of differing length, instead of
    being rewindowed at each partner's truncation point.

    ``scale`` (fs * window energy * n_frames) is the only field that has to be rescaled.
    """
    if n_frames == spectrum.n_frames:
        return spectrum
    return WindowedSpectrum(
        freqs=spectrum.freqs,
        Xf=spectrum.Xf[:n_frames],
        hop=spectrum.hop,
        n_frames=n_frames,
        nperseg=spectrum.nperseg,
        scale=spectrum.scale / spectrum.n_frames * n_frames,
    )


def contiguous_for_matmul(Xf: np.ndarray, dtype: np.dtype | type | None = None) -> np.ndarray:
    """A ``(n_frames, F, C)`` per-window FFT as a C-contiguous ``(F, n_frames, C)`` array.

    ``Xf.transpose(1, 0, 2)`` already has the shape the batched matmul wants, but as a *view* whose
    frequency axis strides ``F*C*itemsize`` bytes -- so the batch loop over frequency jumps
    megabytes per step and misses cache on every one. Hoisting the transpose out of a draw loop
    hoists the syntax and not the cost; materialising it is what removes the cost.

    Measured on a 24-core Xeon w5-2455X at F=2561, n_frames=1048, C=64: with ``Cx = 1`` the
    materialised operand takes a draw from 1.007 s to 0.152 s (**6.6x**), against a one-off ~2 s
    copy. The gain shrinks as ``Cx`` grows and the arithmetic starts to hide the memory traffic,
    so for a single matmul the copy may not pay -- it is worth it when the operand is reused, as
    it is across null draws or across the trials sharing one stimulus.

    ``order="C"`` is not optional when a ``dtype`` is given: ``astype``'s default ``order="K"``
    preserves the source layout, faithfully reproducing the striding this is meant to remove.
    """
    transposed = Xf.transpose(1, 0, 2)
    if dtype is None:
        return np.ascontiguousarray(transposed)
    return transposed.astype(dtype, order="C")


def contiguous_conj_transpose(Xf: np.ndarray, dtype: np.dtype | type | None = None) -> np.ndarray:
    """A ``(n_frames, F, Cx)`` per-window FFT as a C-contiguous conjugated ``(F, Cx, n_frames)``.

    The x-side counterpart of ``contiguous_for_matmul``: ``_conj_transpose_for_matmul`` returns the
    right shape but as a strided view over the conjugated buffer, with the same cache behaviour.
    """
    transposed = np.conj(Xf).transpose(1, 2, 0)
    if dtype is None:
        return np.ascontiguousarray(transposed)
    return transposed.astype(dtype, order="C")


def auto_power_transposed(Xf_t: np.ndarray, scale: float, nperseg: int) -> np.ndarray:
    """(F, C) auto-power from an already-transposed ``(F, C, n_frames)`` spectrum.

    ``_auto_power`` wants the ``(n_frames, F, C)`` original, which a caller holding only the
    contiguous transpose has usually dropped -- it is the larger of the two at ``Cx = 61``.
    Conjugation does not change the magnitude, so the conjugate transpose is a valid input. The sum
    is taken in float64 regardless of the operand's precision.
    """
    power = np.sum(np.abs(Xf_t.astype(np.complex128)) ** 2, axis=2) / scale
    return _onesided_fold(power, nperseg)


def shifted_cross_spectrum(
    Xf_t: np.ndarray, Yf_t: np.ndarray, shift: int, scale: float, nperseg: int
) -> np.ndarray:
    """Cross-spectral density (F, Cx, Cy) between x and a circularly shifted y, without ever
    materializing the shifted copy of y.

    ``np.roll(Yf, shift, axis=0)[t] == Yf[(t - shift) % n_frames]``, so the sum over frames splits
    into two contiguous runs -- x's first ``shift`` frames pairing with y's last ``shift``, and x's
    remainder with y's head -- each a matmul over plain array views. Equivalent to
    ``_cross_spectrum(Xf_t, np.roll(Yf, shift, axis=0), ...)`` up to floating-point summation
    order, but skips copying the whole ``(n_frames, F, Cy)`` spectrum on every draw. That copy, not
    the arithmetic, dominates at realistic sizes: measured ~2.8x faster end to end at F=2561,
    Cx=61, Cy=64, n_frames=1048, where it avoids a ~2.75 GB copy per draw.

    Both operands must already be C-contiguous in ``(F, ..., ...)`` order -- see
    ``contiguous_for_matmul``. Passing the transposed views is correct but far slower.

    At ``shift == 0`` the first matmul is a zero-length contraction: numpy returns the correct
    all-zeros (F, Cx, Cy), leaving the second term alone as the unshifted cross-spectrum.
    """
    n_frames = Yf_t.shape[1]
    shift %= n_frames
    head = np.matmul(Xf_t[:, :, :shift], Yf_t[:, n_frames - shift :, :])
    tail = np.matmul(Xf_t[:, :, shift:], Yf_t[:, : n_frames - shift, :])
    return _onesided_fold((head + tail) / scale, nperseg)


@dataclass
class CoherenceResult:
    freqs: np.ndarray
    Cxy: np.ndarray  # (F, Cx, Cy) complex coherency
    Sxy: np.ndarray
    Pxx: np.ndarray
    Pyy: np.ndarray


@dataclass
class CoherencePrep:
    """Everything the true coherence estimate and every null draw have in common: the windowed
    FFTs of x and y, their auto-power spectra, and x's pre-transposed conjugate FFT. Computing
    this once via ``prepare_coherence`` and passing it to ``coherence_from_prep`` /
    ``null_coherence_shuffled_windows_from_prep`` / ``null_coherence_random_phase_from_prep``
    avoids redoing the windowing+FFT pass for every one of those calls on the same x, y.
    """

    fs: float
    x_spectrum: WindowedSpectrum
    y_spectrum: WindowedSpectrum
    Pxx: np.ndarray
    Pyy: np.ndarray
    Xf_t: np.ndarray
    denom: np.ndarray  # sqrt(Pxx[:, :, None] * Pyy[:, None, :])


def prepare_coherence(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    nperseg: int = 1024,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: str | bool = "constant",
    axis: int = 0,
) -> CoherencePrep:
    """Windowed FFTs, auto-power spectra and pre-transposed conjugate FFT, shared by
    ``coherence_from_prep`` and every ``null_coherence_*_from_prep`` call on the same x, y."""
    x_spectrum, y_spectrum = _matched_windowed_ffts(x, y, fs, nperseg, noverlap, window, detrend, axis)
    Pxx = _auto_power(x_spectrum)
    Pyy = _auto_power(y_spectrum)
    Xf_t = _conj_transpose_for_matmul(x_spectrum)
    denom = np.sqrt(Pxx[:, :, None] * Pyy[:, None, :])
    return CoherencePrep(fs=fs, x_spectrum=x_spectrum, y_spectrum=y_spectrum, Pxx=Pxx, Pyy=Pyy, Xf_t=Xf_t, denom=denom)


def coherence_from_prep(prep: CoherencePrep) -> CoherenceResult:
    """Complex coherency Cxy = Sxy / sqrt(Pxx * Pyy), from a ``prepare_coherence`` result."""
    Sxy = _cross_spectrum(prep.Xf_t, prep.y_spectrum.Xf, prep.x_spectrum.scale, prep.x_spectrum.nperseg)
    Cxy = Sxy / prep.denom
    return CoherenceResult(freqs=prep.x_spectrum.freqs, Cxy=Cxy, Sxy=Sxy, Pxx=prep.Pxx, Pyy=prep.Pyy)


def coherence(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    nperseg: int = 1024,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: str | bool = "constant",
    axis: int = 0,
) -> CoherenceResult:
    """Complex coherency Cxy = Sxy / sqrt(Pxx * Pyy) between (possibly multichannel) x and y."""
    prep = prepare_coherence(x, y, fs, nperseg=nperseg, noverlap=noverlap, window=window, detrend=detrend, axis=axis)
    return coherence_from_prep(prep)


def _min_distance_permutation(
    n_frames: int,
    min_shift_frames: int,
    rng: np.random.Generator,
    max_restarts: int = 20,
    max_local_swaps: int = 2000,
) -> np.ndarray:
    """A random permutation of {0,...,n_frames-1} where every index k maps to a value at least
    min_shift_frames away, circularly.

    Built by drawing a random permutation, then repeatedly picking a violating position and
    swapping its target with a uniformly random other position, until no violations remain (or
    restarting from a fresh random permutation if a run of swaps doesn't converge). Not a
    perfectly uniform sample over the constrained permutation space, but a cheap, well-behaved
    way to get a random pairing that avoids near-diagonal (i.e. near-identity, autocorrelated-
    by-window-overlap) pairs. Only feasible while min_shift_frames is a modest fraction of
    n_frames (roughly under a third) — as it approaches n_frames/2, essentially no valid
    permutation other than a fixed global shift exists, and this will raise.
    """
    positions = np.arange(n_frames)
    for _ in range(max_restarts):
        perm = rng.permutation(n_frames)
        for _ in range(max_local_swaps):
            circular_distance = np.minimum((perm - positions) % n_frames, (positions - perm) % n_frames)
            violations = np.flatnonzero(circular_distance < min_shift_frames)
            if violations.size == 0:
                return perm
            k = violations[rng.integers(violations.size)]
            j = rng.integers(n_frames)
            perm[k], perm[j] = perm[j], perm[k]
    raise ValueError(
        f"Could not find a window shuffle with min_shift_frames={min_shift_frames} out of "
        f"n_frames={n_frames}; try a smaller min_shift_seconds."
    )


def null_coherence_shuffled_windows_from_prep(
    prep: CoherencePrep,
    n_null: int = 1,
    min_shift_seconds: float = 5.0,
    max_shift_seconds: float | None = None,
    mode: str = "shuffle",
    rng: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Null coherence via mismatching y's Welch windows against x's, from a ``prepare_coherence``
    result. See ``null_coherence_shuffled_windows`` for the pairing modes and their rationale.

    Returns
    -------
    (n_null, F, Cx, Cy) complex coherency.
    """
    if mode not in ("shuffle", "global_shift"):
        raise ValueError(f"mode must be 'shuffle' or 'global_shift', got {mode!r}")
    if max_shift_seconds is not None and mode != "global_shift":
        raise ValueError("max_shift_seconds only applies to mode='global_shift'")

    rng = np.random.default_rng(rng)
    x_spectrum, y_spectrum, denom = prep.x_spectrum, prep.y_spectrum, prep.denom

    n_frames = x_spectrum.n_frames
    hop = x_spectrum.hop
    min_shift_frames = int(np.ceil(min_shift_seconds * prep.fs / hop))
    if min_shift_frames * 2 >= n_frames:
        raise ValueError(
            f"min_shift_seconds={min_shift_seconds} needs {min_shift_frames} frames of "
            f"clearance in each direction, but only {n_frames} frames are available."
        )

    # Circular clearance bounds the shift above at n_frames - min_shift_frames regardless of
    # max_shift_seconds (a larger shift is equivalent to a smaller one from the other direction);
    # max_shift_seconds, when given, tightens that further.
    max_shift_frames = n_frames - min_shift_frames
    if max_shift_seconds is not None:
        capped = int(np.floor(max_shift_seconds * prep.fs / hop))
        if capped < min_shift_frames:
            raise ValueError(
                f"max_shift_seconds={max_shift_seconds} is smaller than min_shift_seconds={min_shift_seconds}"
            )
        max_shift_frames = min(max_shift_frames, capped)

    scale, nperseg_ = x_spectrum.scale, x_spectrum.nperseg
    Cxy_null = np.empty((n_null, *denom.shape), dtype=complex)
    for null_idx in range(n_null):
        if mode == "shuffle":
            pairing = _min_distance_permutation(n_frames, min_shift_frames, rng)
            Yf_paired = y_spectrum.Xf[pairing]
        else:
            shift = rng.integers(min_shift_frames, max_shift_frames + 1)
            Yf_paired = np.roll(y_spectrum.Xf, shift, axis=0)
        Cxy_null[null_idx] = _cross_spectrum(prep.Xf_t, Yf_paired, scale, nperseg_) / denom
    return Cxy_null


def null_coherence_shuffled_windows(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    n_null: int = 1,
    min_shift_seconds: float = 5.0,
    max_shift_seconds: float | None = None,
    mode: str = "shuffle",
    rng: int | np.random.Generator | None = None,
    nperseg: int = 1024,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: str | bool = "constant",
    axis: int = 0,
) -> np.ndarray:
    """Null coherence via mismatching y's Welch windows against x's.

    Two pairing modes, both guaranteeing every paired (x-window, y-window) index is at least
    ``min_shift_seconds`` apart, circularly — so paired windows can't be near-duplicates of
    each other from window overlap:

    - ``mode="shuffle"`` (default): a full random permutation of y's window order, redrawn
      independently for each of the ``n_null`` draws. Destroys window-to-window correlation
      from Welch overlap entirely, which can make this null unrealistically low compared to a
      real (mismatched-pairing) signal's spectrum, where that local correlation is still present.
    - ``mode="global_shift"``: roll y's window order by one random offset per draw (the same
      offset applied to every window in that draw), preserving each window's correlation with
      its neighbors. ``max_shift_seconds``, if given, additionally caps the shift's upper bound
      (e.g. ``min_shift_seconds=60, max_shift_seconds=600`` draws shifts of 1-10 minutes) rather
      than allowing anything up to nearly the full recording length.

    Both preserve each signal's own spectral/autocorrelation structure exactly, since
    reordering windows doesn't change their sum of squared magnitudes — Pxx/Pyy are computed
    once, from the unshuffled windows, and reused for every null draw.

    Returns
    -------
    (n_null, F, Cx, Cy) complex coherency.
    """
    prep = prepare_coherence(x, y, fs, nperseg=nperseg, noverlap=noverlap, window=window, detrend=detrend, axis=axis)
    return null_coherence_shuffled_windows_from_prep(
        prep, n_null=n_null, min_shift_seconds=min_shift_seconds, max_shift_seconds=max_shift_seconds, mode=mode, rng=rng
    )


def null_coherence_random_phase_from_prep(
    prep: CoherencePrep,
    n_null: int = 1,
    rng: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Null coherence via a random unit-magnitude phase applied to each window, from a
    ``prepare_coherence`` result. See ``null_coherence_random_phase`` for the rationale.

    Returns
    -------
    (n_null, F, Cx, Cy) complex coherency.
    """
    rng = np.random.default_rng(rng)
    x_spectrum, y_spectrum, denom = prep.x_spectrum, prep.y_spectrum, prep.denom

    n_frames = x_spectrum.n_frames
    thetas = rng.uniform(0.0, 2.0 * np.pi, size=(n_null, n_frames))
    phases = np.exp(1j * thetas)  # (n_null, n_frames)

    scale, nperseg_ = x_spectrum.scale, x_spectrum.nperseg
    Cxy_null = np.empty((n_null, *denom.shape), dtype=complex)
    for null_idx in range(n_null):
        Yf_phased = y_spectrum.Xf * phases[null_idx, :, None, None]
        Cxy_null[null_idx] = _cross_spectrum(prep.Xf_t, Yf_phased, scale, nperseg_) / denom
    return Cxy_null


def null_coherence_random_phase(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    n_null: int = 1,
    rng: int | np.random.Generator | None = None,
    nperseg: int = 1024,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: str | bool = "constant",
    axis: int = 0,
) -> np.ndarray:
    """Null coherence via a random unit-magnitude phase applied to each window before averaging.

    Each draw multiplies every window of y's per-window spectrum by an independent random
    unit-phase (one phase per window, shared across frequency bins and channels within that
    window), then re-averages into Sxy. Since |phase| = 1, Pxx/Pyy (and thus the normalizing
    denominator) are unchanged; only the coherent averaging that produces true coherence
    structure is destroyed.

    Returns
    -------
    (n_null, F, Cx, Cy) complex coherency.
    """
    prep = prepare_coherence(x, y, fs, nperseg=nperseg, noverlap=noverlap, window=window, detrend=detrend, axis=axis)
    return null_coherence_random_phase_from_prep(prep, n_null=n_null, rng=rng)
