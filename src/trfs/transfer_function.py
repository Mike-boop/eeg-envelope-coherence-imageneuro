"""Welch-style frequency-domain transfer functions between a stimulus feature and EEG.

The estimator is the one the old repo's notebooks formed by hand from stored cross-spectra
(``ntracking``; ``trf_playground/env_trfs.ipynb``)::

    H(f) = Sxy(f) / (Sxx(f) + ridge * mean_f Sxx(f))

with ``Sxy`` the cross-spectral density between stimulus feature and EEG and ``Sxx`` the
stimulus feature's auto-spectral density. What has changed is *where* it is formed. The old
pipeline saved ``Sxy``, ``Sxx`` and ``Syy`` per trial -- hundreds of gigabytes -- and divided at
analysis time, at a cost of hundreds of gigabytes. Here the same three arrays are kept but only
one number per trial, not one per trial per lag: the **cross-spectrum and the two auto-spectra**,
which is 2.7 MB a trial rather than gigabytes, and the division is left to the analysis.

Keeping the parts rather than the quotient is what leaves these three open afterwards:

* **Regularisation.** ``ridge`` enters at the division, so it can be chosen, changed and swept
  without re-running anything. :func:`transfer_function_from_parts`.
* **Normalisation.** Both auto-spectra are on disk, so the band-power normalisation the old
  notebooks applied -- or any other -- is an analysis choice. ``S_yy`` in particular is what
  turns ``|H|`` into something whose topography can be read; without it ``|H|`` maps EEG
  amplitude rather than tracking.
* **Reference.** The cross-spectrum is linear in the EEG, so it re-references exactly as the EEG
  does, into any reference that is a linear combination of the channels the file carries. Because
  the fit keeps A1/A2 (see :mod:`trfs.eeg`), that covers all three of the pipeline's references.
  :func:`rereference_transfer_function` applies to ``S_xy`` and to ``H`` alike, since ``S_xx``
  does not involve the EEG at all.

``S_yy`` is the exception to that last point: it is ``|Y|**2``, and squaring is not linear, so a
saved EEG PSD belongs to the reference it was written under and cannot be moved.

Everything here builds on ``coherence.coherence``'s windowing primitives, so a transfer function
and a coherence computed with the same window settings rest on bit-identical FFTs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from coherence.coherence import (
    WindowedSpectrum,
    _auto_power,
    _onesided_fold,
    auto_power_transposed,
    contiguous_for_matmul,
    shifted_cross_spectrum,
    truncate_spectrum,
    windowed_fft,
)

# --------------------------------------------------------------------------------------------
# Output layout
# --------------------------------------------------------------------------------------------

# One directory per window setting, because a transfer function is only comparable with another
# computed on the same windows -- the frequency axis, the number of averaged segments and the
# leakage all follow from them. Putting them in one directory would leave files that differ in
# nothing a reader can see.
def window_directory_name(window_seconds: float, overlap_fraction: float) -> str:
    """The sub-directory name for one window setting, e.g. ``window-5.0s_overlap-0.80``."""
    return f"window-{window_seconds:g}s_overlap-{overlap_fraction:.2f}"


def resolve_trf_dir(
    config: dict[str, Any],
    reference: str,
    window_seconds: float,
    overlap_fraction: float,
    root: str | Path | None = None,
) -> Path:
    """Where this reference's transfer functions for this window setting live.

    Read from ``config["references"][reference]["trf_dir"]`` when it is there, falling back to the
    top-level ``trf_dir`` -- the same resolution order ``pipeline.resolve_reference_dirs`` uses, so
    a config written before transfer functions existed still resolves.
    """
    entry = config.get("references", {}).get(reference, {})
    configured = entry.get("trf_dir") or config.get("trf_dir")
    if configured is None:
        raise KeyError(
            f"No trf_dir configured for reference {reference!r}; add one to config.json"
        )
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = (Path(root or Path(__file__).resolve().parents[2]) / path).resolve()
    return path / window_directory_name(window_seconds, overlap_fraction)


# --------------------------------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------------------------------


@dataclass
class TransferFunctionResult:
    """One recording's transfer function and the spectra it was formed from."""

    freqs: np.ndarray  # (F,)
    H: np.ndarray  # (F, Cx, Cy) complex
    Pxx: np.ndarray  # (F, Cx) stimulus auto-power
    Pyy: np.ndarray  # (F, Cy) EEG auto-power
    n_frames: int


def ridge_term(Pxx: np.ndarray, ridge: float) -> np.ndarray:
    """The denominator offset, as a fraction of each stimulus channel's own mean power.

    Scaling by ``mean_f Pxx`` rather than using an absolute floor is what makes one ``ridge``
    value mean the same thing across features whose power differs by orders of magnitude -- the
    broadband envelope against a single 50 Hz subband, say. ``ridge=0`` returns exact zeros, so
    the unregularised case is the untouched division rather than an addition of 0.0.
    """
    if ridge == 0.0:
        return np.zeros((1, Pxx.shape[1]), dtype=Pxx.dtype)
    return ridge * Pxx.mean(axis=0, keepdims=True)


def transfer_function_from_spectra(
    Xf_t: np.ndarray,
    Yf_t: np.ndarray,
    Pxx: np.ndarray,
    freqs: np.ndarray,
    Pyy: np.ndarray,
    scale: float,
    nperseg: int,
    ridge: float = 0.0,
) -> TransferFunctionResult:
    """``H = Sxy / (Pxx + ridge * mean Pxx)`` from already-windowed, already-transposed FFTs.

    ``Xf_t`` is ``(F, Cx, n_frames)`` conjugated and C-contiguous and ``Yf_t`` is
    ``(F, n_frames, Cy)`` C-contiguous -- the layouts ``coherence.contiguous_conj_transpose`` and
    ``coherence.contiguous_for_matmul`` produce. Taking them pre-transposed is what lets a caller
    window one stimulus once and reuse it across every trial that heard it.

    The cross-spectrum is computed by ``shifted_cross_spectrum`` at shift 0, which is exactly the
    unshifted cross-spectrum; sharing that kernel with the coherence scripts means a transfer
    function and a coherence over the same windows divide the same numerator.
    """
    Sxy = shifted_cross_spectrum(Xf_t, Yf_t, 0, scale, nperseg)  # (F, Cx, Cy)
    denominator = Pxx + ridge_term(Pxx, ridge)
    return TransferFunctionResult(
        freqs=freqs,
        H=Sxy / denominator[:, :, None],
        Pxx=Pxx,
        Pyy=Pyy,
        n_frames=Yf_t.shape[1],
    )


def transfer_function(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    nperseg: int = 1024,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: str | bool = "constant",
    axis: int = 0,
    ridge: float = 0.0,
) -> TransferFunctionResult:
    """Transfer function from stimulus feature ``x`` to EEG ``y``, both ``(n_samples, C)``.

    The plain entry point: windows both signals, truncates them to a common number of frames and
    divides. The scripts use :func:`transfer_function_from_spectra` instead so they can window a
    stimulus once for all the trials that heard it.
    """
    x_spectrum = windowed_fft(x, fs, nperseg=nperseg, noverlap=noverlap, window=window,
                              detrend=detrend, axis=axis)
    y_spectrum = windowed_fft(y, fs, nperseg=nperseg, noverlap=noverlap, window=window,
                              detrend=detrend, axis=axis)

    # Truncate to the shorter side, as the coherence path does: dropping trailing windows leaves
    # the surviving ones bit-identical, so this is the same estimate a common-length truncation
    # before windowing would have given.
    n_frames = min(x_spectrum.n_frames, y_spectrum.n_frames)
    x_spectrum = truncate_spectrum(x_spectrum, n_frames)
    y_spectrum = truncate_spectrum(y_spectrum, n_frames)

    Pxx = _auto_power(x_spectrum)
    Pyy = _auto_power(y_spectrum)
    Sxy = _onesided_fold(
        np.matmul(np.conj(x_spectrum.Xf).transpose(1, 2, 0), y_spectrum.Xf.transpose(1, 0, 2))
        / x_spectrum.scale,
        nperseg,
    )
    denominator = Pxx + ridge_term(Pxx, ridge)
    return TransferFunctionResult(
        freqs=x_spectrum.freqs,
        H=Sxy / denominator[:, :, None],
        Pxx=Pxx,
        Pyy=Pyy,
        n_frames=n_frames,
    )


# --------------------------------------------------------------------------------------------
# Pooling
# --------------------------------------------------------------------------------------------


class WeightedMean:
    """A duration-weighted running mean whose channel axis may differ between contributions.

    Lazily shaped because the EEG channel count depends on the reference and on whether a
    recording carries mastoids at all -- 66 channels for most, 64 for the 21 without A1/A2.
    Rather than refuse the narrow ones or pad them with NaN, each contribution names the columns
    it fills and every column carries **its own** total weight. A participant whose mastoids
    appear in only some recordings therefore gets mastoid columns averaged over exactly those,
    and ``column_weight`` records how much listening is behind each one.

    Accumulates in complex128 whatever the inputs are: 563 trials of complex64 addition would
    lose the low bits of the smallest terms, and the accumulator is one array, not a per-trial
    cost.
    """

    def __init__(self, n_columns: int | None = None) -> None:
        self.total: np.ndarray | None = None
        self.column_weight: np.ndarray | None = None
        self.n_columns = n_columns
        self.n: int = 0

    def add(
        self, values: np.ndarray, weight: float, columns: np.ndarray | list[int] | None = None
    ) -> None:
        """Fold in one contribution. ``columns`` indexes its channels into the full axis."""
        contribution = np.asarray(values)
        contribution = contribution.astype(
            np.complex128 if np.iscomplexobj(contribution) else float
        )
        if columns is None:
            columns = np.arange(contribution.shape[-1])
        columns = np.asarray(columns, dtype=int)

        needed = int(columns.max()) + 1
        if self.total is None:
            width = self.n_columns if self.n_columns is not None else needed
            self.total = np.zeros((*contribution.shape[:-1], width), dtype=contribution.dtype)
            self.column_weight = np.zeros(width)
        elif self.total.shape[:-1] != contribution.shape[:-1]:
            raise ValueError(
                f"shape {contribution.shape} does not match the accumulator's {self.total.shape}"
            )
        if needed > self.total.shape[-1]:
            # The axis widens when a later contribution brings a channel the earlier ones lacked
            # -- a recording with mastoids arriving after one without. Growing is what lets the
            # first recording seen settle the shape without settling the width.
            pad = needed - self.total.shape[-1]
            self.total = np.pad(self.total, [(0, 0)] * (self.total.ndim - 1) + [(0, pad)])
            self.column_weight = np.pad(self.column_weight, (0, pad))

        self.total[..., columns] += contribution * weight
        self.column_weight[columns] += weight
        self.n += 1

    @property
    def weight(self) -> float:
        """The largest per-column weight: the listening behind the fully-covered channels."""
        if self.column_weight is None:
            return 0.0
        return float(self.column_weight.max())

    def mean(self) -> np.ndarray:
        if self.total is None or not np.any(self.column_weight > 0):
            raise ValueError("nothing accumulated")
        # A column no contribution ever filled comes back NaN rather than a silent zero.
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.total / np.where(self.column_weight > 0, self.column_weight, np.nan)


def transfer_function_from_parts(
    cross_spectrum: np.ndarray, stimulus_psd: np.ndarray, ridge: float = 0.0
) -> np.ndarray:
    """``H = S_xy / (S_xx + ridge * mean S_xx)`` from the arrays a trial file holds.

    The division the script deliberately does not do. ``cross_spectrum`` is ``(..., F, Cy)`` and
    ``stimulus_psd`` ``(..., F)``; the stimulus axis broadcasts against the EEG one, so this
    covers the broadband ``(F, Cy)`` case and a subband ``(F, n_subbands, Cy)`` one alike when
    the PSD is shaped to match.
    """
    denominator = stimulus_psd + float(ridge) * np.asarray(stimulus_psd).mean(axis=-1, keepdims=True)
    return cross_spectrum / denominator[..., None]


@dataclass
class TrialSpectra:
    """One trial file, as written by ``scripts/trfs/get_transfer_functions.py``."""

    cross_spectrum: np.ndarray  # (F, Cy) complex, or (F, n_subbands, Cy)
    stimulus_psd: np.ndarray  # (F,) or (F, n_subbands)
    eeg_psd: np.ndarray  # (F, Cy), in the reference it was written under
    eeg_psd_by_reference: dict[str, tuple[np.ndarray, list[str]]]  # the same, per reference
    frequencies: np.ndarray
    channel_names: list[str]
    analysed_duration: float
    duration: float
    stimulus: str
    subject: str
    reference: str

    def psd_for(self, reference: str) -> tuple[np.ndarray, list[str]]:
        """``(S_yy, channel_names)`` under `reference`, or a clear refusal.

        The fit's own reference uses the PSD written alongside it; the others use the ones the
        run computed while the windowed FFT was still in hand. A file written before those
        existed has only the first, and says so rather than silently normalising by the wrong
        reference's power.
        """
        if reference == self.reference:
            return self.eeg_psd, self.channel_names
        if reference in self.eeg_psd_by_reference:
            return self.eeg_psd_by_reference[reference]
        raise ValueError(
            f"no EEG PSD for {reference!r} in this file (it has "
            f"{[self.reference, *self.eeg_psd_by_reference]}). S_yy is a squared quantity and "
            "does not re-reference; re-run get_transfer_functions.py to add it."
        )


def load_trial(path: Path, feature: str = "gammatone_envelope") -> TrialSpectra:
    """Read one trial file. ``feature`` picks which cross-spectrum, by the key naming it."""
    with np.load(path, allow_pickle=True) as saved:
        cross_key = f"cross_spectrum_{feature}"
        if cross_key not in saved.files:
            raise KeyError(
                f"{path.name} has no {cross_key!r}; it holds "
                f"{[k for k in saved.files if k.startswith('cross_spectrum_')]}"
            )
        # A PSD per reference, where the run wrote them. Older files have only the fit's own,
        # so this is keyed on what is present rather than on what ought to be.
        per_reference = {
            key.removeprefix("eeg_psd_"): (
                saved[key],
                [str(name) for name in saved[f"channel_names_{key.removeprefix('eeg_psd_')}"]],
            )
            for key in saved.files
            if key.startswith("eeg_psd_")
        }
        return TrialSpectra(
            cross_spectrum=saved[cross_key],
            eeg_psd_by_reference=per_reference,
            stimulus_psd=saved[f"stimulus_psd_{feature}"],
            eeg_psd=saved["eeg_psd"],
            frequencies=saved["frequencies"],
            channel_names=[str(name) for name in saved["channel_names"]],
            analysed_duration=float(saved["analysed_duration"]),
            duration=float(saved["duration"]),
            stimulus=str(saved["stimulus"]),
            subject=str(saved["subject"]),
            reference=str(saved["reference"]),
        )


def align_columns(axis: list[str], names: list[str]) -> np.ndarray:
    """Indices of ``names`` in ``axis``, extending it in place for anything new.

    The axis grows in first-seen order except that the mastoids are pinned to the end, so a
    participant whose first recording lacks them ends up with the same layout as one whose first
    recording has them -- and the columns already accumulated keep their indices either way.
    """
    from trfs.eeg import MASTOID_CHANNELS

    for name in names:
        if name not in axis:
            axis.append(name)
    scalp = [name for name in axis if name not in MASTOID_CHANNELS]
    mastoids = [name for name in MASTOID_CHANNELS if name in axis]
    axis[:] = scalp + mastoids
    return np.array([axis.index(name) for name in names])


def channel_standard_deviation(eeg_psd: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    """Each channel's time-domain standard deviation, from its one-sided PSD.

    By Parseval, the variance of a signal is the integral of its one-sided PSD over frequency, so
    this is the SD of the referenced EEG trace itself -- not a band-limited proxy for it. Returned
    as ``(Cy,)``.

    It is valid only for the reference the PSD was written under. ``S_yy`` is ``|Y|**2`` and
    squaring is not linear, so a saved EEG PSD cannot be moved to another reference the way the
    cross-spectrum can; recovering it would need the full channel-by-channel CSD, which is not
    stored.
    """
    return np.sqrt(np.sum(eeg_psd, axis=0) * (frequencies[1] - frequencies[0]))


def pool_transfer_functions(
    paths: list[Path],
    ridge: float = 0.0,
    reference: str = "avg",
    feature: str = "gammatone_envelope",
    drop_mastoids: bool = True,
    normalise: str | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray, float]:
    """Fit per trial, then duration-weight the average. ``(H, channels, weights, hours)``.

    The per-participant average the pipeline used to bake in, now assembled at analysis time so
    ``ridge``, ``reference`` and ``normalise`` are arguments rather than things a run decided.
    Fitting per trial and averaging after -- rather than pooling the cross-spectra and dividing
    once -- is the point: every stimulus has its own ``S_xx``, and one summed numerator over a
    denominator no single stimulus had is a different estimator.

    ``normalise``:
        ``None``          leave ``H`` in its own units. A channel's ``|H|`` is then proportional
                          to that channel's EEG amplitude, so it maps where the EEG is large.
        ``"channel_sd"``  divide each channel by its own EEG standard deviation, per trial. This
                          is what makes channels, and participants, comparable. Only available
                          under the reference the files were written with, for the reason
                          :func:`channel_standard_deviation` gives.

    ``weights`` is per channel, because the mastoids are absent from some recordings; a column no
    trial filled comes back NaN rather than as a silent zero.
    """
    if normalise not in (None, "channel_sd"):
        raise ValueError(f"Unknown normalise {normalise!r}; expected None or 'channel_sd'")

    pool = WeightedMean()
    axis: list[str] = []
    duration = 0.0
    for path in paths:
        trial = load_trial(path, feature)
        cross, channels = rereference_transfer_function(
            trial.cross_spectrum, trial.channel_names, reference, drop_mastoids
        )
        H = transfer_function_from_parts(cross, trial.stimulus_psd, ridge)

        if normalise == "channel_sd":
            psd, psd_channels = trial.psd_for(reference)
            keep = [psd_channels.index(name) for name in channels]
            H = H / channel_standard_deviation(psd, trial.frequencies)[keep]

        pool.add(H, trial.analysed_duration, align_columns(axis, channels))
        duration += trial.analysed_duration
    return pool.mean(), axis, pool.column_weight, duration / 3600


# --------------------------------------------------------------------------------------------
# Re-referencing after the fact
# --------------------------------------------------------------------------------------------


def rereference_transfer_function(
    H: np.ndarray, channel_names: list[str], reference: str, drop_mastoids: bool = True
) -> tuple[np.ndarray, list[str]]:
    """Re-reference a saved transfer function by mixing its EEG channel axis.

    ``Sxy`` is linear in the EEG and ``Pxx`` does not involve the EEG at all, so ``H = Sxy / Pxx``
    inherits that linearity exactly: re-referencing the EEG and then fitting gives the same answer
    as fitting and then applying the same channel mixing to ``H``.

    This reaches any reference that is a linear combination of the channels the file carries.
    Because :func:`trfs.eeg.load_avg_referenced_eeg` keeps A1/A2, that is **all three** of the
    pipeline's references -- the common average cancels in every difference:

        ``"avg"``             already applied; re-applying is a no-op on the scalp channels.
        ``"FCz"``             subtract the FCz column, then drop it.
        ``"linked_mastoids"`` subtract the mean of the A1/A2 columns, then drop them.

    ``drop_mastoids`` leaves the result on the same channel axis the coherence pipeline produces
    for that reference -- 63 channels under FCz, 64 under the others. Pass ``False`` to keep the
    mastoids as referenced signals in their own right.

    ``H`` here is anything linear in the EEG on its last axis -- the transfer function itself or
    the cross-spectrum it is formed from, in either the broadband ``(F, Cy)`` or the subband
    ``(F, n_subbands, Cy)`` layout. It is **not** valid for ``S_yy``, which is a squared quantity.
    """
    from trfs.eeg import MASTOID_CHANNELS

    names = list(channel_names)

    if reference == "avg":
        scalp = [i for i, name in enumerate(names) if name not in MASTOID_CHANNELS]
        referenced = H - H[..., scalp].mean(axis=-1, keepdims=True)
    elif reference == "FCz":
        if "FCz" not in names:
            raise ValueError("FCz is not on this channel axis; it may already be the reference")
        referenced = H - H[..., names.index("FCz")][..., None]
    elif reference == "linked_mastoids":
        present = [name for name in MASTOID_CHANNELS if name in names]
        if len(present) != len(MASTOID_CHANNELS):
            raise ValueError(
                f"linked_mastoids needs {list(MASTOID_CHANNELS)} on the channel axis; found "
                f"{present}. Either this file was fitted without them, or this participant's "
                "recordings carried none -- check `channel_durations` in the file."
            )
        index = [names.index(name) for name in present]
        referenced = H - H[..., index].mean(axis=-1, keepdims=True)
    else:
        raise ValueError(f"Unknown reference {reference!r}")

    drop = {"FCz"} if reference == "FCz" else set()
    if drop_mastoids:
        drop.update(MASTOID_CHANNELS)
    keep = [i for i, name in enumerate(names) if name not in drop]
    return referenced[..., keep], [names[i] for i in keep]
