"""Average-referenced EEG that keeps the mastoids, so every reference stays reachable later.

``coherence.pipeline.load_referenced_eeg`` picks the 64 scalp electrodes before referencing, which
drops A1/A2 -- they are ``misc``-typed in these recordings. That is right for the coherence
pipeline, where each reference is computed in its own full pass and the mastoids are never wanted
in the output.

It is wrong for transfer functions, which are fitted once and re-referenced afterwards. ``H`` is
linear in the EEG, so any reference that is a linear combination of the channels the file carries
can be reached by mixing ``H``'s channel axis -- but only those. Dropping A1/A2 would put the
linked-mastoid reference permanently out of reach of an average-referenced fit.

So the average here is still formed **from the 64 standard channels only** -- the mastoids must not
influence the reference they are later used to define -- and is then subtracted from the mastoids
as well. Writing the scalp channels as ``Y``, the mastoids as ``A``, and ``m`` for the mean over
the 64:

    saved:  Y - m,  A - m
    then:   (Y - m) - ((A1 - m) + (A2 - m)) / 2  =  Y - (A1 + A2) / 2

which is the linked-mastoid reference exactly, with ``m`` cancelling. The same cancellation is what
makes the FCz reference reachable.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np

# Misc-typed in these recordings, which is why picking "eeg" loses them.
MASTOID_CHANNELS = ("A1", "A2")


def load_avg_referenced_eeg(fif_path: Path) -> tuple[np.ndarray, float, list[str]]:
    """``(data, fs, channel_names)`` average-referenced, with A1/A2 kept where they exist.

    The channel axis is the 64 standard electrodes in file order, followed by whichever mastoids
    the recording carries -- so it is 66 channels for most recordings and 64 for the 21 that have
    no mastoids. Callers must align on names rather than on position, since the width varies.

    Unlike the coherence pipeline's loader this never returns ``None``: a recording without
    mastoids still has a perfectly good average reference, it simply cannot contribute to the
    mastoid columns.
    """
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)

    types = dict(zip(raw.ch_names, raw.get_channel_types()))
    scalp = [name for name in raw.ch_names if types[name] == "eeg"]
    mastoids = [name for name in MASTOID_CHANNELS if name in raw.ch_names]
    if not scalp:
        raise ValueError(f"{fif_path.name}: no eeg-type channels")

    raw.pick(scalp + mastoids)
    data = raw.get_data()

    # The average is over the scalp channels alone. `raw.pick` above preserved the order we asked
    # for, so the scalp rows are the leading ones -- but index by name rather than trust that.
    order = list(raw.ch_names)
    scalp_rows = [order.index(name) for name in scalp]
    data = data - data[scalp_rows].mean(axis=0, keepdims=True)

    return data, float(raw.info["sfreq"]), order


def canonical_channel_order(names: list[str]) -> list[str]:
    """The scalp channels in file order, then the mastoids -- the axis outputs are aligned to."""
    scalp = [name for name in names if name not in MASTOID_CHANNELS]
    mastoids = [name for name in MASTOID_CHANNELS if name in names]
    return scalp + mastoids


# --------------------------------------------------------------------------------------------
# PSDs under the other references, without a second FFT
# --------------------------------------------------------------------------------------------

# What each reference subtracts, as weights over named channels. The average is already applied
# by `load_avg_referenced_eeg`, so these are the *further* subtractions that take the saved fit
# to each other reference -- exactly the ones `rereference_transfer_function` applies to S_xy.
REFERENCE_SUBTRACTIONS = {
    "linked_mastoids": {"A1": 0.5, "A2": 0.5},
    "FCz": {"FCz": 1.0},
}
# Channels the pipeline drops from each reference's output, so these PSDs land on the same
# channel axis the coherence pipeline produces: 63 under FCz, 64 under linked mastoids.
REFERENCE_DROPS = {
    "linked_mastoids": set(MASTOID_CHANNELS),
    "FCz": {"FCz", *MASTOID_CHANNELS},
}


def rereferenced_auto_power(Yf_t, own_power, weights, scale, nperseg):
    """Auto-power of ``Y - sum_i w_i Y_i`` for every channel, from an already-windowed spectrum.

    ``S_yy`` is a squared quantity, so unlike the cross-spectrum it cannot be re-referenced after
    the fact -- which is why it has to be computed here, while the windowed FFT is still in hand.
    It does *not* need a second FFT: the spectrum is linear in the EEG, so

        |Y - r|^2 = |Y|^2 - 2 Re(conj(r) Y) + |r|^2

    and each term is a reduction over frames of arrays already held. The middle one is the same
    batched matmul the cross-spectrum uses, so the cost is one extra cross-spectrum per reference
    rather than a whole transform -- and no ``(n_frames, F, Cy)`` temporary, which at a realistic
    trial length would be another 2.8 GB per reference.

    ``Yf_t`` is ``(F, n_frames, Cy)`` C-contiguous, ``own_power`` the ``(F, Cy)`` auto-power of
    the un-subtracted signal, and ``weights`` a ``(Cy,)`` vector picking out the reference
    combination. Returns ``(F, Cy)``, folded and scaled to match ``own_power``.
    """
    from coherence.coherence import _onesided_fold

    reference = Yf_t @ weights.astype(Yf_t.dtype)                       # (F, n_frames)
    cross = np.matmul(np.conj(reference)[:, None, :], Yf_t)[:, 0, :]    # (F, Cy)
    reference_power = np.sum(np.abs(reference) ** 2, axis=1)            # (F,)
    correction = (-2.0 * np.real(cross) + reference_power[:, None]) / scale
    return own_power + _onesided_fold(correction, nperseg)


def reference_weights(channel_names, reference):
    """``(Cy,)`` weights for ``reference``, or ``None`` if this recording cannot support it."""
    subtraction = REFERENCE_SUBTRACTIONS[reference]
    if not set(subtraction) <= set(channel_names):
        return None
    weights = np.zeros(len(channel_names))
    for name, weight in subtraction.items():
        weights[channel_names.index(name)] = weight
    return weights
