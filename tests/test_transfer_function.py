import numpy as np
import pytest

from trfs.transfer_function import (
    WeightedMean,
    rereference_transfer_function,
    resolve_trf_dir,
    ridge_term,
    transfer_function,
    window_directory_name,
)

FS = 128.0
NPERSEG = 256
NOVERLAP = 192


def _filtered_pair(n_samples=40_000, seed=0, taps=None):
    """A stimulus-like input and an output that is a known FIR filtering of it."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_samples)
    if taps is None:
        taps = np.array([0.0, 0.5, -0.25, 0.75, 0.1])
    y = np.convolve(x, taps)[:n_samples]
    return x, y, taps


def test_recovers_a_known_filter():
    """H should be the filter's own frequency response, up to estimator noise."""
    x, y, taps = _filtered_pair()
    result = transfer_function(x, y, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    expected = np.fft.rfft(taps, n=NPERSEG)
    got = result.H[:, 0, 0]
    # Welch averaging leaves a few percent of scatter; the shape is what is being checked.
    assert np.allclose(got, expected, atol=0.05)
    assert result.freqs.shape == got.shape


def test_multichannel_shapes():
    x, y, _ = _filtered_pair()
    x2 = np.stack([x, np.roll(x, 7)], axis=1)           # (n, 2)
    y2 = np.stack([y, np.roll(y, 3), -y], axis=1)       # (n, 3)
    result = transfer_function(x2, y2, FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    assert result.H.shape == (NPERSEG // 2 + 1, 2, 3)
    assert result.Pxx.shape == (NPERSEG // 2 + 1, 2)
    assert result.Pyy.shape == (NPERSEG // 2 + 1, 3)


def test_unequal_lengths_truncate():
    """A shorter EEG side drops trailing windows rather than raising."""
    x, y, _ = _filtered_pair()
    full = transfer_function(x, y, FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    short = transfer_function(x, y[: len(y) // 2], FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    assert short.n_frames < full.n_frames
    assert short.H.shape == full.H.shape


def test_ridge_shrinks_towards_zero():
    x, y, _ = _filtered_pair()
    plain = transfer_function(x, y, FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    damped = transfer_function(x, y, FS, nperseg=NPERSEG, noverlap=NOVERLAP, ridge=1.0)
    assert np.all(np.abs(damped.H) < np.abs(plain.H) + 1e-12)
    assert np.abs(damped.H).mean() < np.abs(plain.H).mean()


def test_ridge_term_zero_is_exactly_zero():
    Pxx = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(ridge_term(Pxx, 0.0), np.zeros((1, 2)))
    assert np.allclose(ridge_term(Pxx, 0.5), 0.5 * Pxx.mean(axis=0, keepdims=True))


# ---------------------------------------------------------------------------------------------
# Re-referencing
# ---------------------------------------------------------------------------------------------


def _referenced_fits(n_scalp=6):
    """Fit the same data under each reference directly, the way the pipeline would.

    The saved fit is the average reference *with mastoids kept*: the average is formed from the
    scalp channels alone and then subtracted from A1/A2 as well.
    """
    rng = np.random.default_rng(1)
    n = 30_000
    x = rng.standard_normal(n)
    taps = rng.standard_normal((n_scalp + 2, 5)) * 0.4
    raw = np.stack([np.convolve(x, taps[c])[:n] for c in range(n_scalp + 2)], axis=1)
    # A common-mode component, which is what the references differ in how they remove.
    raw = raw + rng.standard_normal(n)[:, None] * 2.0

    names = [f"ch{i}" for i in range(n_scalp)] + ["A1", "A2"]
    names[2] = "FCz"
    scalp = list(range(n_scalp))

    saved = raw - raw[:, scalp].mean(axis=1, keepdims=True)       # avg, mastoids kept
    fcz = np.delete(raw - raw[:, [2]], [2, n_scalp, n_scalp + 1], axis=1)
    linked = np.delete(raw - raw[:, [n_scalp, n_scalp + 1]].mean(axis=1, keepdims=True),
                       [n_scalp, n_scalp + 1], axis=1)

    def fit(data):
        return transfer_function(x, data, FS, nperseg=NPERSEG, noverlap=NOVERLAP).H[:, 0, :]

    return fit(saved), fit(fcz), fit(linked), names


def test_rereference_to_fcz_matches_a_direct_fit():
    """The claim the script rests on: H re-references exactly as the EEG does."""
    H_saved, H_fcz_direct, _, names = _referenced_fits()
    H_fcz, kept = rereference_transfer_function(H_saved, names, "FCz")

    assert kept == [n for n in names if n not in ("FCz", "A1", "A2")]
    assert H_fcz.shape == H_fcz_direct.shape
    assert np.allclose(H_fcz, H_fcz_direct, rtol=1e-8, atol=1e-10)


def test_rereference_to_linked_mastoids_matches_a_direct_fit():
    """Reachable only because A1/A2 are kept in the fit; the common average cancels."""
    H_saved, _, H_linked_direct, names = _referenced_fits()
    H_linked, kept = rereference_transfer_function(H_saved, names, "linked_mastoids")

    assert kept == [n for n in names if n not in ("A1", "A2")]
    assert H_linked.shape == H_linked_direct.shape
    assert np.allclose(H_linked, H_linked_direct, rtol=1e-8, atol=1e-10)


def test_rereference_to_average_is_idempotent_on_the_scalp():
    H_saved, _, _, names = _referenced_fits()
    again, kept = rereference_transfer_function(H_saved, names, "avg")
    scalp = [i for i, n in enumerate(names) if n not in ("A1", "A2")]
    assert kept == [names[i] for i in scalp]
    assert np.allclose(again, H_saved[..., scalp], atol=1e-10)


def test_rereference_without_mastoids_refuses_linked():
    """A participant whose recordings carried none cannot reach that reference."""
    H_saved, _, _, names = _referenced_fits()
    scalp = [i for i, n in enumerate(names) if n not in ("A1", "A2")]
    with pytest.raises(ValueError, match="linked_mastoids needs"):
        rereference_transfer_function(H_saved[..., scalp], [names[i] for i in scalp],
                                      "linked_mastoids")


def test_rereference_can_keep_the_mastoids():
    H_saved, _, _, names = _referenced_fits()
    kept_axis = rereference_transfer_function(H_saved, names, "FCz", drop_mastoids=False)[1]
    assert "A1" in kept_axis and "FCz" not in kept_axis


def test_rereference_handles_a_subband_axis():
    """(F, n_subbands, Cy) re-references on the last axis like (F, Cy) does."""
    H_saved, _, _, names = _referenced_fits()
    stacked = np.stack([H_saved, 2 * H_saved], axis=1)      # (F, 2, Cy)
    got, kept = rereference_transfer_function(stacked, names, "FCz")
    expected, _ = rereference_transfer_function(H_saved, names, "FCz")
    assert got.shape == (H_saved.shape[0], 2, len(kept))
    assert np.allclose(got[:, 0, :], expected)


# ---------------------------------------------------------------------------------------------
# Pooling and layout
# ---------------------------------------------------------------------------------------------


def test_weighted_mean_weights_by_duration():
    pool = WeightedMean()
    pool.add(np.ones((2, 2)) * 1.0, 300.0)
    pool.add(np.ones((2, 2)) * 3.0, 100.0)
    assert pool.n == 2
    assert pool.weight == pytest.approx(400.0)
    assert np.allclose(pool.mean(), (300 * 1.0 + 100 * 3.0) / 400.0)


def test_weighted_mean_grows_and_weights_each_column_separately():
    """A recording without mastoids, then one with: the mastoid columns average over the one."""
    pool = WeightedMean()
    pool.add(np.ones((2, 3)), 100.0, [0, 1, 2])
    pool.add(np.ones((2, 5)) * 3.0, 100.0, [0, 1, 2, 3, 4])
    assert pool.total.shape == (2, 5)
    assert np.allclose(pool.column_weight, [200, 200, 200, 100, 100])
    assert np.allclose(pool.mean()[0], [2, 2, 2, 3, 3])


def test_weighted_mean_unfilled_column_is_nan():
    pool = WeightedMean(n_columns=3)
    pool.add(np.ones((2, 2)), 10.0, [0, 1])
    assert np.isnan(pool.mean()[:, 2]).all()


def test_weighted_mean_rejects_a_changed_shape():
    pool = WeightedMean()
    pool.add(np.ones((2, 2)), 1.0)
    with pytest.raises(ValueError, match="does not match"):
        pool.add(np.ones((3, 2)), 1.0)


def test_weighted_mean_needs_something():
    with pytest.raises(ValueError, match="nothing accumulated"):
        WeightedMean().mean()


def test_window_directory_name():
    assert window_directory_name(5.0, 0.8) == "window-5s_overlap-0.80"
    assert window_directory_name(10.0, 0.5) == "window-10s_overlap-0.50"
    assert window_directory_name(0.5, 0.25) == "window-0.5s_overlap-0.25"


def test_resolve_trf_dir_prefers_the_reference_entry(tmp_path):
    config = {
        "trf_dir": str(tmp_path / "fallback"),
        "references": {"FCz": {"trf_dir": str(tmp_path / "fcz")}},
    }
    assert resolve_trf_dir(config, "FCz", 5.0, 0.8) == tmp_path / "fcz" / "window-5s_overlap-0.80"
    assert resolve_trf_dir(config, "avg", 5.0, 0.8) == tmp_path / "fallback" / "window-5s_overlap-0.80"


def test_resolve_trf_dir_needs_a_configured_path():
    with pytest.raises(KeyError, match="No trf_dir configured"):
        resolve_trf_dir({}, "avg", 5.0, 0.8)
