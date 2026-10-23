import numpy as np
import pytest

from coherence.coherence import (
    _min_distance_permutation,
    coherence,
    coherence_from_prep,
    null_coherence_random_phase,
    null_coherence_random_phase_from_prep,
    null_coherence_shuffled_windows,
    null_coherence_shuffled_windows_from_prep,
    prepare_coherence,
    windowed_fft,
)


FS = 500.0
NPERSEG = 100
NOVERLAP = 50


def _coupled_signals(duration=40.0, noise_amplitude=0.3, seed=0):
    """A pair of signals sharing a common broadband noise component, each with independent
    noise added on top. Unlike a pure tone, this shared component has no periodicity that a
    whole-hop circular shift could accidentally realign, so it's a meaningful test case for
    the shift-based null."""
    rng = np.random.default_rng(seed)
    n = int(duration * FS)
    shared = rng.standard_normal(n)
    x = shared + noise_amplitude * rng.standard_normal(n)
    y = shared + noise_amplitude * rng.standard_normal(n)
    return x, y


def _mean_magnitude(Cxy):
    """Mean |Cxy| over all non-DC bins (DC is near-zero after per-window mean detrending, so
    dividing by its near-zero power makes that bin numerically unstable)."""
    return np.mean(np.abs(Cxy[1:]))


def test_windowed_fft_shapes_and_frequencies():
    x, _ = _coupled_signals(duration=10.0)
    spectrum = windowed_fft(x, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    expected_n_frames = 1 + (x.size - NPERSEG) // (NPERSEG - NOVERLAP)
    assert spectrum.n_frames == expected_n_frames
    assert spectrum.Xf.shape == (expected_n_frames, NPERSEG // 2 + 1, 1)
    assert spectrum.freqs[0] == 0.0
    assert spectrum.freqs[1] == pytest.approx(FS / NPERSEG)


def test_coherence_is_high_for_shared_signal_and_low_for_independent_noise():
    x, y = _coupled_signals()
    result = coherence(x, y, FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    assert _mean_magnitude(result.Cxy) > 0.7

    rng = np.random.default_rng(1)
    noise_x = rng.standard_normal(x.size)
    noise_y = rng.standard_normal(y.size)
    noise_result = coherence(noise_x, noise_y, FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    assert _mean_magnitude(noise_result.Cxy) < 0.3


def test_min_distance_permutation_respects_guard():
    n_frames = 200
    min_shift_frames = 15
    rng = np.random.default_rng(0)

    perm = _min_distance_permutation(n_frames, min_shift_frames, rng)
    positions = np.arange(n_frames)

    assert sorted(perm) == list(positions)  # still a valid permutation
    circular_distance = np.minimum((perm - positions) % n_frames, (positions - perm) % n_frames)
    assert np.all(circular_distance >= min_shift_frames)


@pytest.mark.parametrize("mode", ["shuffle", "global_shift"])
def test_null_coherence_shuffled_windows_shape_and_reduces_true_coupling(mode):
    x, y = _coupled_signals()
    n_null = 50

    null_cxy = null_coherence_shuffled_windows(
        x, y, FS, n_null=n_null, min_shift_seconds=2.0, mode=mode, rng=0, nperseg=NPERSEG, noverlap=NOVERLAP
    )

    assert null_cxy.shape == (n_null, NPERSEG // 2 + 1, 1, 1)

    true_magnitude = _mean_magnitude(coherence(x, y, FS, nperseg=NPERSEG, noverlap=NOVERLAP).Cxy)
    null_magnitude = np.mean([_mean_magnitude(draw) for draw in null_cxy])
    assert null_magnitude < 0.5 * true_magnitude


def test_null_coherence_shuffled_windows_is_seeded_and_raises():
    x, y = _coupled_signals()

    same_seed_a = null_coherence_shuffled_windows(x, y, FS, n_null=5, min_shift_seconds=2.0, rng=42)
    same_seed_b = null_coherence_shuffled_windows(x, y, FS, n_null=5, min_shift_seconds=2.0, rng=42)
    different_seed = null_coherence_shuffled_windows(x, y, FS, n_null=5, min_shift_seconds=2.0, rng=7)

    assert np.allclose(same_seed_a, same_seed_b)
    assert not np.allclose(same_seed_a, different_seed)

    with pytest.raises(ValueError, match="min_shift_seconds"):
        null_coherence_shuffled_windows(x, y, FS, n_null=5, min_shift_seconds=100.0)

    with pytest.raises(ValueError, match="mode"):
        null_coherence_shuffled_windows(x, y, FS, n_null=5, mode="not_a_real_mode")


def test_null_coherence_shuffled_windows_global_shift_respects_max_shift_seconds():
    """Every draw's shift must land within [min_shift_frames, max_shift_frames] — verified
    black-box by checking each draw against the small set of Cxy outcomes those shifts produce
    (via np.roll directly), and confirming an out-of-range shift's outcome never appears."""
    from coherence.coherence import _cross_spectrum, prepare_coherence

    x, y = _coupled_signals()
    min_shift_seconds, max_shift_seconds = 2.0, 2.5
    hop = NPERSEG - NOVERLAP
    min_shift_frames = int(np.ceil(min_shift_seconds * FS / hop))
    max_shift_frames = int(np.floor(max_shift_seconds * FS / hop))
    assert max_shift_frames > min_shift_frames  # otherwise this test isn't exercising a range

    prep = prepare_coherence(x, y, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    def _expected_for_shift(shift):
        Yf_rolled = np.roll(prep.y_spectrum.Xf, shift, axis=0)
        Sxy = _cross_spectrum(prep.Xf_t, Yf_rolled, prep.x_spectrum.scale, prep.x_spectrum.nperseg)
        return Sxy / prep.denom

    in_range_outcomes = [_expected_for_shift(s) for s in range(min_shift_frames, max_shift_frames + 1)]
    out_of_range_outcome = _expected_for_shift(prep.x_spectrum.n_frames // 2)

    null_cxy = null_coherence_shuffled_windows(
        x, y, FS, n_null=30, min_shift_seconds=min_shift_seconds, max_shift_seconds=max_shift_seconds,
        mode="global_shift", rng=0, nperseg=NPERSEG, noverlap=NOVERLAP,
    )

    for draw in null_cxy:
        assert any(np.allclose(draw, outcome) for outcome in in_range_outcomes)
        assert not np.allclose(draw, out_of_range_outcome)


def test_null_coherence_shuffled_windows_max_shift_seconds_validation():
    x, y = _coupled_signals()

    with pytest.raises(ValueError, match="max_shift_seconds"):
        null_coherence_shuffled_windows(
            x, y, FS, n_null=5, min_shift_seconds=2.0, max_shift_seconds=1.0, mode="global_shift"
        )

    with pytest.raises(ValueError, match="max_shift_seconds"):
        null_coherence_shuffled_windows(x, y, FS, n_null=5, max_shift_seconds=2.0, mode="shuffle")


def test_null_coherence_random_phase_shape_and_reduces_true_coupling():
    x, y = _coupled_signals()
    n_null = 50

    null_cxy = null_coherence_random_phase(
        x, y, FS, n_null=n_null, rng=0, nperseg=NPERSEG, noverlap=NOVERLAP
    )

    assert null_cxy.shape == (n_null, NPERSEG // 2 + 1, 1, 1)

    true_magnitude = _mean_magnitude(coherence(x, y, FS, nperseg=NPERSEG, noverlap=NOVERLAP).Cxy)
    null_magnitude = np.mean([_mean_magnitude(draw) for draw in null_cxy])
    assert null_magnitude < 0.5 * true_magnitude


def test_null_coherence_random_phase_is_seeded():
    x, y = _coupled_signals()

    same_seed_a = null_coherence_random_phase(x, y, FS, n_null=5, rng=42)
    same_seed_b = null_coherence_random_phase(x, y, FS, n_null=5, rng=42)
    different_seed = null_coherence_random_phase(x, y, FS, n_null=5, rng=7)

    assert np.allclose(same_seed_a, same_seed_b)
    assert not np.allclose(same_seed_a, different_seed)


def test_coherence_from_prep_matches_coherence():
    x, y = _coupled_signals()

    prep = prepare_coherence(x, y, FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    from_prep = coherence_from_prep(prep)
    direct = coherence(x, y, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    assert np.allclose(from_prep.Cxy, direct.Cxy)
    assert np.allclose(from_prep.Pxx, direct.Pxx)
    assert np.allclose(from_prep.Pyy, direct.Pyy)
    assert np.allclose(from_prep.freqs, direct.freqs)


def test_null_coherence_shuffled_windows_from_prep_matches_direct_call():
    x, y = _coupled_signals()

    prep = prepare_coherence(x, y, FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    from_prep = null_coherence_shuffled_windows_from_prep(prep, n_null=5, min_shift_seconds=2.0, rng=0)
    direct = null_coherence_shuffled_windows(x, y, FS, n_null=5, min_shift_seconds=2.0, rng=0, nperseg=NPERSEG, noverlap=NOVERLAP)

    assert np.allclose(from_prep, direct)


def test_null_coherence_random_phase_from_prep_matches_direct_call():
    x, y = _coupled_signals()

    prep = prepare_coherence(x, y, FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    from_prep = null_coherence_random_phase_from_prep(prep, n_null=5, rng=0)
    direct = null_coherence_random_phase(x, y, FS, n_null=5, rng=0, nperseg=NPERSEG, noverlap=NOVERLAP)

    assert np.allclose(from_prep, direct)


def test_prep_is_reused_across_coherence_and_null_calls():
    """The whole point of prepare_coherence: one prep feeds both the true coherence estimate
    and null draws, without redoing the windowing+FFT pass for each."""
    x, y = _coupled_signals()

    prep = prepare_coherence(x, y, FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    result = coherence_from_prep(prep)
    null_cxy = null_coherence_shuffled_windows_from_prep(prep, n_null=10, min_shift_seconds=2.0, rng=0)

    assert result.Cxy.shape == (NPERSEG // 2 + 1, 1, 1)
    assert null_cxy.shape == (10, NPERSEG // 2 + 1, 1, 1)
    assert _mean_magnitude(result.Cxy) > np.mean([_mean_magnitude(draw) for draw in null_cxy])
