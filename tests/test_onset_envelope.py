import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from trfs.onset_envelope import (
    lowpass_envelope,
    onset_envelope,
    onset_envelope_bank,
    resolve_onset_trf_dir,
)

FS = 1024.0


def test_lowpass_is_zero_phase():
    """The claim latency rests on: an impulse comes back centred where it went in."""
    n, centre = 8192, 4096
    impulse = np.zeros(n)
    impulse[centre] = 1.0
    response = lowpass_envelope(impulse, FS, 4.0)

    offsets = np.arange(1, 2000)
    assert np.allclose(response[centre + offsets], response[centre - offsets], atol=1e-12)
    assert int(np.argmax(response)) == centre


def test_causal_filter_would_have_delayed_it():
    """Why filtfilt and not a single pass -- the delay it removes is the size of the effect."""
    n, centre = 8192, 4096
    impulse = np.zeros(n)
    impulse[centre] = 1.0
    causal = sosfilt(butter(3, 4.0, btype="low", fs=FS, output="sos"), impulse)
    delay_ms = (int(np.argmax(causal)) - centre) / FS * 1000
    assert delay_ms > 50  # measured 82 ms at this cutoff and order


def test_onset_keeps_length_and_is_non_negative():
    rng = np.random.default_rng(0)
    envelope = np.abs(rng.standard_normal(4096).cumsum())
    onset = onset_envelope(envelope, FS, 8.0)
    assert onset.shape == envelope.shape
    assert (onset >= 0).all()
    assert onset[0] == 0.0  # the prepended sample, not an invented step


def test_onset_marks_rises_not_falls():
    """A triangular envelope: the feature must live on the way up and be zero on the way down."""
    n = 4096
    ramp = np.concatenate([np.linspace(0, 1, n // 2), np.linspace(1, 0, n // 2)])
    onset = onset_envelope(ramp, FS, 20.0)
    rising, falling = onset[: n // 2], onset[n // 2:]
    # Edges excluded: the filter's transient at the very start, and the turning point itself.
    assert rising[100:-100].min() > 0
    assert np.allclose(falling[100:-100], 0.0)


def test_onset_scales_per_second_not_per_sample():
    """A known slope comes back as that slope, in units per second."""
    seconds = 4.0
    slope = 3.0
    ramp = slope * np.arange(int(seconds * FS)) / FS
    onset = onset_envelope(ramp, FS, 20.0)
    middle = onset[len(onset) // 4: -len(onset) // 4]
    assert np.allclose(middle, slope, rtol=1e-3)


def test_bank_matches_one_at_a_time():
    rng = np.random.default_rng(1)
    envelope = np.abs(rng.standard_normal(4096).cumsum())
    cutoffs = np.array([4.0, 10.0, 20.0])
    bank = onset_envelope_bank(envelope, FS, cutoffs)
    assert bank.shape == (3, envelope.size)
    for row, cutoff in enumerate(cutoffs):
        assert np.allclose(bank[row], onset_envelope(envelope, FS, float(cutoff)))


def test_cutoff_must_be_below_nyquist():
    with pytest.raises(ValueError, match="cutoff must be in"):
        lowpass_envelope(np.zeros(128), FS, FS)


def test_resolve_onset_trf_dir_prefers_the_reference_entry(tmp_path):
    config = {
        "onset_trf_dir": str(tmp_path / "fallback"),
        "references": {"FCz": {"onset_trf_dir": str(tmp_path / "fcz")}},
    }
    assert resolve_onset_trf_dir(config, "FCz", 5.0, 0.8) == (
        tmp_path / "fcz" / "window-5s_overlap-0.80")
    assert resolve_onset_trf_dir(config, "avg", 5.0, 0.8) == (
        tmp_path / "fallback" / "window-5s_overlap-0.80")


def test_resolve_onset_trf_dir_needs_a_configured_path():
    with pytest.raises(KeyError, match="No onset_trf_dir configured"):
        resolve_onset_trf_dir({}, "avg", 5.0, 0.8)
