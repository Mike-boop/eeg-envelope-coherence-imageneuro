import numpy as np
import pytest

pytest.importorskip("brian2hears")

from audio_features.factory import Brian2HearsGammatoneCochleogramExtractor
from audio_features.gammatone import erb_spaced_center_freqs
from audio_features.gammatone_brian2 import brian2hears_gammatone_filterbank_envelopes


def test_brian2hears_gammatone_filterbank_envelopes_shape_and_tuning():
    fs = 8000.0
    duration = 1.0
    n_channels = 8
    f_min, f_max = 200.0, 3000.0
    target_fs = 200.0

    center_freqs = erb_spaced_center_freqs(f_min, f_max, n_channels)
    tone_channel = n_channels // 2
    t = np.arange(int(duration * fs)) / fs
    tone = np.sin(2.0 * np.pi * center_freqs[tone_channel] * t)

    subband_envelopes, returned_center_freqs = brian2hears_gammatone_filterbank_envelopes(
        tone, fs, target_fs, n_channels=n_channels, f_min=f_min, f_max=f_max
    )

    assert np.allclose(returned_center_freqs, center_freqs)
    assert subband_envelopes.shape[0] == n_channels
    assert subband_envelopes.shape[1] == pytest.approx(duration * target_fs, abs=2)

    channel_energy = subband_envelopes.mean(axis=1)
    farthest_channel = 0 if tone_channel != 0 else n_channels - 1
    assert channel_energy[tone_channel] > channel_energy[farthest_channel]


def test_brian2hears_gammatone_has_small_causal_group_delay_not_the_fir_artifact():
    """Regression test for the gammatone.py FIR bug: fftconvolve(mode="same") introduced a
    spurious 30-50ms non-causal advance. brian2hears' IIR filter should show only its own true,
    much smaller (sub-20ms) causal group delay, with no non-causal (negative) shift at all."""
    fs = 30000.0
    center_freqs = erb_spaced_center_freqs(50.0, 8000.0, 8)

    n = int(fs * 0.5)
    impulse_idx = int(fs * 0.25)
    audio = np.zeros(n)
    audio[impulse_idx] = 1.0

    subband_envelopes, _ = brian2hears_gammatone_filterbank_envelopes(
        audio, fs, target_fs=fs, n_channels=8, f_min=50.0, f_max=8000.0
    )

    for channel in range(8):
        peak_idx = np.argmax(subband_envelopes[channel])
        shift_ms = (peak_idx - impulse_idx) / fs * 1000
        assert shift_ms >= 0.0  # causal: no non-causal advance
        assert shift_ms < 20.0  # true gammatone group delay is well under 20ms in this range


def test_brian2hears_gammatone_remove_group_delay_aligns_channel_peaks():
    """remove_group_delay=True shifts each channel back by its own analytic group delay, so an
    impulse's response peaks at (approximately) the same sample across channels instead of being
    spread out by the true, frequency-dependent gammatone group delay."""
    fs = 30000.0
    center_freqs = erb_spaced_center_freqs(50.0, 8000.0, 8)

    n = int(fs * 0.3)
    impulse_idx = int(fs * 0.15)
    audio = np.zeros(n)
    audio[impulse_idx] = 1.0

    default_env, _ = brian2hears_gammatone_filterbank_envelopes(
        audio, fs, target_fs=fs, n_channels=8, f_min=50.0, f_max=8000.0
    )
    aligned_env, _ = brian2hears_gammatone_filterbank_envelopes(
        audio, fs, target_fs=fs, n_channels=8, f_min=50.0, f_max=8000.0, remove_group_delay=True
    )

    default_spread_ms = (np.argmax(default_env, axis=1).max() - np.argmax(default_env, axis=1).min()) / fs * 1000
    aligned_spread_ms = (np.argmax(aligned_env, axis=1).max() - np.argmax(aligned_env, axis=1).min()) / fs * 1000

    assert default_spread_ms > 10.0  # the true causal group delay still spreads channels out
    assert aligned_spread_ms < default_spread_ms


def test_brian2hears_gammatone_chunking_matches_single_chunk_processing():
    """Regression test: Gammatone(...).process() materializes every channel of the whole signal
    at once, which OOM-kills long recordings (measured ~73GB peak RSS for a 20-minute file at 60
    channels). Processing in overlapping chunks bounds memory to one chunk's size instead of the
    whole recording's -- this checks that chunking doesn't change the numeric result."""
    fs = 30000.0
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(int(fs * 12)) * 0.02  # 12s, chunked into 4s windows below

    kwargs = dict(n_channels=8, f_min=50.0, f_max=8000.0)

    single_chunk_env, _ = brian2hears_gammatone_filterbank_envelopes(
        audio, fs, target_fs=1024.0, chunk_seconds=60.0, overlap_seconds=1.0, **kwargs
    )
    multi_chunk_env, _ = brian2hears_gammatone_filterbank_envelopes(
        audio, fs, target_fs=1024.0, chunk_seconds=4.0, overlap_seconds=1.0, **kwargs
    )

    assert single_chunk_env.shape == multi_chunk_env.shape
    assert np.allclose(single_chunk_env, multi_chunk_env, atol=1e-10)


def test_brian2hears_gammatone_cochleogram_extractor_matches_output_contract():
    fs = 8000.0
    t = np.arange(int(fs)) / fs
    tone = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)

    extractor = Brian2HearsGammatoneCochleogramExtractor(
        target_fs=100.0, n_channels=8, f_min=50.0, f_max=3000.0
    )
    features = extractor(tone, fs)

    assert set(features) == {"subband_envelopes", "gammatone_envelope", "gammatone_center_freqs"}
    assert features["subband_envelopes"].shape[0] == 8
    assert features["gammatone_envelope"].shape[0] == 1
    assert features["gammatone_center_freqs"].shape == (8,)
    assert np.allclose(features["gammatone_envelope"], features["subband_envelopes"].sum(axis=0, keepdims=True))
