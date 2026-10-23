import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from audio_features.factory import (
    AudioExtractor,
    BroadbandEnvelopeExtractor,
    GammatoneCochleogramExtractor,
    StimulusLowpassFilter,
    extract_features,
    extract_features_from_wav,
    read_wav_mono,
)
from audio_features.gammatone import (
    apply_stimulus_lowpass,
    broadband_hilbert_envelope,
    erb_rate_to_hz,
    erb_spaced_center_freqs,
    gammatone_filterbank_envelopes,
    gammatone_group_delay,
    hz_to_erb_rate,
)


def load_extract_features_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "feature_extraction" / "extract_features.py"
    )
    spec = importlib.util.spec_from_file_location("extract_features_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_erb_rate_hz_round_trip():
    frequencies = np.array([50.0, 500.0, 1000.0, 4000.0, 8000.0])
    round_tripped = erb_rate_to_hz(hz_to_erb_rate(frequencies))
    assert np.allclose(round_tripped, frequencies, rtol=1e-9)


def test_erb_spaced_center_freqs_endpoints_and_monotonic():
    center_freqs = erb_spaced_center_freqs(f_min=100.0, f_max=4000.0, n_channels=16)

    assert center_freqs.shape == (16,)
    assert center_freqs[0] == pytest.approx(100.0)
    assert center_freqs[-1] == pytest.approx(4000.0)
    assert np.all(np.diff(center_freqs) > 0)


def test_gammatone_filterbank_envelopes_shape_and_tuning():
    fs = 8000.0
    duration = 0.5
    n_channels = 8
    f_min, f_max = 200.0, 3000.0
    target_fs = 200.0

    center_freqs = erb_spaced_center_freqs(f_min, f_max, n_channels)
    tone_channel = n_channels // 2
    t = np.arange(int(duration * fs)) / fs
    tone = np.sin(2.0 * np.pi * center_freqs[tone_channel] * t)

    subband_envelopes, returned_center_freqs = gammatone_filterbank_envelopes(
        tone, fs, target_fs, n_channels=n_channels, f_min=f_min, f_max=f_max, ir_duration=0.02
    )

    assert np.allclose(returned_center_freqs, center_freqs)
    assert subband_envelopes.shape[0] == n_channels
    assert subband_envelopes.shape[1] == pytest.approx(duration * target_fs, abs=2)

    channel_energy = subband_envelopes.mean(axis=1)
    farthest_channel = 0 if tone_channel != 0 else n_channels - 1
    assert channel_energy[tone_channel] > channel_energy[farthest_channel]


def test_gammatone_group_delay_decreases_with_frequency():
    low, high = gammatone_group_delay(50.0), gammatone_group_delay(8000.0)
    assert low > high > 0.0


def test_gammatone_filterbank_remove_group_delay_aligns_channel_peaks():
    """Regression test: without remove_group_delay, fftconvolve(mode="same") crops around the
    impulse response array's fixed midpoint, not each channel's true (frequency-dependent) peak,
    so impulse-response peaks land at very different times across channels. remove_group_delay=True
    should crop each channel at its own analytic peak instead, collapsing that spread."""
    fs = 30000.0
    n = int(fs * 0.3)
    impulse_idx = int(fs * 0.15)
    audio = np.zeros(n)
    audio[impulse_idx] = 1.0

    kwargs = dict(n_channels=8, f_min=50.0, f_max=8000.0, ir_duration=0.100)

    default_env, _ = gammatone_filterbank_envelopes(audio, fs, fs, **kwargs)
    aligned_env, _ = gammatone_filterbank_envelopes(audio, fs, fs, remove_group_delay=True, **kwargs)

    default_spread_ms = (np.argmax(default_env, axis=1).max() - np.argmax(default_env, axis=1).min()) / fs * 1000
    aligned_spread_ms = (np.argmax(aligned_env, axis=1).max() - np.argmax(aligned_env, axis=1).min()) / fs * 1000

    assert default_spread_ms > 15.0  # the un-aligned crop-artifact spread is large
    assert aligned_spread_ms < default_spread_ms
    assert aligned_spread_ms < 10.0


def test_broadband_hilbert_envelope_matches_constant_amplitude():
    fs = 4000.0
    target_fs = 200.0
    amplitude = 0.7
    t = np.arange(int(2 * fs)) / fs
    tone = amplitude * np.sin(2.0 * np.pi * 100.0 * t)

    envelope = broadband_hilbert_envelope(tone, fs, target_fs)

    assert envelope.shape[0] == pytest.approx(2 * target_fs, abs=2)
    # Away from the transient edges, the analytic envelope tracks the amplitude.
    steady_state = envelope[int(0.1 * target_fs) : int(-0.1 * target_fs)]
    assert np.median(steady_state) == pytest.approx(amplitude, rel=0.1)


def test_apply_stimulus_lowpass_returns_target_fs_and_attenuates_highs():
    fs = 20000.0
    target_fs = 8000.0
    t = np.arange(int(fs)) / fs
    high_freq_tone = np.sin(2.0 * np.pi * 9200.0 * t)  # above the default f_stop=9000

    filtered, filtered_fs = apply_stimulus_lowpass(high_freq_tone, fs, target_fs=target_fs)

    assert filtered_fs == target_fs
    assert filtered.shape[0] == pytest.approx(target_fs, abs=2)
    assert np.std(filtered) < 0.1 * np.std(high_freq_tone)


def _small_gammatone_extractor() -> GammatoneCochleogramExtractor:
    """A cheap GammatoneCochleogramExtractor sized for 8kHz test audio (real usage sizes for ~44.1kHz)."""
    return GammatoneCochleogramExtractor(
        target_fs=100.0,
        n_channels=4,
        f_min=50.0,
        f_max=1500.0,
        ir_duration=0.02,
    )


def _small_lowpass_filter() -> StimulusLowpassFilter:
    """A StimulusLowpassFilter sized for 8kHz test audio (real usage sizes for ~44.1kHz)."""
    return StimulusLowpassFilter(target_fs=4000.0, f_pass=1200.0, f_stop=1800.0)


def _write_test_wav(path: Path, fs: int = 8000, duration: float = 0.5) -> None:
    t = np.arange(int(fs * duration)) / fs
    tone = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    pcm = np.round(tone * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write(str(path), fs, pcm)


def test_read_wav_mono_normalizes_to_unit_range(tmp_path):
    wav_path = tmp_path / "tone.wav"
    _write_test_wav(wav_path)

    audio, fs = read_wav_mono(wav_path)

    assert fs == 8000
    assert audio.ndim == 1
    assert np.max(np.abs(audio)) <= 1.0


def test_extract_features_merges_registered_extractors():
    fs = 8000.0
    t = np.arange(int(fs)) / fs
    tone = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)

    extractors = (
        _small_gammatone_extractor(),
        BroadbandEnvelopeExtractor(target_fs=100.0),
        AudioExtractor(target_fs=100.0),
    )
    features = extract_features(tone, fs, extractors=extractors, lowpass=_small_lowpass_filter())

    assert set(features) == {
        "subband_envelopes",
        "gammatone_envelope",
        "gammatone_center_freqs",
        "broadband_envelope",
        "audio_1024Hz",
    }
    assert features["subband_envelopes"].shape[0] == 4
    assert features["gammatone_envelope"].shape[0] == 1
    assert features["gammatone_center_freqs"].shape == (4,)
    assert features["broadband_envelope"].shape[0] == 1
    assert features["audio_1024Hz"].shape[0] == 1


def test_extract_features_raises_on_duplicate_keys():
    fs = 8000.0
    tone = np.zeros(int(fs))

    extractors = (
        BroadbandEnvelopeExtractor(target_fs=100.0),
        BroadbandEnvelopeExtractor(target_fs=100.0),
    )
    with pytest.raises(ValueError, match="duplicate feature keys"):
        extract_features(tone, fs, extractors=extractors, lowpass=_small_lowpass_filter())


def test_extract_features_broadband_envelope_uses_filtered_audio():
    """The broadband envelope must reflect the tube-filtered signal, not the raw input."""
    fs = 8000.0
    t = np.arange(int(fs)) / fs
    # A tone above the small test filter's f_stop=1800Hz should be attenuated away.
    high_freq_tone = np.sin(2.0 * np.pi * 3000.0 * t)

    filtered_features = extract_features(
        high_freq_tone,
        fs,
        extractors=(BroadbandEnvelopeExtractor(target_fs=100.0),),
        lowpass=_small_lowpass_filter(),
    )
    raw_envelope = broadband_hilbert_envelope(high_freq_tone, fs, 100.0)

    assert np.median(filtered_features["broadband_envelope"]) < 0.1 * np.median(raw_envelope)


def test_extract_features_from_wav_end_to_end(tmp_path):
    wav_path = tmp_path / "tone.wav"
    _write_test_wav(wav_path)

    extractors = (
        _small_gammatone_extractor(),
        BroadbandEnvelopeExtractor(target_fs=100.0),
        AudioExtractor(target_fs=100.0),
    )
    features = extract_features_from_wav(wav_path, extractors=extractors, lowpass=_small_lowpass_filter())

    assert features["subband_envelopes"].shape[0] == 4
    assert features["broadband_envelope"].ndim == 2
    assert features["audio_1024Hz"].ndim == 2
    assert features["audio_1024Hz"].shape[1] == pytest.approx(0.5 * 100.0, abs=2)


def test_resolve_configured_inputs_uses_project_config_directories(tmp_path):
    module = load_extract_features_module()

    config = {
        "raw_waves_dir": str(tmp_path / "raw_waves"),
        "audio_features_dir": str(tmp_path / "audio_features"),
    }

    audio_dir, output_dir = module.resolve_configured_inputs(config, root=tmp_path)

    assert audio_dir == (tmp_path / "raw_waves").resolve()
    assert output_dir == (tmp_path / "audio_features").resolve()
