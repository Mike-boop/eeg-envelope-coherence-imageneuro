"""Feature factory: a registry of extractors that turn one audio waveform into named features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from .gammatone import apply_stimulus_lowpass, broadband_hilbert_envelope, gammatone_filterbank_envelopes


class FeatureExtractor(Protocol):
    name: str

    def __call__(self, audio: np.ndarray, fs: float) -> dict[str, np.ndarray]: ...


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a WAV file as a mono float64 waveform in [-1, 1] and its sample rate."""
    fs, samples = wavfile.read(str(path))
    samples = np.asarray(samples)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if np.issubdtype(samples.dtype, np.integer):
        samples = samples.astype(np.float64) / float(np.iinfo(samples.dtype).max)
    else:
        samples = samples.astype(np.float64)
    return samples, int(fs)


@dataclass
class StimulusLowpassFilter:
    """Low-pass filters raw audio to the insert-tube transducer response and resamples it down.

    Applied once, up front, so every extractor below runs on the same tube-filtered signal
    (what the cochlea actually saw) instead of each one re-deriving its own copy.
    """

    target_fs: float = 30000.0
    f_pass: float = 3500.0
    f_stop: float = 9000.0
    atten_db: float = 45.0

    def __call__(self, audio: np.ndarray, fs: float) -> tuple[np.ndarray, float]:
        return apply_stimulus_lowpass(
            audio, fs, target_fs=self.target_fs, f_pass=self.f_pass, f_stop=self.f_stop, atten_db=self.atten_db
        )


@dataclass
class GammatoneCochleogramExtractor:
    """Runs the gammatone filterbank over the (already tube-filtered) audio."""

    target_fs: float = 1024.0
    n_channels: int = 60
    f_min: float = 50.0
    f_max: float = 8000.0
    order: int = 4
    ir_duration: float = 0.100
    compression: float = 0.6
    remove_group_delay: bool = False
    name: str = "gammatone_cochleogram"

    def __call__(self, audio: np.ndarray, fs: float) -> dict[str, np.ndarray]:
        subband_envelopes, center_freqs = gammatone_filterbank_envelopes(
            audio,
            fs,
            self.target_fs,
            n_channels=self.n_channels,
            f_min=self.f_min,
            f_max=self.f_max,
            order=self.order,
            ir_duration=self.ir_duration,
            compression=self.compression,
            remove_group_delay=self.remove_group_delay,
        )
        return {
            "subband_envelopes": subband_envelopes,
            "gammatone_envelope": subband_envelopes.sum(axis=0, keepdims=True),
            "gammatone_center_freqs": center_freqs,
        }


@dataclass
class Brian2HearsGammatoneCochleogramExtractor:
    """Faithful port of the accou2023 dataset's actual gammatone cochleogram pipeline, via
    brian2hears' IIR Gammatone filter rather than GammatoneCochleogramExtractor's FIR
    approximation (see gammatone_brian2.py for why that matters). Requires the optional
    ``gammatone-brian2`` dependency group; imported lazily so the default pipeline doesn't need it.
    """

    target_fs: float = 1024.0
    n_channels: int = 60
    f_min: float = 50.0
    f_max: float = 8000.0
    b: float = 1.019
    compression: float = 0.6
    remove_group_delay: bool = False
    chunk_seconds: float = 120.0
    overlap_seconds: float = 1.0
    name: str = "gammatone_cochleogram"

    def __call__(self, audio: np.ndarray, fs: float) -> dict[str, np.ndarray]:
        from .gammatone_brian2 import brian2hears_gammatone_filterbank_envelopes

        subband_envelopes, center_freqs = brian2hears_gammatone_filterbank_envelopes(
            audio,
            fs,
            self.target_fs,
            n_channels=self.n_channels,
            f_min=self.f_min,
            f_max=self.f_max,
            b=self.b,
            compression=self.compression,
            remove_group_delay=self.remove_group_delay,
            chunk_seconds=self.chunk_seconds,
            overlap_seconds=self.overlap_seconds,
        )
        return {
            "subband_envelopes": subband_envelopes,
            "gammatone_envelope": subband_envelopes.sum(axis=0, keepdims=True),
            "gammatone_center_freqs": center_freqs,
        }


@dataclass
class BroadbandEnvelopeExtractor:
    """Broadband temporal envelope via the Hilbert transform of the tube-filtered audio."""

    target_fs: float = 1024.0
    name: str = "broadband_envelope"

    def __call__(self, audio: np.ndarray, fs: float) -> dict[str, np.ndarray]:
        envelope = broadband_hilbert_envelope(audio, fs, self.target_fs)
        return {"broadband_envelope": envelope[None, :]}


@dataclass
class AudioExtractor:
    """Downsamples the tube-filtered audio itself, so the waveform behind the other features is saved too."""

    target_fs: float = 1024.0
    name: str = "audio"

    def __call__(self, audio: np.ndarray, fs: float) -> dict[str, np.ndarray]:
        resampled = resample_poly(audio, int(round(self.target_fs)), int(round(fs)))
        return {"audio_1024Hz": resampled[None, :]}


DEFAULT_LOWPASS = StimulusLowpassFilter()

DEFAULT_EXTRACTORS: tuple[FeatureExtractor, ...] = (
    GammatoneCochleogramExtractor(),
    BroadbandEnvelopeExtractor(),
    AudioExtractor(),
)


def extract_features(
    audio: np.ndarray,
    fs: float,
    extractors: tuple[FeatureExtractor, ...] = DEFAULT_EXTRACTORS,
    lowpass: StimulusLowpassFilter = DEFAULT_LOWPASS,
) -> dict[str, np.ndarray]:
    """Low-pass filter audio to the stimulus tube response once, then run each extractor over it."""
    filtered, filtered_fs = lowpass(audio, fs)

    features: dict[str, np.ndarray] = {}
    for extractor in extractors:
        result = extractor(filtered, filtered_fs)
        overlapping = set(result) & set(features)
        if overlapping:
            raise ValueError(f"Extractor '{extractor.name}' produced duplicate feature keys: {overlapping}")
        features.update(result)
    return features


def extract_features_from_wav(
    wav_path: str | Path,
    extractors: tuple[FeatureExtractor, ...] = DEFAULT_EXTRACTORS,
    lowpass: StimulusLowpassFilter = DEFAULT_LOWPASS,
) -> dict[str, np.ndarray]:
    """Load a WAV file and run the feature factory over it."""
    audio, fs = read_wav_mono(wav_path)
    return extract_features(audio, fs, extractors=extractors, lowpass=lowpass)
