"""Gammatone filterbank cochleogram and broadband envelope extraction.

The ERB-rate scale and filter shapes follow Slaney's gammatone approximation:
each channel is a cosine carrier windowed by a gamma-distribution envelope,
spaced uniformly in Equivalent Rectangular Bandwidth (ERB) rate so that
channel spacing matches the resolution of the human cochlea.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve, hilbert, lfilter, remez, resample_poly


def hz_to_erb_rate(frequency_hz: np.ndarray | float) -> np.ndarray | float:
    """Convert frequency in Hz to ERB-rate (Slaney's linear approximation)."""
    return 21.4 * np.log10(4.37e-3 * frequency_hz + 1.0)


def erb_rate_to_hz(erb_rate: np.ndarray | float) -> np.ndarray | float:
    """Convert ERB-rate back to frequency in Hz."""
    return (10 ** (erb_rate / 21.4) - 1.0) / 4.37e-3


def erb_width(frequency_hz: np.ndarray | float) -> np.ndarray | float:
    """Equivalent rectangular bandwidth (Hz) of the auditory filter centered at frequency_hz."""
    return 24.7 * (4.37e-3 * frequency_hz + 1.0)


def gammatone_group_delay(
    center_freq_hz: np.ndarray | float, order: int = 4, b_factor: float = 1.019
) -> np.ndarray | float:
    """Peak envelope delay (seconds) of an order-n gammatone filter at center_freq_hz.

    The gamma envelope t^(order-1) * exp(-2*pi*b*t) peaks at t = (order-1) / (2*pi*b), with
    b = b_factor * ERB(center_freq_hz). This is the true, physically-motivated group delay shared
    by any correctly-implemented gammatone filter (FIR or IIR) at that center frequency -- higher
    channels resonate and peak faster than lower ones.
    """
    bandwidth = b_factor * erb_width(center_freq_hz)
    return (order - 1) / (2.0 * np.pi * bandwidth)


def erb_spaced_center_freqs(f_min: float, f_max: float, n_channels: int) -> np.ndarray:
    """Return n_channels center frequencies spaced uniformly on the ERB-rate scale."""
    if n_channels < 2:
        raise ValueError("n_channels must be >= 2")
    erb_lo = hz_to_erb_rate(f_min)
    erb_hi = hz_to_erb_rate(f_max)
    erb_rates = np.linspace(erb_lo, erb_hi, n_channels)
    return erb_rate_to_hz(erb_rates)


def design_stimulus_lowpass_filter(
    fs: float, f_pass: float = 3500.0, f_stop: float = 9000.0, atten_db: float = 45.0
) -> np.ndarray:
    """Design an equiripple FIR low-pass filter: flat passband to f_pass, ~atten_db by f_stop.

    Matches the frequency response of the insert-tube transducer used for stimulus delivery,
    so filtering audio through this before feature extraction reflects what the cochlea saw.
    """
    nyquist = fs / 2.0
    bands = [0, f_pass, f_stop, nyquist]
    desired = [1, 0]
    trans_width = f_stop - f_pass
    numtaps = int(np.ceil(atten_db * fs / (22 * trans_width))) | 1  # force odd length
    return remez(numtaps, bands, desired, fs=fs)


def apply_stimulus_lowpass(
    audio: np.ndarray,
    fs: float,
    target_fs: float = 30000.0,
    f_pass: float = 3500.0,
    f_stop: float = 9000.0,
    atten_db: float = 45.0,
) -> tuple[np.ndarray, float]:
    """Low-pass filter audio to the tube-delivery response, then resample to target_fs.

    Resampling down from a high native rate (e.g. 44.1/48 kHz) to target_fs also keeps the
    downstream gammatone convolutions (one FFT per channel) from blowing up in memory.
    """
    audio = np.asarray(audio, dtype=float)
    taps = design_stimulus_lowpass_filter(fs, f_pass=f_pass, f_stop=f_stop, atten_db=atten_db)
    filtered = lfilter(taps, [1.0], audio)
    resampled = resample_poly(filtered, int(round(target_fs)), int(round(fs)))
    return resampled, float(target_fs)


def gammatone_filterbank_envelopes(
    audio: np.ndarray,
    fs: float,
    target_fs: float,
    n_channels: int = 64,
    f_min: float = 50.0,
    f_max: float | None = 8000.0,
    order: int = 4,
    ir_duration: float = 0.100,
    compression: float = 0.6,
    remove_group_delay: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Cochleogram: bandpass a mono waveform through an ERB-spaced gammatone filterbank.

    Each subband is half-wave rectified and power-law compressed (a coarse model of hair-cell
    transduction) before being resampled to target_fs.

    By default (remove_group_delay=False), each channel is cropped with fftconvolve(mode="same"),
    which centers the output on the impulse response array's geometric midpoint (ir_duration / 2)
    rather than each channel's actual, frequency-dependent peak -- this introduces a spurious
    ~30-50ms non-causal advance on top of the filter's true, much smaller group delay, and that
    advance itself varies by channel. Set remove_group_delay=True to instead crop each channel at
    its own analytic peak (see gammatone_group_delay), so every channel's response to an impulse
    peaks at (approximately) the same sample -- the right alignment when you're going to sum
    across the channel axis and don't want the summed envelope smeared by channel-dependent delay.

    Returns
    -------
    subband_envelopes : (n_channels, n_samples_out) ndarray
    center_freqs : (n_channels,) ndarray, Hz
    """
    audio = np.asarray(audio, dtype=float)
    if audio.ndim != 1:
        raise ValueError("audio must be a mono 1D waveform")
    if f_max is None:
        f_max = 0.95 * fs / 2.0

    center_freqs = erb_spaced_center_freqs(f_min, f_max, n_channels)

    ir_len = int(round(ir_duration * fs))
    if ir_len < 2:
        raise ValueError("ir_duration is too short for the given sample rate")
    t = np.arange(ir_len) / fs

    subbands = np.empty((n_channels, audio.size), dtype=float)
    for channel, center_freq in enumerate(center_freqs):
        bandwidth = 1.019 * erb_width(center_freq)
        gamma_envelope = (t ** (order - 1)) * np.exp(-2.0 * np.pi * bandwidth * t)
        impulse_response = gamma_envelope * np.cos(2.0 * np.pi * center_freq * t)
        impulse_response = impulse_response / (np.max(np.abs(impulse_response)) + 1e-12)

        if remove_group_delay:
            delay_samples = int(round(gammatone_group_delay(center_freq, order=order) * fs))
            full = fftconvolve(audio, impulse_response, mode="full")
            subbands[channel] = full[delay_samples : delay_samples + audio.size]
        else:
            subbands[channel] = fftconvolve(audio, impulse_response, mode="same")

    subbands = np.maximum(subbands, 0.0)
    subbands = np.power(subbands, compression)
    subband_envelopes = resample_poly(subbands, int(round(target_fs)), int(round(fs)), axis=-1)

    return subband_envelopes, center_freqs


def broadband_hilbert_envelope(audio: np.ndarray, fs: float, target_fs: float) -> np.ndarray:
    """Broadband temporal envelope via the analytic-signal (Hilbert transform) magnitude."""
    audio = np.asarray(audio, dtype=float)
    if audio.ndim != 1:
        raise ValueError("audio must be a mono 1D waveform")
    envelope = np.abs(hilbert(audio))
    return resample_poly(envelope, int(round(target_fs)), int(round(fs)))
