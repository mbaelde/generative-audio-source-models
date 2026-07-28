"""Framing, DFT, normalized power spectrum. Thesis: chapter2RTASC.tex.

Monophonic feature (section 2.2, eq. spectrum, l.140-170):
    d_{s,a} = window * signal[a - obslength + (s-1)*shift : ... + framelength]
    spec_complex_{s,a} = rfft(d_{s,a})[:n_freq_bins]
    spectrum_{s,a} = |spec_complex_{s,a}|^2 / ||spec_complex_{s,a}||^2   (sums to 1)

Polyphonic mixing (section 2.4, eq. polyphonic_spectrum, l.400-441), from
complex-spectrum additivity under the decorrelation assumption:
    P_k = ||spec_complex_k||^2                    (power of source k alone)
    power_ratio = P_1 / (P_1 + P_2)
    spectrum = power_ratio * spectrum_1 + (1 - power_ratio) * spectrum_2
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def frame_signal(
    signal: FloatArray,
    frame_length: int,
    shift_length: int,
    window: FloatArray | None = None,
) -> FloatArray:
    """Split ``signal`` into overlapping, windowed frames of shape (n_frames, frame_length)."""
    if window is None:
        window = np.hanning(frame_length)
    n_frames = 1 + (len(signal) - frame_length) // shift_length
    frames = np.stack(
        [signal[i * shift_length : i * shift_length + frame_length] for i in range(n_frames)]
    )
    return frames * window


def complex_spectrum(frames: FloatArray, n_freq_bins: int | None = None) -> ComplexArray:
    """DFT of each frame (last axis), keeping only the first ``n_freq_bins`` bins."""
    spectrum = np.fft.rfft(frames, axis=-1)
    if n_freq_bins is not None:
        spectrum = spectrum[..., :n_freq_bins]
    return spectrum


def total_power(spec_complex: ComplexArray) -> FloatArray:
    """P = ||spec_complex||^2, per frame (last axis)."""
    power: FloatArray = np.sum(np.abs(spec_complex) ** 2, axis=-1)
    return power


def normalized_power_spectrum(spec_complex: ComplexArray) -> FloatArray:
    """eq. spectrum: |spec_complex|^2 / ||spec_complex||^2, per frame (sums to 1 over last axis).
    A totally silent frame (zero total power) has no defined direction to
    normalize towards; falls back to the uniform spectrum rather than NaN."""
    power: FloatArray = np.abs(spec_complex) ** 2
    total = power.sum(axis=-1, keepdims=True)
    safe_total = np.where(total > 0, total, 1.0)
    normalized: FloatArray = np.where(total > 0, power / safe_total, 1.0 / power.shape[-1])
    return normalized


def mix_normalized_power_spectra(
    spectrum_1: FloatArray,
    power_1: FloatArray,
    spectrum_2: FloatArray,
    power_2: FloatArray,
) -> FloatArray:
    """eq. polyphonic_spectrum: power_ratio * spectrum_1 + (1 - power_ratio) * spectrum_2.
    Both sources silent (zero total power) has no defined ratio; falls back
    to an even 50/50 blend rather than NaN."""
    total_power = power_1 + power_2
    safe_total_power = np.where(total_power > 0, total_power, 1.0)
    power_ratio = np.where(total_power > 0, power_1 / safe_total_power, 0.5)
    power_ratio = power_ratio[..., np.newaxis]
    return power_ratio * spectrum_1 + (1.0 - power_ratio) * spectrum_2
