"""The thesis' RARE dataset pipeline, as ``create_dataset.m`` actually runs it.

The manuscript describes framing, windowing and an energy gate at
``param.threshold`` dB. The code that produced the published scores also adds
white Gaussian noise at 1% of each signal's peak amplitude, to every signal,
train and test alike (``create_dataset.m`` l.44-45). That step is not in the
manuscript but it is not cosmetic:

- it lifts the noise floor to 20*log10(frame_length*(0.01*peak)^2) ~ -26 dB
  for frame_length=512, well above the -60 dB gate, so the gate is essentially
  inert. Measured on ESC-10 at the thesis' parameters: 1.5% of frames gated
  with the noise, 23.8% without;
- it leaves no bin of the normalized spectrum at zero, so log(theta) is finite
  everywhere without an artificial floor. This is the data-dependent version of
  ``compute_feature.m``'s ``feature(feature == 0) = eps``.

Two quirks of the original are reproduced verbatim, since the point of this
module is to reproduce its numbers:

- the frame count is ``floor((n - frame_length) / shift_length)`` computed on
  the length *before* the l.45 noise pads are prepended (l.43, l.49), so the
  last partial frame of the signal is dropped;
- frames are cut from the *padded* signal, so with shift_length == frame_length
  the first frame is entirely leading noise pad.

Thesis: chapter2RTASC.tex, table:q reports 64.7 (2.9) on ESC-10 at these
parameters. See python/README.md for what this pipeline reproduces.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gasm.common.power_spectrum import complex_spectrum, normalized_power_spectrum

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

NOISE_VARIANCE = 0.01  # create_dataset.m l.16
THRESHOLD_DB = -60.0  # main_monophonic.m l.26: param.threshold

_TINY = 1e-300  # keeps log10 finite on an exactly-silent frame


def _noise(signal: FloatArray, size: int, variance: float, rng: np.random.Generator) -> FloatArray:
    """create_dataset.m: variance * max(abs(sound)) * randn(size, 1)."""
    scale: float = variance * float(np.abs(signal).max())
    return scale * rng.standard_normal(size)


def prepare_clip(
    signal: FloatArray,
    frame_length: int,
    shift_length: int,
    n_freq_bins: int,
    rng: np.random.Generator,
    noise_variance: float = NOISE_VARIANCE,
    threshold_db: float = THRESHOLD_DB,
) -> tuple[FloatArray, BoolArray]:
    """One mono clip -> (normalized power spectra, keep mask), per create_dataset.m.

    ``signal`` must already be 1-D (downmix stereo yourself, as l.33-35 does).
    ``noise_variance=0.0`` skips the additive noise and gives the manuscript's
    pipeline instead of the executed one. The keep mask is the energy gate
    (``identification.m`` l.45): frames below ``threshold_db`` are False, to be
    dropped by the caller rather than silently zeroed.
    """
    signal = signal - signal.mean()  # l.36
    if noise_variance > 0.0:
        n_aux = len(signal)
        if n_aux <= frame_length + shift_length:  # l.40-42
            pad = shift_length + frame_length - n_aux
            signal = np.concatenate([signal, _noise(signal, pad, noise_variance, rng)])
        n_aux = len(signal)  # l.43: fixed here, before the l.45 pads
        signal = signal + _noise(signal, n_aux, noise_variance, rng)  # l.44
        signal = np.concatenate(  # l.45
            [
                _noise(signal, shift_length, noise_variance, rng),
                signal,
                _noise(signal, shift_length, noise_variance, rng),
            ]
        )
    else:
        n_aux = len(signal)

    n_frames = (n_aux - frame_length) // shift_length  # l.49
    frames = np.stack(
        [signal[b * shift_length : b * shift_length + frame_length] for b in range(n_frames)]
    ) * np.hanning(frame_length)  # l.15, l.52

    # identification.m l.45: 20*log10 of a power, as written (not 10*log10).
    energy_db = 20.0 * np.log10(np.maximum((frames**2).sum(axis=1), _TINY))
    spectra = normalized_power_spectrum(complex_spectrum(frames, n_freq_bins))
    return spectra, energy_db >= threshold_db
