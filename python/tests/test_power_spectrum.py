import numpy as np

from gasm.common.power_spectrum import (
    complex_spectrum,
    frame_signal,
    mix_normalized_power_spectra,
    normalized_power_spectrum,
    total_power,
)


def test_frame_signal_shape_and_window() -> None:
    signal = np.ones(100)
    frames = frame_signal(signal, frame_length=20, shift_length=10)
    assert frames.shape == (9, 20)
    np.testing.assert_allclose(frames[0], np.hanning(20))


def test_normalized_power_spectrum_sums_to_one() -> None:
    rng = np.random.default_rng(0)
    frames = rng.normal(size=(5, 32))
    spec = complex_spectrum(frames, n_freq_bins=10)
    spectrum = normalized_power_spectrum(spec)
    np.testing.assert_allclose(spectrum.sum(axis=-1), 1.0)


def test_polyphonic_mix_matches_additivity_on_disjoint_bins() -> None:
    # Two orthogonal (disjoint-support) complex spectra: mixing must recover
    # the normalized power spectrum of their exact sum (decorrelation is exact here).
    spec_1 = np.array([3.0 + 0j, 0.0, 0.0, 0.0])
    spec_2 = np.array([0.0, 0.0, 4.0 + 0j, 0.0])
    mixed_direct = normalized_power_spectrum(spec_1 + spec_2)

    spectrum_1 = normalized_power_spectrum(spec_1)
    spectrum_2 = normalized_power_spectrum(spec_2)
    power_1 = total_power(spec_1)
    power_2 = total_power(spec_2)
    mixed_from_formula = mix_normalized_power_spectra(spectrum_1, power_1, spectrum_2, power_2)

    np.testing.assert_allclose(mixed_from_formula, mixed_direct)
    assert mixed_from_formula.sum() == 1.0


def test_normalized_power_spectrum_falls_back_to_uniform_on_silence() -> None:
    silent = np.zeros(4, dtype=np.complex128)
    spectrum = normalized_power_spectrum(silent)
    assert not np.any(np.isnan(spectrum))
    np.testing.assert_allclose(spectrum, np.full(4, 0.25))


def test_mix_normalized_power_spectra_falls_back_to_even_blend_when_both_silent() -> None:
    spectrum_1 = np.full(4, 0.25)
    spectrum_2 = np.array([0.0, 0.0, 1.0, 0.0])
    mixed = mix_normalized_power_spectra(spectrum_1, np.array(0.0), spectrum_2, np.array(0.0))
    assert not np.any(np.isnan(mixed))
    np.testing.assert_allclose(mixed, 0.5 * spectrum_1 + 0.5 * spectrum_2)
