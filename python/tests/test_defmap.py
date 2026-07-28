import numpy as np

from gasm.rase.defmap import DefMAPSeparator, apply_deformation, solve_amplitude_deformation, solve_complex_deformation


def test_solve_amplitude_deformation_exact_when_mix_equals_sum() -> None:
    rng = np.random.default_rng(0)
    amplitude1 = rng.uniform(0.1, 2.0, size=5)
    amplitude2 = rng.uniform(0.1, 2.0, size=5)
    amplitude_mix = amplitude1 + amplitude2

    transform1, transform2 = solve_amplitude_deformation(amplitude1, amplitude2, amplitude_mix)

    np.testing.assert_allclose(transform1 * amplitude1, amplitude1, atol=1e-10)
    np.testing.assert_allclose(transform2 * amplitude2, amplitude2, atol=1e-10)


def test_solve_amplitude_deformation_satisfies_constraint() -> None:
    rng = np.random.default_rng(1)
    amplitude1 = rng.uniform(0.1, 2.0, size=6)
    amplitude2 = rng.uniform(0.1, 2.0, size=6)
    amplitude_mix = rng.uniform(0.1, 4.0, size=6)

    transform1, transform2 = solve_amplitude_deformation(amplitude1, amplitude2, amplitude_mix)

    np.testing.assert_allclose(transform1 * amplitude1 + transform2 * amplitude2, amplitude_mix, atol=1e-10)


def test_solve_amplitude_deformation_falls_back_to_identity_on_silent_bin() -> None:
    amplitude1 = np.array([0.0, 1.0])
    amplitude2 = np.array([0.0, 1.0])
    amplitude_mix = np.array([0.0, 2.0])

    transform1, transform2 = solve_amplitude_deformation(amplitude1, amplitude2, amplitude_mix)

    assert transform1[0] == 1.0
    assert transform2[0] == 1.0


def test_solve_complex_deformation_falls_back_on_silent_bin() -> None:
    spec1 = np.array([0.0 + 0.0j, 1.0 + 1.0j])
    spec2 = np.array([0.0 + 0.0j, 1.0 + 1.0j])
    spec_mix = np.array([0.0 + 0.0j, 2.0 + 2.0j])

    transform1, transform2 = solve_complex_deformation(spec1, spec2, spec_mix)

    assert transform1[0] == 1.0 + 0.0j
    assert transform2[0] == 1.0 + 0.0j


def test_solve_complex_deformation_recovers_real_part_exactly_when_mix_equals_sum() -> None:
    rng = np.random.default_rng(2)
    spec1 = rng.uniform(0.1, 2.0, size=5) + 1j * rng.uniform(0.1, 2.0, size=5)
    spec2 = rng.uniform(0.1, 2.0, size=5) + 1j * rng.uniform(0.1, 2.0, size=5)
    spec_mix = spec1 + spec2

    transform1, transform2 = solve_complex_deformation(spec1, spec2, spec_mix)
    est1 = apply_deformation(transform1, spec1)
    est2 = apply_deformation(transform2, spec2)

    np.testing.assert_allclose(est1.real, spec1.real, atol=1e-10)
    np.testing.assert_allclose(est2.real, spec2.real, atol=1e-10)
    np.testing.assert_allclose(est1.real + est2.real, spec_mix.real, atol=1e-10)


def test_defmap_separator_picks_matching_sound_indices() -> None:
    freqbin, n_frames = 4, 3
    sound1_a = np.full((n_frames, freqbin), 5.0 + 0.0j)
    sound1_b = np.full((n_frames, freqbin), 1.0 + 1.0j)
    sound2_a = np.full((n_frames, freqbin), 0.0 + 5.0j)
    sound2_b = np.full((n_frames, freqbin), 2.0 + 2.0j)
    library1 = np.stack([sound1_a, sound1_b])
    library2 = np.stack([sound2_a, sound2_b])

    sep = DefMAPSeparator().fit([library1, library2])

    mixed = sound1_a + sound2_a
    est1, est2 = sep.predict(mixed)

    np.testing.assert_allclose(est1, sound1_a, atol=1e-8)
    np.testing.assert_allclose(est2, sound2_a, atol=1e-8)
    np.testing.assert_allclose(est1 + est2, mixed, atol=1e-8)
