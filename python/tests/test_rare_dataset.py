import numpy as np

from gasm.rare.dataset import prepare_clip

FRAME_LENGTH = 512
SHIFT_LENGTH = 512
N_FREQ_BINS = 257


def _burst_then_silence(n_frames: int, rng: np.random.Generator) -> np.ndarray:
    """One loud frame, then silence: every frame after the first is gated unless
    create_dataset.m's additive noise lifts the whole clip's floor. Long enough
    that the DC left by mean removal stays well under the gate on its own."""
    signal = np.zeros(n_frames * SHIFT_LENGTH)
    signal[:SHIFT_LENGTH] = rng.normal(size=SHIFT_LENGTH)
    return signal


def test_additive_noise_makes_the_energy_gate_inert() -> None:
    rng = np.random.default_rng(0)
    signal = _burst_then_silence(40, rng)

    _, keep_noisy = prepare_clip(signal, FRAME_LENGTH, SHIFT_LENGTH, N_FREQ_BINS, rng)
    _, keep_clean = prepare_clip(
        signal, FRAME_LENGTH, SHIFT_LENGTH, N_FREQ_BINS, rng, noise_variance=0.0
    )

    assert keep_noisy.all()
    assert keep_clean.sum() == 1  # only the burst survives without the noise floor


def test_additive_noise_leaves_no_zero_bin_so_log_theta_stays_finite() -> None:
    rng = np.random.default_rng(1)
    signal = _burst_then_silence(6, rng)

    spectra, _ = prepare_clip(signal, FRAME_LENGTH, SHIFT_LENGTH, N_FREQ_BINS, rng)

    assert np.all(spectra > 0.0)
    assert np.all(np.isfinite(np.log(spectra)))


def test_frame_count_matches_create_dataset_m() -> None:
    """l.49 computes N on the length *before* the l.45 pads, so the trailing
    partial frame is dropped: one fewer frame than plain framing would give."""
    rng = np.random.default_rng(2)
    n_samples = 10 * SHIFT_LENGTH
    signal = rng.normal(size=n_samples)

    spectra, keep = prepare_clip(signal, FRAME_LENGTH, SHIFT_LENGTH, N_FREQ_BINS, rng)

    expected = (n_samples - FRAME_LENGTH) // SHIFT_LENGTH
    assert len(spectra) == expected == len(keep)
    np.testing.assert_allclose(spectra.sum(axis=1), 1.0)
