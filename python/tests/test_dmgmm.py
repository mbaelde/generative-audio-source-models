import numpy as np

from gasm.rase import dmgmm
from gasm.rase.dmgmm import DMGMMSeparator


def _tight_source(mean: complex, freqbin: int, n: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    """n frames of a near-deterministic complex spectrum clustered around `mean`."""
    base = np.full(freqbin, mean)
    noise = rng.normal(scale=scale, size=(n, freqbin)) + 1j * rng.normal(scale=scale, size=(n, freqbin))
    return base + noise


def test_two_source_separation_recovers_well_separated_sources() -> None:
    rng = np.random.default_rng(0)
    freqbin = 4
    source1 = _tight_source(mean=5.0 + 0.0j, freqbin=freqbin, n=200, scale=0.01, rng=rng)
    source2 = _tight_source(mean=0.0 + 5.0j, freqbin=freqbin, n=200, scale=0.01, rng=rng)

    sep = DMGMMSeparator(n_components=[1, 1]).fit([source1, source2], random_state=0)

    mixed = (5.0 + 0.0j) + (0.0 + 5.0j) + np.zeros((1, freqbin))
    est1, est2 = sep.predict(mixed)

    np.testing.assert_allclose(est1[0], np.full(freqbin, 5.0 + 0.0j), atol=0.1)
    np.testing.assert_allclose(est2[0], np.full(freqbin, 0.0 + 5.0j), atol=0.1)
    np.testing.assert_allclose(est1[0] + est2[0], mixed[0], atol=1e-8)


def test_three_source_estimates_sum_back_to_mixture() -> None:
    rng = np.random.default_rng(1)
    freqbin = 3
    sources = [
        _tight_source(mean=5.0 + 0.0j, freqbin=freqbin, n=150, scale=0.01, rng=rng),
        _tight_source(mean=0.0 + 5.0j, freqbin=freqbin, n=150, scale=0.01, rng=rng),
        _tight_source(mean=-5.0 + 0.0j, freqbin=freqbin, n=150, scale=0.01, rng=rng),
    ]

    sep = DMGMMSeparator(n_components=[1, 1, 1]).fit(sources, random_state=0)

    mixed = np.full((1, freqbin), (5.0 + 0.0j) + (0.0 + 5.0j) + (-5.0 + 0.0j))
    estimates = sep.predict(mixed)

    total = sum(est[0] for est in estimates)
    np.testing.assert_allclose(total, mixed[0], atol=1e-8)
    np.testing.assert_allclose(estimates[0][0], np.full(freqbin, 5.0 + 0.0j), atol=0.2)
    np.testing.assert_allclose(estimates[1][0], np.full(freqbin, 0.0 + 5.0j), atol=0.2)
    np.testing.assert_allclose(estimates[2][0], np.full(freqbin, -5.0 + 0.0j), atol=0.2)


def test_two_source_separation_with_multiple_components_per_source() -> None:
    """Exercises the n1*n2 > 1 path in _regress/_combine (Cartesian product of
    GMM components), never hit by the single-component tests above."""
    rng = np.random.default_rng(3)
    freqbin = 4
    source1 = np.concatenate(
        [
            _tight_source(mean=5.0 + 0.0j, freqbin=freqbin, n=150, scale=0.01, rng=rng),
            _tight_source(mean=-5.0 + 0.0j, freqbin=freqbin, n=150, scale=0.01, rng=rng),
        ]
    )
    source2 = np.concatenate(
        [
            _tight_source(mean=0.0 + 5.0j, freqbin=freqbin, n=150, scale=0.01, rng=rng),
            _tight_source(mean=0.0 - 5.0j, freqbin=freqbin, n=150, scale=0.01, rng=rng),
        ]
    )

    sep = DMGMMSeparator(n_components=[2, 2]).fit([source1, source2], random_state=0)
    assert len(sep.sources_[0].weights) == 2
    assert len(sep.sources_[1].weights) == 2

    mixed = np.full((1, freqbin), (-5.0 + 0.0j) + (0.0 - 5.0j))
    est1, est2 = sep.predict(mixed)

    np.testing.assert_allclose(est1[0], np.full(freqbin, -5.0 + 0.0j), atol=0.1)
    np.testing.assert_allclose(est2[0], np.full(freqbin, 0.0 - 5.0j), atol=0.1)
    np.testing.assert_allclose(est1[0] + est2[0], mixed[0], atol=1e-8)


def test_batched_predict_matches_frame_by_frame() -> None:
    """The regression is vectorized over frames (one Cholesky per component pair
    for the whole block), so every frame must still get exactly the estimate it
    would get on its own. Guards the batching axes, not the model."""
    rng = np.random.default_rng(4)
    freqbin = 4
    source1 = np.concatenate(
        [
            _tight_source(mean=5.0 + 0.0j, freqbin=freqbin, n=150, scale=0.5, rng=rng),
            _tight_source(mean=-5.0 + 0.0j, freqbin=freqbin, n=150, scale=0.5, rng=rng),
        ]
    )
    source2 = _tight_source(mean=0.0 + 5.0j, freqbin=freqbin, n=300, scale=0.5, rng=rng)

    sep = DMGMMSeparator(n_components=[2, 1]).fit([source1, source2], random_state=0)

    mixed = source1[:8] + source2[:8]
    batched = sep.predict(mixed)
    per_frame = [sep.predict(mixed[i : i + 1]) for i in range(len(mixed))]

    for s, estimate in enumerate(batched):
        expected = np.concatenate([frame[s] for frame in per_frame])
        np.testing.assert_allclose(estimate, expected, atol=1e-10)


def test_score_samples_is_higher_for_typical_mixtures() -> None:
    rng = np.random.default_rng(2)
    freqbin = 4
    source1 = _tight_source(mean=5.0 + 0.0j, freqbin=freqbin, n=200, scale=0.01, rng=rng)
    source2 = _tight_source(mean=0.0 + 5.0j, freqbin=freqbin, n=200, scale=0.01, rng=rng)
    sep = DMGMMSeparator(n_components=[1, 1]).fit([source1, source2], random_state=0)

    typical = np.full((1, freqbin), (5.0 + 0.0j) + (0.0 + 5.0j))
    atypical = np.full((1, freqbin), 100.0 + 100.0j)

    scores = sep.score_samples(np.vstack([typical, atypical]))
    assert scores[0] > scores[1]


def test_diagonal_shortcut_matches_the_dense_solve(monkeypatch) -> None:
    """The diagonal path is a speedup and nothing else.

    An ablation that has to run at high freqbin fits diagonal covariances and
    hands them back widened into matrices, which _pair_terms and _regress then
    recognize to avoid a Cholesky of a diagonal matrix. Forcing the dense path
    back on through the recognizer is what makes this the same fitted model
    compared against itself rather than two differently-fitted ones.
    """
    rng = np.random.default_rng(5)
    freqbin = 4
    source1 = np.concatenate(
        [
            _tight_source(mean=5.0 + 0.0j, freqbin=freqbin, n=150, scale=0.5, rng=rng),
            _tight_source(mean=-5.0 + 0.0j, freqbin=freqbin, n=150, scale=0.5, rng=rng),
        ]
    )
    source2 = _tight_source(mean=0.0 + 5.0j, freqbin=freqbin, n=300, scale=0.5, rng=rng)

    sep = DMGMMSeparator(n_components=[2, 1]).fit([source1, source2], random_state=0)
    for source in sep.sources_:
        source.covariances = np.stack([np.diag(np.diag(cov)) for cov in source.covariances])

    mixed = source1[:6] + source2[:6]
    fast = sep.predict(mixed)
    monkeypatch.setattr(dmgmm, "_diagonal_variances", lambda covariances: None)
    dense = sep.predict(mixed)

    for shortcut, reference in zip(fast, dense):
        np.testing.assert_allclose(shortcut, reference, rtol=1e-9, atol=1e-9)
