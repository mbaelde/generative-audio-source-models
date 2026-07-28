import numpy as np

from gasm.common.gmm_binned import fit_gmm_binned


def _histogram_with_infinite_tails(x: np.ndarray, inner_edges: np.ndarray) -> np.ndarray:
    edges = np.concatenate([[-np.inf], inner_edges, [np.inf]])
    counts, _ = np.histogram(x, bins=edges)
    return edges, counts.astype(np.float64)


def test_single_component_recovers_mean_and_std() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(loc=3.0, scale=2.0, size=20000)
    edges, counts = _histogram_with_infinite_tails(x, np.linspace(-10, 16, 60))

    pi, mu, sigma2 = fit_gmm_binned(edges, counts, n_components=1, n_iter=30, rng=rng)

    assert np.isclose(pi[0], 1.0)
    assert np.isclose(mu[0], 3.0, atol=0.1)
    assert np.isclose(np.sqrt(sigma2[0]), 2.0, atol=0.1)


def test_two_components_recovers_separated_modes() -> None:
    rng = np.random.default_rng(1)
    x = np.concatenate(
        [rng.normal(loc=-5.0, scale=1.0, size=10000), rng.normal(loc=5.0, scale=1.0, size=10000)]
    )
    edges, counts = _histogram_with_infinite_tails(x, np.linspace(-12, 12, 80))

    pi, mu, sigma2 = fit_gmm_binned(edges, counts, n_components=2, n_iter=50, rng=rng)

    order = np.argsort(mu)
    np.testing.assert_allclose(mu[order], [-5.0, 5.0], atol=0.2)
    np.testing.assert_allclose(pi[order], [0.5, 0.5], atol=0.05)
    np.testing.assert_allclose(np.sqrt(sigma2[order]), [1.0, 1.0], atol=0.2)
