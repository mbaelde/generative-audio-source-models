import numpy as np
from scipy.stats import dirichlet

from gasm.common.dirichlet_mixture import dirichlet_log_pdf, fit_dirichlet_mixture


def test_dirichlet_log_pdf_matches_scipy() -> None:
    rng = np.random.default_rng(0)
    alpha = np.array([2.0, 3.0, 1.5])
    x = dirichlet.rvs(alpha, size=5, random_state=rng)
    np.testing.assert_allclose(dirichlet_log_pdf(x, alpha), dirichlet.logpdf(x.T, alpha))


def test_fit_single_component_recovers_alpha() -> None:
    rng = np.random.default_rng(42)
    true_alpha = np.array([5.0, 2.0, 3.0])
    x = dirichlet.rvs(true_alpha, size=4000, random_state=rng)

    proportions, alpha = fit_dirichlet_mixture(x, n_components=1, n_iter=20, rng=rng)

    assert np.isclose(proportions[0], 1.0)
    np.testing.assert_allclose(alpha[0], true_alpha, rtol=0.15)


def test_fit_two_components_separates_clusters() -> None:
    rng = np.random.default_rng(7)
    alpha_a = np.array([10.0, 1.0, 1.0])
    alpha_b = np.array([1.0, 1.0, 10.0])
    x = np.vstack(
        [dirichlet.rvs(alpha_a, size=1000, random_state=rng), dirichlet.rvs(alpha_b, size=1000, random_state=rng)]
    )

    proportions, alpha = fit_dirichlet_mixture(x, n_components=2, n_iter=30, rng=rng)

    np.testing.assert_allclose(np.sort(proportions), [0.5, 0.5], atol=0.1)
    # each fitted component should be closer to one true mode than the other
    dist_to_a = np.linalg.norm(alpha - alpha_a, axis=1)
    dist_to_b = np.linalg.norm(alpha - alpha_b, axis=1)
    assert {np.argmin(dist_to_a), np.argmin(dist_to_b)} == {0, 1}
