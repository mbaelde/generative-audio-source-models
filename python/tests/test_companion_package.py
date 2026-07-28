"""The appendix B.1/B.2 mixtures live in nongaussian-mixtures, not here.

These are not tests of the estimators (the package tests those); they check that
the declared dependency really provides what this repo's README says it does, on
the two models the thesis' appendices describe.
"""

import numpy as np
from nongaussian_mixtures import BinnedGaussianMixture, DirichletMixture
from scipy.stats import dirichlet


def test_dirichlet_mixture_separates_two_modes() -> None:
    """Appendix B.2: EM fit of a Dirichlet mixture."""
    rng = np.random.default_rng(7)
    alpha_a = np.array([10.0, 1.0, 1.0])
    alpha_b = np.array([1.0, 1.0, 10.0])
    x = np.vstack(
        [
            dirichlet.rvs(alpha_a, size=1000, random_state=rng),
            dirichlet.rvs(alpha_b, size=1000, random_state=rng),
        ]
    )

    model = DirichletMixture(n_components=2, random_state=0).fit(x)

    np.testing.assert_allclose(np.sort(model.weights_), [0.5, 0.5], atol=0.1)
    # each fitted component should be closer to one true mode than the other
    dist_to_a = np.linalg.norm(model.alphas_ - alpha_a, axis=1)
    dist_to_b = np.linalg.norm(model.alphas_ - alpha_b, axis=1)
    assert {np.argmin(dist_to_a), np.argmin(dist_to_b)} == {0, 1}


def test_binned_gaussian_mixture_recovers_mean_and_std() -> None:
    """Appendix B.1: GMM fitted to a histogram rather than to samples."""
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=3.0, scale=2.0, size=(20000, 1))
    edges = np.arange(-9.0, 15.5, 0.5)
    counts, _ = np.histogram(samples, bins=edges)
    centers = ((edges[:-1] + edges[1:]) / 2)[:, None]

    model = BinnedGaussianMixture(bin_width=0.5, random_state=0).fit(
        centers, counts=counts.astype(np.float64)
    )

    assert np.isclose(model.means_[0, 0], 3.0, atol=0.1)
    assert np.isclose(np.sqrt(model.variances_[0, 0]), 2.0, atol=0.1)
