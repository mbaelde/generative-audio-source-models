"""Dirichlet mixture EM fit (Newton-Raphson on the digamma system). Thesis:
annexes/calculus.tex, section B.2. Matches the "newton" branch of
matlab/common/Statistics/fit_mixture_dirichlet.m (the correct fitter -- its
sibling mixture_dirichlet_fit.m is broken and was already deleted from the
repo; the variational/Ma-Leijon branch is the alternative the thesis
rejected as too slow, not implemented here).

E-step: responsibility z_{i,m} propto prop_m * Dir(x_i; alpha_m), computed
    in log-space with a softmax (scipy.special.log_softmax) for stability.
M-step (proportions): prop_m = mean_i(z_{i,m}).
M-step (shape params): for each component m, solve for alpha_m the root of
    f_k(alpha) = psi(alpha_k) - psi(sum(alpha)) - weighted_mean_log_x_k = 0,
    the stationarity condition of the (responsibility-weighted) Dirichlet
    log-likelihood, via Newton-Raphson with Jacobian
    J = diag(trigamma(alpha)) - trigamma(sum(alpha)) * ones(D, D).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma, gammaln, log_softmax, polygamma

FloatArray = NDArray[np.float64]


def dirichlet_log_pdf(x: FloatArray, alpha: FloatArray) -> FloatArray:
    """log Dir(x_i; alpha) for each row x_i of ``x`` (shape (n, d)), alpha of shape (d,)."""
    log_norm = gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))
    return log_norm + np.sum((alpha - 1.0) * np.log(x), axis=-1)


def _fit_component_newton(
    x: FloatArray,
    weights: FloatArray,
    alpha: FloatArray,
    n_newton_iter: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Weighted-MLE Dirichlet shape parameters for one component, via Newton-Raphson."""
    n_dims = x.shape[1]
    weighted_mean_log_x = (weights @ np.log(x)) / np.sum(weights)
    for _ in range(n_newton_iter):
        f = digamma(alpha) - digamma(np.sum(alpha)) - weighted_mean_log_x
        jacobian = np.diag(polygamma(1, alpha)) - polygamma(1, np.sum(alpha))
        alpha = alpha - np.linalg.solve(jacobian, f)
        negative = alpha < 0
        if np.any(negative):
            alpha[negative] = rng.random(np.count_nonzero(negative))
    return alpha


def fit_dirichlet_mixture(
    x: FloatArray,
    n_components: int,
    n_iter: int = 50,
    n_newton_iter: int = 10,
    alpha_scale: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[FloatArray, FloatArray]:
    """EM fit of a Dirichlet mixture. Returns (proportions, alpha) of shapes (M,) and (M, D)."""
    if rng is None:
        rng = np.random.default_rng()
    n_samples, n_dims = x.shape
    proportions = np.full(n_components, 1.0 / n_components)
    alpha = alpha_scale * rng.random((n_components, n_dims))

    for _ in range(n_iter):
        log_pdf = np.stack([dirichlet_log_pdf(x, alpha[m]) for m in range(n_components)], axis=1)
        log_responsibilities = log_softmax(np.log(proportions) + log_pdf, axis=1)
        responsibilities = np.exp(log_responsibilities)

        proportions = responsibilities.mean(axis=0)
        for m in range(n_components):
            alpha[m] = _fit_component_newton(
                x, responsibilities[:, m], alpha[m], n_newton_iter, rng
            )

    return proportions, alpha
