"""EM for a GMM fitted from interval (binned) data. Thesis: annexes/calculus.tex,
section B.1 (McLachlan & Jones 1988). Backs the optional bGMM kernel (RARE's
experiments used the multinomial kernel; bGMM is secondary).

Data is r disjoint intervals (a_j, b_j] with observed counts n_j (a histogram).
Per-iteration E-step expectations (eq. A_{s,k,j}, using the CURRENT-iteration
pi/mu/sigma, denoted with a `_cur` suffix below):
    H0[k,j] = cdf_k(b_j) - cdf_k(a_j)
    H1[k,j] = pdf_k(b_j) - pdf_k(a_j)
    H2[k,j] = b_j*pdf_k(b_j) - a_j*pdf_k(a_j)          (0 at infinite edges)
    G0 = H0
    G1 = mu_k_cur*H0 - sigma2_k_cur*H1
    A0[k,j] = pi_k_cur*G0[k,j] / (F_cur(b_j) - F_cur(a_j))
    A1[k,j] = pi_k_cur*G1[k,j] / (F_cur(b_j) - F_cur(a_j))
M-step (pi, mu) from A0, A1, THEN (since G2 needs the updated mu_k_new):
    G2 = sigma2_k_cur*(H0 + (2*mu_k_new - mu_k_cur)*H1 - H2) + (mu_k_new - mu_k_cur)^2*H0
    A2[k,j] = pi_k_cur*G2[k,j] / (F_cur(b_j) - F_cur(a_j))
M-step (sigma2) from A0, A2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def _edge_term(x: FloatArray, pdf_x: FloatArray) -> FloatArray:
    """x * pdf(x), forced to 0 at infinite edges (where pdf(x) already -> 0)."""
    safe_x: FloatArray = np.where(np.isinf(x), 0.0, x)
    return safe_x * pdf_x


def fit_gmm_binned(
    bin_edges: FloatArray,
    counts: FloatArray,
    n_components: int,
    n_iter: int = 100,
    rng: np.random.Generator | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """EM fit of a GMM from a histogram. Returns (pi, mu, sigma2), each shape (n_components,)."""
    if rng is None:
        rng = np.random.default_rng()
    a_edges, b_edges = bin_edges[:-1], bin_edges[1:]
    n_total = counts.sum()

    finite_centers = np.where(np.isfinite(a_edges) & np.isfinite(b_edges), (a_edges + b_edges) / 2, np.nan)
    weighted_mean = np.nansum(finite_centers * counts) / np.nansum(np.where(np.isnan(finite_centers), 0, counts))
    weighted_std = np.sqrt(
        np.nansum(counts * (finite_centers - weighted_mean) ** 2)
        / np.nansum(np.where(np.isnan(finite_centers), 0, counts))
    )
    pi = np.full(n_components, 1.0 / n_components)
    mu = weighted_mean + weighted_std * rng.normal(size=n_components)
    sigma2 = np.full(n_components, weighted_std**2)

    for _ in range(n_iter):
        sigma = np.sqrt(sigma2)
        pdf_a = norm.pdf(a_edges[np.newaxis, :], mu[:, np.newaxis], sigma[:, np.newaxis])
        pdf_b = norm.pdf(b_edges[np.newaxis, :], mu[:, np.newaxis], sigma[:, np.newaxis])
        cdf_a = norm.cdf(a_edges[np.newaxis, :], mu[:, np.newaxis], sigma[:, np.newaxis])
        cdf_b = norm.cdf(b_edges[np.newaxis, :], mu[:, np.newaxis], sigma[:, np.newaxis])

        h0 = cdf_b - cdf_a
        h1 = pdf_b - pdf_a
        h2 = _edge_term(b_edges, pdf_b) - _edge_term(a_edges, pdf_a)

        mixture_cdf_a = pi @ cdf_a
        mixture_cdf_b = pi @ cdf_b
        denom = mixture_cdf_b - mixture_cdf_a

        g0 = h0
        g1 = mu[:, np.newaxis] * h0 - sigma2[:, np.newaxis] * h1
        a0 = pi[:, np.newaxis] * g0 / denom[np.newaxis, :]
        a1 = pi[:, np.newaxis] * g1 / denom[np.newaxis, :]

        pi_new = (counts * a0).sum(axis=1) / n_total
        mu_new = (counts * a1).sum(axis=1) / (counts * a0).sum(axis=1)

        g2 = sigma2[:, np.newaxis] * (
            h0 + (2 * mu_new[:, np.newaxis] - mu[:, np.newaxis]) * h1 - h2
        ) + (mu_new[:, np.newaxis] - mu[:, np.newaxis]) ** 2 * h0
        a2 = pi[:, np.newaxis] * g2 / denom[np.newaxis, :]
        sigma2_new = (counts * a2).sum(axis=1) / (counts * a0).sum(axis=1)

        pi, mu, sigma2 = pi_new, mu_new, sigma2_new

    return pi, mu, sigma2
