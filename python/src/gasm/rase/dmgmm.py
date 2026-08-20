"""DM-GMM separator (Proposition 1). Thesis: chapter3RTASS.tex, l.296-478
(two-source case) + annexe calculus.tex, "Séparation : calcul des
distributions hiérarchiques" (l.144-184, 3+ source cascade).

Each source s is modeled as a GMM on the stacked real/imaginary complex
spectrum vector tspectrumc_s = [Re(spectrumc_s); Im(spectrumc_s)] (eq.
distSource, l.322): p_s(x) = sum_k pi_{s,k} N(x; mu_{s,k}, Sigma_{s,k}).

Missing-data regression (eq. distcond, l.410-434): for a two-source mixture
x = x1 + x2 with x1 independent of x2, the conditional p(x1|x) is a GMM
with n1*n2 components, one per (component_1, component_2) pair:

    Sigma_sum      = Sigma_1k1 + Sigma_2k2
    mu_tilde_k1k2  = mu_1k1 + Sigma_1k1 @ inv(Sigma_sum) @ (x - mu_1k1 - mu_2k2)
    Sigma_tilde_k1k2 = Sigma_1k1 - Sigma_1k1 @ inv(Sigma_sum) @ Sigma_1k1
    phi_k1k2(x)    prop_to pi_1k1 * pi_2k2 * N(x; mu_1k1 + mu_2k2, Sigma_sum)

The reconstruction used is the conditional mean (Gaussian mixture
regression, l.435-448): x1_hat = sum_k1,k2 phi_k1k2(x) * mu_tilde_k1k2.
x2_hat is recovered as x - x1_hat rather than re-derived symmetrically,
since the two must sum back to the observed mixture by construction.

3+ source hierarchical extension (annexe, l.144-184): to split
x = x_1 + ... + x_n, the last n-1 sources are first combined into one
synthetic GMM (Cartesian product of components: weights multiply, means
and covariances add -- the same "sum of independent Gaussians" rule used
for p(x) above), the pairwise regression above splits x into
x_{1..n-1}_hat and x_n_hat, then the same procedure recurses on
x_{1..n-1}_hat to split it into the remaining n-1 sources.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_LOG_2PI = float(np.log(2.0 * np.pi))


def _stack_real_imag(spectra: ComplexArray) -> FloatArray:
    return np.concatenate([spectra.real, spectra.imag], axis=-1)


def _unstack_real_imag(stacked: FloatArray) -> ComplexArray:
    freqbin = stacked.shape[-1] // 2
    return stacked[..., :freqbin] + 1j * stacked[..., freqbin:]


@dataclass
class SourceGMM:
    """GMM over a source's stacked real/imaginary spectrum (eq. distSource)."""

    weights: FloatArray  # (n_components,)
    means: FloatArray  # (n_components, 2*freqbin)
    covariances: FloatArray  # (n_components, 2*freqbin, 2*freqbin)

    @staticmethod
    def fit(
        spectra: ComplexArray,
        n_components: int,
        random_state: int | None = None,
        verbose: bool = False,
        max_iter: int = 100,
    ) -> SourceGMM:
        """spectra: (n_frames, freqbin) complex. Fits a full-covariance GMM
        on [Re(spectrum); Im(spectrum)]. verbose: sklearn prints per-EM-iteration
        log-likelihood/timing (this fit is the slow step at high freqbin --
        each iteration re-estimates n_components full (2*freqbin)^2 covariances).

        max_iter caps EM. sklearn's stopping rule is an absolute tolerance on the
        mean lower bound, so at high freqbin the fit runs to the cap: one iteration
        costs O(n_frames * n_components * (2*freqbin)^2), which is ~40 s at 15k
        frames and freqbin=513. Capping it also makes fits over different training
        set sizes comparable, since a small set converges in a few iterations by
        overfitting while a large one keeps improving."""
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=random_state,
            verbose=2 if verbose else 0,
            verbose_interval=1,
            max_iter=max_iter,
        )
        gmm.fit(_stack_real_imag(spectra))
        return SourceGMM(gmm.weights_, gmm.means_, gmm.covariances_)

    def log_pdf(self, x: FloatArray) -> float:
        """log p_s(x) under this source's GMM."""
        log_terms = [
            np.log(w) + multivariate_normal.logpdf(x, mean=mu, cov=sigma)
            for w, mu, sigma in zip(self.weights, self.means, self.covariances)
        ]
        return float(logsumexp(log_terms))


def _combine(a: SourceGMM, b: SourceGMM) -> SourceGMM:
    """Sum of two independent GMM sources is a GMM over the Cartesian
    product of components (l.144-152) -- also used to build the synthetic
    combined source for the hierarchical extension."""
    weights = (a.weights[:, None] * b.weights[None, :]).ravel()
    means = (a.means[:, None, :] + b.means[None, :, :]).reshape(-1, a.means.shape[-1])
    covariances = (a.covariances[:, None] + b.covariances[None, :]).reshape(-1, *a.covariances.shape[-2:])
    return SourceGMM(weights, means, covariances)


def _diagonal_variances(covariances: FloatArray) -> FloatArray | None:
    """The variance vectors when every covariance is diagonal, else None.

    A diagonal covariance is not an exotic case here, it is the only one that
    fits at high freqbin: a full covariance per component needs more frames than
    a track has once the stacked dimension passes a thousand, so a fit run at
    that dimension is diagonal and hands its variances back widened into
    matrices. Recognizing them turns the per-pair solve and the regression gain
    from O(n_frames * dim^2) into O(n_frames * dim), three orders of magnitude
    at freqbin = 513, and returns the same numbers: dividing by the variances is
    what the Cholesky of a diagonal matrix does, with fewer roundings on the way.

    A matrix is diagonal exactly when its nonzeros are its diagonal's, which
    counts in place rather than building a (n_components, dim, dim) comparison.
    """
    for covariance in covariances:
        if np.count_nonzero(covariance) != np.count_nonzero(np.diagonal(covariance)):
            return None
    return np.diagonal(covariances, axis1=-2, axis2=-1)


def _pair_terms(
    source1: SourceGMM, source2: SourceGMM, x: FloatArray
) -> Iterator[tuple[int, FloatArray, FloatArray]]:
    """For each (k1, k2) component pair, yield (k1, log_phi, solved) evaluated
    on every frame of x at once:

        log_phi[t] = log(pi_1k1 * pi_2k2) + log N(x_t; mu_1k1 + mu_2k2, Sigma_sum)
        solved[t]  = inv(Sigma_sum) @ (x_t - mu_1k1 - mu_2k2)

    Sigma_sum and its Cholesky factor depend only on the pair, never on the
    frame, so factorizing once per pair and applying it to the whole block is
    what makes this tractable: O(n1*n2*d^3) instead of O(n_frames*n1*n2*d^3).
    The single triangular solve serves both the Gaussian's quadratic form and
    the regression gain Sigma_1k1 @ inv(Sigma_sum), which are the same inverse.

    When both sources are diagonal so is every Sigma_sum, and the factorization
    and the solve collapse to a reciprocal (see _diagonal_variances). The test
    runs once per call rather than once per pair, so it costs O(n1*d^2) against
    the O(n1*n2*n_frames*d^2) it removes.
    """
    dim = x.shape[-1]
    variances1 = _diagonal_variances(source1.covariances)
    variances2 = _diagonal_variances(source2.covariances)
    diagonal = variances1 is not None and variances2 is not None
    for k1, (weight1, mean1, cov1) in enumerate(
        zip(source1.weights, source1.means, source1.covariances)
    ):
        for k2, (weight2, mean2, cov2) in enumerate(
            zip(source2.weights, source2.means, source2.covariances)
        ):
            delta = x - (mean1 + mean2)
            if diagonal:
                total = variances1[k1] + variances2[k2]
                solved = delta / total
                log_det = np.log(total).sum()
            else:
                chol = cho_factor(cov1 + cov2, lower=True)
                solved = cho_solve(chol, delta.T).T
                log_det = 2.0 * np.log(np.diag(chol[0])).sum()
            log_phi = np.log(weight1 * weight2) - 0.5 * (
                np.einsum("td,td->t", delta, solved) + log_det + dim * _LOG_2PI
            )
            yield k1, log_phi, solved


def _regress(
    source1: SourceGMM, source2: SourceGMM, x: FloatArray, verbose: bool = False
) -> FloatArray:
    """Conditional mean E[X1 | X1+X2=x] (eq. distcond's Gaussian mixture
    regression, l.410-448), for a whole block of frames x: (n_frames, 2*freqbin).

    The (k1, k2) loop runs twice, once to accumulate the mixture weights and
    once to accumulate the weighted conditional means, because caching
    mu_tilde for every pair would need n1*n2*n_frames*dim floats (480 GB at
    100 pairs, 585 frames, freqbin=513). Redoing the factorizations is the
    cheaper half of that trade.
    """
    n_pairs = len(source1.weights) * len(source2.weights)
    if verbose:
        print(f"DM-GMM regress: {n_pairs} component pairs x {len(x)} frames", flush=True)
    log_phi = np.empty((n_pairs, len(x)))
    for i, (_, phi_row, _) in enumerate(_pair_terms(source1, source2, x)):
        log_phi[i] = phi_row
    log_norm = logsumexp(log_phi, axis=0)
    estimate = np.zeros_like(x)
    # the gain Sigma_1k1 @ inv(Sigma_sum) is the second O(n_frames * dim^2) term
    # of the pair, and it collapses on its own terms: source2 may be dense here
    # and the multiply still holds as long as source1 is diagonal
    variances1 = _diagonal_variances(source1.covariances)
    for i, (k1, phi_row, solved) in enumerate(_pair_terms(source1, source2, x)):
        gain = solved * variances1[k1] if variances1 is not None else solved @ source1.covariances[k1].T
        mu_tilde = source1.means[k1] + gain
        estimate += np.exp(phi_row - log_norm)[:, None] * mu_tilde
        if verbose and (i + 1) % 10 == 0:
            print(f"DM-GMM regress: pair {i + 1}/{n_pairs}", flush=True)
    return estimate


def _separate(sources: list[SourceGMM], x: FloatArray, verbose: bool = False) -> list[FloatArray]:
    """Hierarchical cascade (annexe, l.144-184): peel the last source off by
    combining the rest, regress the pair, then recurse on the remainder. The
    cascade structure depends only on the sources, so every frame of x follows
    the same path and the whole block goes through it together."""
    if len(sources) == 1:
        return [x]
    *rest, last = sources
    combined_rest = rest[0]
    for source in rest[1:]:
        combined_rest = _combine(combined_rest, source)
    rest_hat = _regress(combined_rest, last, x, verbose)
    last_hat = x - rest_hat
    return _separate(rest, rest_hat, verbose) + [last_hat]


class DMGMMSeparator:
    """DM-GMM separator (Proposition 1, chapter3RTASS.tex l.296-478), applied
    hierarchically for 3+ sources (annexe calculus.tex, l.144-184)."""

    def __init__(self, n_components: list[int]) -> None:
        self.n_components = n_components

    def fit(
        self,
        source_spectra: list[ComplexArray],
        random_state: int | None = None,
        verbose: bool = False,
        max_iter: int = 100,
    ) -> DMGMMSeparator:
        """source_spectra[s]: (n_frames_s, freqbin) complex training frames for
        source s. max_iter caps each source's EM (see SourceGMM.fit)."""
        self.sources_ = [
            SourceGMM.fit(spectra, n, random_state, verbose, max_iter)
            for spectra, n in zip(source_spectra, self.n_components)
        ]
        return self

    def predict(self, mixed_spectra: ComplexArray, verbose: bool = False) -> list[ComplexArray]:
        """mixed_spectra: (n_frames, freqbin) complex. Returns one (n_frames,
        freqbin) complex array per source, the DM-GMM point estimate per frame.
        verbose: print a component-pair counter to stdout (the O(n_components^2)
        regression is the slow step at high freqbin/component count -- useful to
        see it's progressing, not hung)."""
        estimates = _separate(self.sources_, _stack_real_imag(mixed_spectra), verbose)
        return [_unstack_real_imag(estimate) for estimate in estimates]

    def score_samples(self, mixed_spectra: ComplexArray) -> FloatArray:
        """Per-frame log p(mixture) under the fitted joint model p(x) =
        sum over all sources' component combinations (l.144-152's combine
        rule applied across every source)."""
        joint = self.sources_[0]
        for source in self.sources_[1:]:
            joint = _combine(joint, source)
        stacked = _stack_real_imag(mixed_spectra)
        return np.array([joint.log_pdf(x) for x in stacked])
