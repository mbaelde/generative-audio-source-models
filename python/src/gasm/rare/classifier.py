"""RareClassifier: real-time mono audio classification. Thesis: chapter2RTASC.tex.

fit stores, per training frame, the quantized spectrum's multinomial params
p_(i) and its label, plus the class priors p_hat_label = n_label/n_samples
(l.302-306). Scoring uses eq. l.564 (following eq:distapproxmono): the
multinomial PMF's combinatorial factor C(x) = n!/prod(x_b!) depends only on
the quantized query (not on i or on label), so

    p_hat(spectrum|label) = (1/n_label) * C(x) * sum_{i: label_(i)=label} exp(x^T log(p_(i)))

Multiplying by the prior n_label/n_samples cancels the 1/n_label exactly,
leaving posterior(label) propto sum_{i: label_(i)=label} exp(x^T log(p_(i))),
with the remaining C(x)/n_samples factor common to every label (drops out of
predict_proba's softmax). So the decision rule needs no explicit prior or
kernel-count bookkeeping -- priors_ is still exposed for inspection since the
thesis names it explicitly, but scoring does not use it.

log(0) from a training p_(i,b) == 0 is floored at log(eps), which is what the
thesis' own code does (``feature(feature == 0) = eps``): finite rather than
-inf, so a query with x_b == 0 there contributes exactly 0 (the x*log(p)
convention 0*log(0):=0) instead of 0 * -inf = nan.

``kernel`` selects between two members of the same family:

- ``"multinomial"`` (default) is eq. def_xi as written: both the training
  spectra and the query are quantized to integer counts before scoring.
- ``"cross_entropy"`` leaves both continuous, scoring
  L_i = sum_b quantization*spectrum_b * log(p_(i,b)) = -quantization *
  H(spectrum, p_(i)), so quantization acts as an inverse temperature.

The second is what the thesis' MATLAB actually ran: ``compute_feature.m``
l.16-18 (theta) and ``identification.m`` l.51 (query) both have their
rounding commented out. It is also the quantization -> infinity limit of the
first, since round(q*p)/q -> p, which is the convergence the manuscript
invokes to explain the plateau in table:q. Reproducing the published scores
needs ``kernel="cross_entropy"``; see python/README.md.

Polyphonic mode (eq. distapproxpoly, l.505-544): a polyphonic training sample
mixes two monophonic spectra, spectrum_(i) = rho_(i)*spectrum_1(i) +
(1-rho_(i))*spectrum_2(i), and carries a joint label (label_1(i), label_2(i))
drawn i.i.d. (without replacement, so label_1 != label_2) from the mono class
prior. Thesis: p_hat(spectrum|label_1,label_2) uses the same multinomial
kernel with denominator n_label_1*n_label_2, and the class-pair prior
factorizes as p(label_1)*p(label_2). With i.i.d. label-pair sampling,
n_label_1*n_label_2/n_samples_poly plays exactly the role n_label/n_samples
played in the mono case, so the same cancellation applies and no separate
classifier is needed: build the mixed dataset with make_polyphonic_dataset
below, then hand it to the same RareClassifier.fit/predict, using the joint
label as an ordinary (if compound) class label.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp, softmax

from gasm.common.multinomial_kernel import multinomial_params, quantize_spectrum

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

Kernel = Literal["multinomial", "cross_entropy"]

_EPS = float(np.finfo(np.float64).eps)


def pad_with_white_noise(
    signal: FloatArray,
    frame_length: int,
    start_offset: int,
    rng: np.random.Generator,
    noise_std: float = 1.0,
) -> FloatArray:
    """Place ``signal`` at ``start_offset`` in a frame of ``frame_length``, filling
    the rest with white Gaussian noise (thesis: remarque near Fig. gwn, l.310 --
    not silence, so partial edge frames don't look artificially quiet)."""
    frame = rng.normal(scale=noise_std, size=frame_length)
    lo, hi = max(start_offset, 0), min(start_offset + len(signal), frame_length)
    frame[lo:hi] = signal[lo - start_offset : hi - start_offset]
    return frame


class RareClassifier:
    """Monophonic RARE classifier (eq. distapproxmono, l.239 + l.564)."""

    def __init__(self, quantization: int, kernel: Kernel = "multinomial") -> None:
        self.quantization = quantization
        self.kernel = kernel

    def fit(self, spectra: FloatArray, labels: FloatArray) -> RareClassifier:
        """spectra: (n, freqbin) normalized power spectra. labels: (n,)."""
        if self.kernel == "cross_entropy":
            params = spectra
        else:
            params = multinomial_params(
                quantize_spectrum(spectra, self.quantization), self.quantization
            )
        self._log_params = np.log(np.maximum(params, _EPS))
        self.classes_, self._label_index = np.unique(labels, return_inverse=True)
        self.priors_ = np.bincount(self._label_index) / len(labels)
        return self

    def decision_function(self, spectra: FloatArray) -> FloatArray:
        """Per-class log-score (l.564), up to an additive constant common to all
        classes. Shape (n_queries, n_classes)."""
        if self.kernel == "cross_entropy":
            query = self.quantization * spectra
        else:
            query = quantize_spectrum(spectra, self.quantization).astype(np.float64)
        dots = query @ self._log_params.T
        return np.stack(
            [logsumexp(dots[:, self._label_index == c], axis=1) for c in range(len(self.classes_))],
            axis=1,
        )

    def predict_proba(self, spectra: FloatArray) -> FloatArray:
        return softmax(self.decision_function(spectra), axis=1)

    def predict(self, spectra: FloatArray) -> FloatArray:
        return self.classes_[np.argmax(self.decision_function(spectra), axis=1)]


def make_polyphonic_dataset(
    spectra: FloatArray,
    labels: FloatArray,
    n_samples: int,
    rng: np.random.Generator,
    proportion: float | None = None,
) -> tuple[FloatArray, NDArray[np.object_]]:
    """Synthetic polyphonic training set from monophonic (spectrum, label) pairs
    (l.505-533). Returns (mixed_spectra, joint_labels); joint_labels[i] is the
    (label_1, label_2) tuple, directly usable as RareClassifier.fit's ``labels``.

    ``proportion``: fixed rho for every sample if given, else Uniform(0,1) per
    sample -- the thesis leaves rho's distribution to the implementation and
    only requires test-time proportions to match those seen in training.
    """
    class_values, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    label_1 = rng.choice(class_values, size=n_samples, p=probs)
    label_2 = rng.choice(class_values, size=n_samples, p=probs)
    resample = label_1 == label_2
    while np.any(resample):
        label_2[resample] = rng.choice(class_values, size=int(resample.sum()), p=probs)
        resample = label_1 == label_2

    label_to_indices = {value: np.flatnonzero(labels == value) for value in class_values}
    idx_1 = np.array([rng.choice(label_to_indices[label]) for label in label_1])
    idx_2 = np.array([rng.choice(label_to_indices[label]) for label in label_2])

    rho = np.full(n_samples, proportion) if proportion is not None else rng.uniform(size=n_samples)
    mixed = rho[:, np.newaxis] * spectra[idx_1] + (1.0 - rho[:, np.newaxis]) * spectra[idx_2]
    joint_labels: NDArray[np.object_] = np.empty(n_samples, dtype=object)
    joint_labels[:] = list(zip(label_1.tolist(), label_2.tolist()))
    return mixed, joint_labels
