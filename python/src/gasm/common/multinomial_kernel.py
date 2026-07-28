"""Multinomial kernel density estimation. Thesis: chapter2RTASC.tex, l.257-293.

quantize_spectrum (eq. def_xi): nearest integer vector to quantization*spectrum
    xi_(i) = argmin_{v in N^freqbin} ||quantization*spectrum_(i) - v||
           = round(quantization*spectrum_(i))   (thesis: "vecteur arrondi ou tronqué")

multinomial_params: p_(i) = xi_(i) / quantization

multinomial_kernel: K(x, xi_(i)) = Mult_freqbin(x; quantization, p_(i)), the
multinomial PMF evaluated at a quantized query point x. Computed directly
(not via scipy.stats.multinomial) because p_(i) need not sum to exactly 1
after independent per-bin rounding.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def quantize_spectrum(spectrum: FloatArray, quantization: int) -> IntArray:
    """eq. def_xi: nearest integer vector to quantization * spectrum."""
    return np.round(quantization * spectrum).astype(np.int64)


def multinomial_params(quantized_spectrum: IntArray, quantization: int) -> FloatArray:
    """p_(i) = xi_(i) / quantization."""
    return quantized_spectrum / quantization


def multinomial_kernel(
    quantized_query: IntArray, params: FloatArray, quantization: int
) -> float:
    """Mult_freqbin(quantized_query; quantization, params), evaluated as a PMF."""
    with np.errstate(divide="ignore"):
        log_params = np.log(params)
    support = quantized_query > 0
    if np.any(support & (params == 0)):
        return 0.0
    log_pmf = (
        gammaln(quantization + 1)
        - np.sum(gammaln(quantized_query + 1))
        + np.sum(quantized_query[support] * log_params[support])
    )
    return float(np.exp(log_pmf))
