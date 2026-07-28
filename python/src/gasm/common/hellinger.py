"""Discrete Hellinger distance. Thesis: chapter2RTASC.tex, l.578-582 (definition,
continuous form H^2(P,Q) = 1/2 * integral (sqrt(dP) - sqrt(dQ))^2), specialized
here to discrete probability vectors (the multinomial kernel parameters
multparam_(i)) for the hierarchical reduction of RARE's kernels (Ward linkage).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def hellinger_squared(p: FloatArray, q: FloatArray) -> float:
    """H^2(P,Q) = 1/2 * sum((sqrt(p_i) - sqrt(q_i))^2), for discrete P, Q."""
    return float(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))


def hellinger_distance(p: FloatArray, q: FloatArray) -> float:
    """H(P,Q) = sqrt(H^2(P,Q)), a proper metric in [0, 1]."""
    return float(np.sqrt(hellinger_squared(p, q)))


def pairwise_hellinger_distance(params: FloatArray) -> FloatArray:
    """Pairwise H(P_i, P_j) for ``params`` of shape (n_kernels, freqbin).

    Vectorized via ||sqrt(p_i) - sqrt(p_j)||, suitable as input to
    scipy.cluster.hierarchy (e.g. squareform + Ward linkage).
    """
    sqrt_params = np.sqrt(params)
    diff = sqrt_params[:, np.newaxis, :] - sqrt_params[np.newaxis, :, :]
    return np.sqrt(0.5 * np.sum(diff**2, axis=-1))
