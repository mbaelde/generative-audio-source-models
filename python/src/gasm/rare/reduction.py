"""Prototype hierarchy + per-class thresholding to keep RareClassifier
real-time. Thesis: chapter2RTASC.tex, section "Réduire la complexité"
(l.546-664): Hellinger distance + Ward linkage hierarchical clustering,
then a thresholding procedure adjusting the per-class reduction factor
until a precision target is met. Polyphonic classification must be built
from these already-reduced monophonic kernels, not the raw ones.

reduce_class_kernels implements eq. l.599-606: cut the Ward tree (built on
pairwise Hellinger distance between a class's multparam_(i)) to n_clusters,
then average multparam_(i) per cluster -- p_bar_(i') = mean over the cluster.

reduce_rare_kernels applies the same reduction_factor r to every class
(n_label' = round(n_label / r)) and returns kernels re-expressed as
"spectra" (they already sum to ~1, like a normalized power spectrum) so
they can be handed straight to RareClassifier(quantization).fit -- fit's
own quantize_spectrum re-snaps the averaged params onto the quantization
grid, which is required for reduced and unreduced kernels to score on a
common footing. The same trick lets make_polyphonic_dataset mix pairs of
*reduced* kernels (l.664's branching remark) unchanged: pass its return
value as reduce_rare_kernels's spectra/labels.

threshold_reduce_rare_kernels implements the thresholding heuristic
(l.658-664, prose only -- the thesis gives no closed-form formula, only
"start from r^(0), reduce, measure precision, increase kernel count if
below threshold, repeat"): per class, r starts at initial_reduction_factor
and is halved (less reduction) until the caller-supplied
evaluate_class_accuracy callback clears precision_threshold or r reaches 1
(no reduction). The loop's shape is this implementation's choice; the
callback -- typically "refit a RareClassifier and measure this class's
recall on a validation set" -- is left to the caller since the thesis's
metric depends on a validation set reduction.py has no reason to own.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from gasm.common.hellinger import pairwise_hellinger_distance
from gasm.common.multinomial_kernel import multinomial_params, quantize_spectrum

FloatArray = NDArray[np.float64]


def reduce_class_kernels(params: FloatArray, n_clusters: int) -> FloatArray:
    """Ward-linkage (Hellinger distance) reduction of one class's kernels to
    ``n_clusters`` averaged prototypes (eq. l.599-606)."""
    n = len(params)
    if n_clusters >= n:
        return params
    condensed = squareform(pairwise_hellinger_distance(params), checks=False)
    cluster_ids = fcluster(linkage(condensed, method="ward"), t=n_clusters, criterion="maxclust")
    return np.stack([params[cluster_ids == c].mean(axis=0) for c in np.unique(cluster_ids)])


def reduce_rare_kernels(
    spectra: FloatArray, labels: FloatArray, quantization: int, reduction_factor: float
) -> tuple[FloatArray, FloatArray]:
    """Reduce every class's kernels by the same factor r (l.593-599):
    n_label' = max(1, round(n_label / r)). Returns (reduced_spectra,
    reduced_labels), ready for RareClassifier(quantization).fit."""
    classes, label_index = np.unique(labels, return_inverse=True)
    reduced_spectra = []
    reduced_labels = []
    for c, class_value in enumerate(classes):
        params = multinomial_params(quantize_spectrum(spectra[label_index == c], quantization), quantization)
        n_clusters = max(1, round(len(params) / reduction_factor))
        reduced = reduce_class_kernels(params, n_clusters)
        reduced_spectra.append(reduced)
        reduced_labels.append(np.full(len(reduced), class_value))
    return np.concatenate(reduced_spectra), np.concatenate(reduced_labels)


def threshold_reduce_rare_kernels(
    spectra: FloatArray,
    labels: FloatArray,
    quantization: int,
    evaluate_class_accuracy: Callable[[FloatArray, float], float],
    precision_threshold: float,
    initial_reduction_factor: float = 2.0,
    growth_rate: float = 2.0,
    max_iter: int = 10,
) -> tuple[FloatArray, FloatArray]:
    """Per-class adaptive reduction (l.658-664). For each class independently,
    r starts at ``initial_reduction_factor`` and is divided by ``growth_rate``
    (less aggressive reduction, more kernels kept) each time
    ``evaluate_class_accuracy(reduced_class_params, class_value)`` falls
    below ``precision_threshold``, until it clears the threshold, r reaches
    1 (no reduction left to give), or ``max_iter`` is hit."""
    classes, label_index = np.unique(labels, return_inverse=True)
    reduced_spectra = []
    reduced_labels = []
    for c, class_value in enumerate(classes):
        params = multinomial_params(quantize_spectrum(spectra[label_index == c], quantization), quantization)
        r = initial_reduction_factor
        reduced = reduce_class_kernels(params, max(1, round(len(params) / r)))
        for _ in range(max_iter):
            if evaluate_class_accuracy(reduced, class_value) >= precision_threshold or r <= 1.0:
                break
            r = max(1.0, r / growth_rate)
            reduced = reduce_class_kernels(params, max(1, round(len(params) / r)))
        reduced_spectra.append(reduced)
        reduced_labels.append(np.full(len(reduced), class_value))
    return np.concatenate(reduced_spectra), np.concatenate(reduced_labels)
