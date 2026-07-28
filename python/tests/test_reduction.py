import numpy as np

from gasm.rare.classifier import RareClassifier
from gasm.rare.reduction import reduce_class_kernels, reduce_rare_kernels, threshold_reduce_rare_kernels


def _make_spectra(peak_bin: int, n_bins: int, n: int, rng: np.random.Generator) -> np.ndarray:
    base = np.full(n_bins, 0.02)
    base[peak_bin] = 1.0 - 0.02 * (n_bins - 1)
    noise = rng.normal(scale=0.01, size=(n, n_bins))
    spectra = np.clip(base + noise, 1e-6, None)
    return spectra / spectra.sum(axis=1, keepdims=True)


def test_reduce_class_kernels_groups_two_tight_clusters() -> None:
    rng = np.random.default_rng(0)
    near_a = _make_spectra(peak_bin=1, n_bins=8, n=10, rng=rng)
    near_b = _make_spectra(peak_bin=6, n_bins=8, n=10, rng=rng)
    params = np.vstack([near_a, near_b])

    reduced = reduce_class_kernels(params, n_clusters=2)

    assert reduced.shape == (2, 8)
    peak_bins = sorted(np.argmax(reduced, axis=1).tolist())
    assert peak_bins == [1, 6]


def test_reduce_class_kernels_no_op_when_n_clusters_covers_all() -> None:
    params = np.array([[0.5, 0.5], [0.3, 0.7]])
    reduced = reduce_class_kernels(params, n_clusters=5)
    np.testing.assert_array_equal(reduced, params)


def test_reduce_rare_kernels_preserves_classification() -> None:
    rng = np.random.default_rng(1)
    n_bins = 16
    low = _make_spectra(peak_bin=1, n_bins=n_bins, n=40, rng=rng)
    high = _make_spectra(peak_bin=14, n_bins=n_bins, n=40, rng=rng)
    spectra = np.vstack([low, high])
    labels = np.array([0] * 40 + [1] * 40)

    reduced_spectra, reduced_labels = reduce_rare_kernels(spectra, labels, quantization=64, reduction_factor=4.0)
    assert reduced_spectra.shape[0] == 20
    assert reduced_labels.shape[0] == 20

    clf = RareClassifier(quantization=64).fit(reduced_spectra, reduced_labels)
    test_low = _make_spectra(peak_bin=1, n_bins=n_bins, n=10, rng=rng)
    test_high = _make_spectra(peak_bin=14, n_bins=n_bins, n=10, rng=rng)
    assert np.all(clf.predict(test_low) == 0)
    assert np.all(clf.predict(test_high) == 1)


def test_threshold_reduce_stops_immediately_when_already_above_threshold() -> None:
    rng = np.random.default_rng(2)
    spectra = _make_spectra(peak_bin=2, n_bins=8, n=20, rng=rng)
    labels = np.zeros(20)

    reduced_spectra, _ = threshold_reduce_rare_kernels(
        spectra,
        labels,
        quantization=32,
        evaluate_class_accuracy=lambda reduced, class_value: 1.0,
        precision_threshold=0.9,
        initial_reduction_factor=4.0,
    )
    assert reduced_spectra.shape[0] == 5


def test_threshold_reduce_falls_back_to_no_reduction_when_never_satisfied() -> None:
    rng = np.random.default_rng(3)
    spectra = _make_spectra(peak_bin=2, n_bins=8, n=20, rng=rng)
    labels = np.zeros(20)

    reduced_spectra, _ = threshold_reduce_rare_kernels(
        spectra,
        labels,
        quantization=32,
        evaluate_class_accuracy=lambda reduced, class_value: 0.0,
        precision_threshold=0.9,
        initial_reduction_factor=4.0,
        max_iter=5,
    )
    assert reduced_spectra.shape[0] == 20
