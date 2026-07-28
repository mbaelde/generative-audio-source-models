import numpy as np

from gasm.rare.classifier import RareClassifier, make_polyphonic_dataset, pad_with_white_noise


def _make_spectra(peak_bin: int, n_bins: int, n: int, rng: np.random.Generator) -> np.ndarray:
    base = np.full(n_bins, 0.02)
    base[peak_bin] = 1.0 - 0.02 * (n_bins - 1)
    noise = rng.normal(scale=0.01, size=(n, n_bins))
    spectra = np.clip(base + noise, 1e-6, None)
    return spectra / spectra.sum(axis=1, keepdims=True)


def test_predict_recovers_well_separated_classes() -> None:
    rng = np.random.default_rng(0)
    n_bins = 16
    low = _make_spectra(peak_bin=1, n_bins=n_bins, n=50, rng=rng)
    high = _make_spectra(peak_bin=14, n_bins=n_bins, n=50, rng=rng)
    spectra = np.vstack([low, high])
    labels = np.array([0] * 50 + [1] * 50)

    clf = RareClassifier(quantization=64).fit(spectra, labels)

    test_low = _make_spectra(peak_bin=1, n_bins=n_bins, n=10, rng=rng)
    test_high = _make_spectra(peak_bin=14, n_bins=n_bins, n=10, rng=rng)
    assert np.all(clf.predict(test_low) == 0)
    assert np.all(clf.predict(test_high) == 1)


def test_cross_entropy_kernel_is_the_large_quantization_limit() -> None:
    """round(q*p)/q -> p, so the multinomial kernel converges to the cross-entropy
    one as quantization grows. This is the convergence the manuscript invokes for
    table:q's plateau, and it is why the two kernels are the same estimator at the
    published quantization values.

    Dirichlet spectra, so every bin sits well above 1/quantization: a bin below
    that rounds to zero and stays pinned at log(eps), which is exactly where the
    two kernels keep disagreeing however large quantization gets."""
    rng = np.random.default_rng(6)
    n_bins = 16
    spectra = rng.dirichlet(np.full(n_bins, 5.0), size=60)
    labels = np.repeat([0, 1, 2], 20)
    queries = rng.dirichlet(np.full(n_bins, 5.0), size=8)

    quantization = 100_000
    multinomial = RareClassifier(quantization, kernel="multinomial").fit(spectra, labels)
    cross_entropy = RareClassifier(quantization, kernel="cross_entropy").fit(spectra, labels)

    np.testing.assert_allclose(
        multinomial.decision_function(queries), cross_entropy.decision_function(queries), rtol=1e-4
    )
    np.testing.assert_array_equal(multinomial.predict(queries), cross_entropy.predict(queries))


def test_cross_entropy_kernel_recovers_well_separated_classes() -> None:
    rng = np.random.default_rng(7)
    n_bins = 16
    low = _make_spectra(peak_bin=1, n_bins=n_bins, n=50, rng=rng)
    high = _make_spectra(peak_bin=14, n_bins=n_bins, n=50, rng=rng)
    clf = RareClassifier(quantization=n_bins, kernel="cross_entropy").fit(
        np.vstack([low, high]), np.array([0] * 50 + [1] * 50)
    )

    assert np.all(clf.predict(_make_spectra(peak_bin=1, n_bins=n_bins, n=10, rng=rng)) == 0)
    assert np.all(clf.predict(_make_spectra(peak_bin=14, n_bins=n_bins, n=10, rng=rng)) == 1)


def test_zero_valued_training_bins_score_finitely() -> None:
    """A training spectrum with an empty bin used to log to -1e300; the thesis'
    own eps floor keeps every score finite, whatever the query puts in that bin."""
    spectra = np.array([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])
    clf = RareClassifier(quantization=8).fit(spectra, np.array([0, 1]))

    scores = clf.decision_function(np.array([[0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0]]))
    assert np.all(np.isfinite(scores))
    assert np.all(np.isfinite(clf.predict_proba(spectra)))


def test_predict_proba_sums_to_one() -> None:
    rng = np.random.default_rng(1)
    spectra = _make_spectra(peak_bin=3, n_bins=8, n=20, rng=rng)
    labels = rng.integers(0, 3, size=20)
    clf = RareClassifier(quantization=32).fit(spectra, labels)

    proba = clf.predict_proba(spectra[:5])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert np.all(np.isfinite(proba))


def test_make_polyphonic_dataset_never_pairs_a_label_with_itself() -> None:
    rng = np.random.default_rng(4)
    spectra = _make_spectra(peak_bin=3, n_bins=8, n=30, rng=rng)
    labels = rng.integers(0, 3, size=30)

    _, joint_labels = make_polyphonic_dataset(spectra, labels, n_samples=200, rng=rng, proportion=0.6)

    assert all(pair[0] != pair[1] for pair in joint_labels)


def test_polyphonic_mode_recovers_joint_labels() -> None:
    rng = np.random.default_rng(5)
    n_bins = 16
    spectra = np.vstack([_make_spectra(peak_bin=b, n_bins=n_bins, n=200, rng=rng) for b in (1, 7, 13)])
    labels = np.array([0] * 200 + [1] * 200 + [2] * 200)

    mixed, joint_labels = make_polyphonic_dataset(spectra, labels, n_samples=1500, rng=rng, proportion=0.7)
    clf = RareClassifier(quantization=64).fit(mixed, joint_labels)

    test0 = _make_spectra(peak_bin=1, n_bins=n_bins, n=5, rng=rng)
    test1 = _make_spectra(peak_bin=7, n_bins=n_bins, n=5, rng=rng)
    query = 0.7 * test0 + 0.3 * test1
    predicted = clf.predict(query)

    assert all(pair == (0, 1) for pair in predicted)


def test_pad_with_white_noise_preserves_signal_and_fills_rest() -> None:
    rng = np.random.default_rng(2)
    signal = np.array([1.0, 2.0, 3.0])
    frame = pad_with_white_noise(signal, frame_length=10, start_offset=4, rng=rng, noise_std=1.0)

    assert frame.shape == (10,)
    np.testing.assert_array_equal(frame[4:7], signal)
    assert np.var(frame[:4]) > 0
    assert np.var(frame[7:]) > 0


def test_pad_with_white_noise_clips_out_of_bounds() -> None:
    rng = np.random.default_rng(3)
    signal = np.array([1.0, 2.0, 3.0, 4.0])
    frame = pad_with_white_noise(signal, frame_length=5, start_offset=-1, rng=rng, noise_std=1.0)

    assert frame.shape == (5,)
    np.testing.assert_array_equal(frame[:3], signal[1:4])
