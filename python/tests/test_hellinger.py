import numpy as np

from gasm.common.hellinger import (
    hellinger_distance,
    hellinger_squared,
    pairwise_hellinger_distance,
)


def test_hellinger_zero_for_identical_distributions() -> None:
    p = np.array([0.2, 0.3, 0.5])
    assert hellinger_squared(p, p) == 0.0
    assert hellinger_distance(p, p) == 0.0


def test_hellinger_max_for_disjoint_support() -> None:
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert np.isclose(hellinger_squared(p, q), 1.0)
    assert np.isclose(hellinger_distance(p, q), 1.0)


def test_pairwise_matches_pointwise() -> None:
    params = np.array([[0.2, 0.8], [0.5, 0.5], [1.0, 0.0]])
    pairwise = pairwise_hellinger_distance(params)
    for i in range(3):
        for j in range(3):
            assert np.isclose(pairwise[i, j], hellinger_distance(params[i], params[j]))
