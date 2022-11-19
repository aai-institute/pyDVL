import numpy as np
import pytest

from pydvl.utils.numeric import (
    powerset,
    random_matrix_with_condition_number,
    random_powerset,
    spearman,
)


def test_powerset():
    with pytest.raises(TypeError):
        set(powerset(1))

    assert set(powerset(np.array([]))) == {()}
    assert set(powerset(np.array([1, 2]))) == {(), (1,), (1, 2), (2,)}

    # Check correct number of sets of each size
    n = 10
    size_counts = np.zeros(n + 1)
    item_counts = np.zeros(n, dtype=float)
    for subset in powerset(np.arange(n)):
        size_counts[len(subset)] += 1
        for x in subset:
            item_counts[x] += 1
    assert np.allclose(item_counts / 2**n, 0.5)
    assert all([np.math.comb(n, j) for j in range(n + 1)] == size_counts)


@pytest.mark.parametrize(
    "n, max_subsets", [(1, 10), (10, 2**10), (5, 2**7), (0, 1)]
)
def test_random_powerset(n, max_subsets):
    s = np.arange(n)
    item_counts = np.zeros_like(s, dtype=np.float)
    size_counts = np.zeros(n + 1)
    for subset in random_powerset(s, max_subsets=max_subsets):
        size_counts[len(subset)] += 1
        for item in subset:
            item_counts[item] += 1
    item_counts /= max_subsets

    # Test frequency of items in sets. the rtol is a hack to allow for more error
    # when fewer subsets are sampled
    assert np.allclose(item_counts, 0.5, rtol=1 / (1 + np.log10(max_subsets)))

    # Test frequencies of set sizes. We divide counts by number of sets to allow
    # for larger errors at the extrema (low and high set sizes), where the
    # counts are lower.
    true_size_counts = np.array([np.math.comb(n, j) for j in range(n + 1)])
    assert np.allclose(
        true_size_counts / 2**n, size_counts / max_subsets, atol=1 / (1 + n)
    )


@pytest.mark.parametrize(
    "n, cond, exception",
    [
        (1, 2, ValueError),
        (0, 2, ValueError),
        (4, -2, ValueError),
        (10, 1, ValueError),
        (2, 10, None),
        (7, 23, None),
        (10, 2, None),
    ],
)
def test_random_matrix_with_condition_number(n, cond, exception):
    if exception is not None:
        with pytest.raises(exception):
            random_matrix_with_condition_number(n, cond)
    else:
        mat = random_matrix_with_condition_number(n, cond)
        assert np.isclose(np.linalg.cond(mat), cond), "Condition number does not match"
        assert np.array_equal(mat, mat.T), "Matrix is not symmetric"
        try:
            np.linalg.cholesky(mat)
        except np.linalg.LinAlgError:
            pytest.fail("Matrix is not positive definite")
