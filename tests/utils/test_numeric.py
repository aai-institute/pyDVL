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

    assert set(powerset([])) == {()}
    assert set(powerset((1, 2))) == {(), (1,), (1, 2), (2,)}

    # Check correct number of sets of each size
    n = 10
    sizes = np.zeros(n + 1)
    for s in powerset(range(n)):
        sizes[len(s)] += 1

    assert all([np.math.comb(n, j) for j in range(n + 1)] == sizes)


@pytest.mark.parametrize("n, max_subsets", [(0, 10), (1, 1e3)])
def test_random_powerset(n, max_subsets, count_amplifier=3):
    """
    Tests that random_powerset samples the same items as the powerset method and
    with constant frequency.
    Sampling a number max_subsets of sets, we need to check that their relative frequency
    is the same, up to sampling errors. To do so, we count the occurrence of each set, and
    assert that the difference in count between the max and the min is much smaller than
    the mean value count. More precisely, we assert that
    (maximum_count - minimum_count) * count_amplifier < mean_count
    where count_amplifier must be bigger than 1.
    """
    s = np.arange(1, n + 1)
    result = random_powerset(
        s,
        max_subsets=max_subsets,
    )

    result_exact = set(powerset(s))
    count_powerset = {key: 0 for key in result_exact}

    for res_pow in result:
        res_pow = tuple(np.sort(res_pow))
        count_powerset[tuple(res_pow)] += 1
    value_counts = list(count_powerset.values())
    assert count_amplifier * (np.max(value_counts) - np.min(value_counts)) < np.mean(
        value_counts
    )


@pytest.mark.parametrize(
    "x, y, expected",
    [
        ([], [], ValueError),
        ([1], [1], TypeError),
        ([1, 2, 3], [1, 2, 3], 1.0),
        ([1, 2, 3], [3, 2, 1], -1.0),
        (np.arange(1, 4), np.arange(4, 7), ValueError),
        # FIXME: non deterministic test
        pytest.param(
            np.random.permutation(np.arange(100)),
            np.random.permutation(np.arange(100)),
            (0.0, 0.1),
            marks=pytest.mark.skip("This test case is flaky."),
        ),
    ],
)
def test_spearman(x, y, expected):
    if isinstance(expected, float):
        x = np.array(x, dtype=int)
        y = np.array(y, dtype=int)
        assert spearman(x, y) == expected
    elif isinstance(expected, tuple):
        value, atol = expected
        assert np.isclose(spearman(x, y), value, atol=atol)
    else:
        with pytest.raises(expected):
            spearman(x, y)


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
