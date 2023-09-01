import numpy as np
import pytest

from pydvl.utils.numeric import (
    powerset,
    random_matrix_with_condition_number,
    random_powerset,
    random_powerset_group_conditional,
    random_subset_of_size,
    running_moments,
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


# TODO: include tests for multiple values of q, including 0 and 1
@pytest.mark.parametrize(
    "n, max_subsets", [(1, 10), (10, 2**10), (5, 2**7), (0, 1)]
)
def test_random_powerset(n, max_subsets):
    """Tests frequency of items in sets and frequencies of set sizes.

    By Hoeffding for a Bernoulli, we have for each item in the set:
        P(|ΣX-mq| > mε) < 2 exp(-2mε^2)
    where m=max_subsets
    For m=100, q=0.5, ε=0.1 this means that the count for any item will be
    within ±1 of 5, with probability at least ≈ 1 - 2exp(-2) ≈ 0.73 which is
    admittedly a rather crappy bound (and vacuous for low values of max_subsets)

    For the frequencies of set sizes, we divide counts by number of sets to
    allow for larger errors at the extrema (low and high set sizes), where the
    counts are lower.
    """
    s = np.arange(n)
    item_counts = np.zeros_like(s, dtype=np.float_)
    size_counts = np.zeros(n + 1)
    for subset in random_powerset(s, n_samples=max_subsets):
        size_counts[len(subset)] += 1
        for item in subset:
            item_counts[item] += 1
    q = 0.5
    eps = 0.1
    item_frequencies = item_counts / max_subsets

    assert np.count_nonzero(
        np.abs(item_frequencies - q) > eps
    ) < max_subsets * 2 * np.exp(-2 * max_subsets * eps**2)

    true_size_counts = np.array([np.math.comb(n, j) for j in range(n + 1)])
    assert np.allclose(
        true_size_counts / 2**n, size_counts / max_subsets, atol=1 / (1 + n)
    )


@pytest.mark.parametrize(
    "n, size, exception",
    [(0, 0, None), (0, 1, ValueError), (10, 0, None), (10, 3, None), (1000, 40, None)],
)
def test_random_subset_of_size(n, size, exception):
    s = np.arange(n)
    if exception:
        with pytest.raises(exception):
            ss = random_subset_of_size(s, size=size)
    else:
        ss = random_subset_of_size(s, size=size)
        assert len(ss) == size
        assert np.all([x in s for x in ss])


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


def test_running_moments():
    """Test that running moments are correct."""
    n_samples, n_values = 15, 1000
    max_init_values = 100
    # Generate sequences of varying lengths and compute their moments
    values = [
        np.random.randn(np.random.randint(0, max_init_values)) for _ in range(n_samples)
    ]
    means = np.array([np.mean(v) for v in values])
    variances = np.array([np.var(v) for v in values])
    # Each of the n_samples values has been computed from a sequence of length counts[i]
    counts = np.array([len(v) for v in values], dtype=np.int_)

    # successively add values to the running moments
    data = np.random.randn(n_samples, n_values)
    for i in range(n_values):
        new_values = data[:, i]
        new_means, new_variances = running_moments(means, variances, counts, new_values)
        means, variances = new_means, new_variances
        counts += 1

        values = [
            np.concatenate([values[j], [new_values[j]]]) for j in range(n_samples)
        ]

        true_means = [np.mean(vv) for vv in values]
        true_variances = [np.var(vv) for vv in values]
        assert np.allclose(means, true_means)
        assert np.allclose(variances, true_variances)


@pytest.mark.parametrize(
    "min_elements,elements_per_group,num_groups,check_num_samples",
    [(1, 10, 3, 3), (2, 10, 3, 3)],
)
def test_random_powerset_group_conditional(
    min_elements: int,
    elements_per_group: int,
    num_groups: int,
    check_num_samples: int,
):
    s = np.arange(num_groups * elements_per_group)
    groups = np.arange(num_groups).repeat(elements_per_group)

    for idx, subset in enumerate(
        random_powerset_group_conditional(s, groups, min_elements)
    ):
        assert np.all(np.isin(subset, s))
        assert np.all(np.unique(groups[subset]) == np.unique(groups))

        for group in np.unique(groups):
            assert np.sum(group == groups[subset]) >= min_elements

        if idx == check_num_samples:
            break
