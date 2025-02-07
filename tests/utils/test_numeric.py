import math

import numpy as np
import pytest

from pydvl.utils.numeric import (
    complement,
    powerset,
    random_matrix_with_condition_number,
    random_powerset,
    random_powerset_label_min,
    random_subset,
    random_subset_of_size,
    running_moments,
)
from pydvl.utils.types import Seed


@pytest.mark.parametrize(
    "include, exclude, expected",
    [
        (np.array([1, 2, 3, 4, 5]), [2, 3], np.array([1, 4, 5])),
        (np.array([1, 2, 3, 4, 5]), [None], np.array([1, 2, 3, 4, 5])),
        (np.array([1, 2, 3, 4, 5]), [], np.array([1, 2, 3, 4, 5])),
        (np.array([1, 2, 3, 4, 5]), [1, 2, 3, 4, 5], np.array([])),
        (np.array([]), [1, 2], np.array([])),
    ],
)
def test_complement(include, exclude, expected):
    result = complement(include, exclude)
    assert np.array_equal(result, expected)


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


@pytest.mark.parametrize("n, max_subsets", [(1, 10), (10, 2**10), (5, 2**7), (0, 1)])
@pytest.mark.parametrize("q", [0.0, 0.1, 0.26, 0.49, 0.5, 0.6, 1])
def test_random_powerset(n, max_subsets, q):
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
    item_counts = np.zeros_like(s, dtype=np.float64)
    size_counts = np.zeros(n + 1)
    for subset in random_powerset(s, n_samples=max_subsets, q=q):
        size_counts[len(subset)] += 1
        for item in subset:
            item_counts[item] += 1

    eps = 0.1
    item_frequencies = item_counts / max_subsets

    assert np.count_nonzero(
        np.abs(item_frequencies - q) > eps
    ) < max_subsets * 2 * np.exp(-2 * max_subsets * eps**2)

    # True distribution of set sizes follows a binomial distribution
    # with parameters n and q
    def binomial_pmf(j: int):
        return math.comb(n, j) * q**j * (1 - q) ** (n - j)

    true_size_counts = np.array([binomial_pmf(j) for j in range(n + 1)])
    assert np.allclose(true_size_counts, size_counts / max_subsets, atol=1 / (1 + n))


@pytest.mark.parametrize("n, max_subsets", [(10, 2**10)])
def test_random_powerset_reproducible(n, max_subsets, seed):
    """
    Test that the same seeds produce the same results, and different seeds produce
    different results for method :func:`random_powerset`.
    """
    n_collisions = _count_random_powerset_generator_collisions(
        n, max_subsets, seed, seed
    )
    assert n_collisions == max_subsets


@pytest.mark.parametrize("n, max_subsets", [(10, 2**10)])
def test_random_powerset_stochastic(n, max_subsets, seed, seed_alt, collision_tol):
    """
    Test that the same seeds produce the same results, and different seeds produce
    different results for method :func:`random_powerset`.
    """
    n_collisions = _count_random_powerset_generator_collisions(
        n, max_subsets, seed, seed_alt
    )
    assert n_collisions / max_subsets < collision_tol


def _count_random_powerset_generator_collisions(
    n: int, max_subsets: int, seed: Seed, seed_alt: Seed
):
    """
    Count the number of collisions between two generators of random subsets of a set
    with `n` elements, each generating `max_subsets` subsets, using two different seeds.

    Args:
        n: number of elements in the set.
        max_subsets: number of subsets to generate.
        seed: Seed for the first generator.
        seed_alt: Seed for the second generator.

    Returns:
        Number of collisions between the two generators.
    """
    s = np.arange(n)
    parallel_subset_generators = zip(
        random_powerset(s, n_samples=max_subsets, seed=seed),
        random_powerset(s, n_samples=max_subsets, seed=seed_alt),
    )
    n_collisions = sum(
        map(lambda t: set(t[0]) == set(t[1]), parallel_subset_generators)
    )
    return n_collisions


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
    "n, size",
    [(10, 3), (1000, 40)],
)
def test_random_subset_of_size_stochastic_unequal(n, size, seed, seed_alt):
    """
    Test that the same seeds produce the same results, and different seeds produce
    different results for method :func:`random_subset_of_size`.
    """
    s = np.arange(n)
    subset_1 = random_subset_of_size(s, size=size, seed=seed)
    subset_2 = random_subset_of_size(s, size=size, seed=seed_alt)
    assert set(subset_1) != set(subset_2)


@pytest.mark.parametrize(
    "n, size",
    [(10, 3), (1000, 40)],
)
def test_random_subset_of_size_stochastic_equal(n, size, seed):
    """
    Test that the same seeds produce the same results, and different seeds produce
    different results for method :func:`random_subset_of_size`.
    """
    s = np.arange(n)
    subset_1 = random_subset_of_size(s, size=size, seed=seed)
    subset_2 = random_subset_of_size(s, size=size, seed=seed)
    assert set(subset_1) == set(subset_2)


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


@pytest.mark.parametrize(
    "n, cond",
    [
        (2, 10),
        (7, 23),
        (10, 2),
    ],
)
def test_random_matrix_with_condition_number_reproducible(n, cond, seed):
    mat_1 = random_matrix_with_condition_number(n, cond, seed=seed)
    mat_2 = random_matrix_with_condition_number(n, cond, seed=seed)
    assert np.all(mat_1 == mat_2)


@pytest.mark.parametrize(
    "n, cond",
    [
        (2, 10),
        (7, 23),
        (10, 2),
    ],
)
def test_random_matrix_with_condition_number_stochastic(n, cond, seed, seed_alt):
    mat_1 = random_matrix_with_condition_number(n, cond, seed=seed)
    mat_2 = random_matrix_with_condition_number(n, cond, seed=seed_alt)
    assert np.any(mat_1 != mat_2)


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


def test_running_moment_initialization():
    """We often use running moments for updates in ValuationResult where the means
    and variances are initialized to zero. This test makes sure that case is handled
    correctly.

    """
    got_mean, got_var = running_moments(
        previous_avg=0.0, previous_variance=0.0, count=0, new_value=1.0
    )

    assert got_mean == 1.0
    assert got_var == 0.0  # TODO: shouldn't this be undefined?

    got_mean, got_var = running_moments(
        previous_avg=got_mean, previous_variance=got_var, count=1, new_value=2.0
    )

    assert np.isclose(got_mean, 1.5)
    assert np.isclose(got_var, np.var([1.0, 2.0]))


@pytest.mark.parametrize(
    "min_elements_per_label,num_elements_per_label,num_labels,check_num_samples",
    [(0, 10, 3, 1000), (1, 10, 3, 1000), (2, 10, 3, 1000)],
)
def test_random_powerset_label_min(
    min_elements_per_label: int,
    num_elements_per_label: int,
    num_labels: int,
    check_num_samples: int,
):
    s = np.arange(num_labels * num_elements_per_label)
    labels = np.arange(num_labels).repeat(num_elements_per_label)

    for idx, subset in enumerate(
        random_powerset_label_min(s, labels, min_elements_per_label)
    ):
        assert np.all(np.isin(subset, s))
        for group in np.unique(labels):
            assert np.sum(group == labels[subset]) >= min_elements_per_label

        if idx == check_num_samples:
            break


@pytest.mark.flaky(reruns=1)
def test_size_of_random_subset():
    """This test discovered an actual bug where (1 - q) was used instead of q."""
    subset = random_subset(np.arange(10), q=0)
    assert len(subset) == 0

    subset = random_subset(np.arange(10), q=1)
    assert len(subset) == 10
