from __future__ import annotations

import math
from typing import Callable, Literal

import numpy as np
import pytest
from scipy.special import gammaln

from pydvl.utils.numeric import (
    complement,
    log_running_moments,
    logcomb,
    logexp,
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
    np.testing.assert_allclose(item_counts / 2**n, 0.5)
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
    np.testing.assert_allclose(
        true_size_counts, size_counts / max_subsets, atol=1 / (1 + n)
    )


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
        (
            np.testing.assert_allclose(np.linalg.cond(mat), cond, atol=1e-5),
            "Condition number does not match",
        )
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


@pytest.mark.parametrize(
    "n, k, expected",
    [
        (5, 2, np.log(10)),
        (10, 5, np.log(252)),
        (20, 10, np.log(184756)),
        (0, 0, 0.0),
        (1, 0, 0.0),
        (1, 1, 0.0),
        (100, 50, gammaln(101) - gammaln(51) - gammaln(51)),
    ],
    ids=[
        "C(5,2) = 10",
        "C(10,5) = 252",
        "C(20,10) = 184756",
        "C(0,0) = 1",
        "C(1,0) = 1",
        "C(1,1) = 1",
        "Large values",
    ],
)
def test_logcomb(n, k, expected):
    result = logcomb(n, k)
    np.testing.assert_allclose(
        result, expected, atol=1e-6, err_msg=f"Failed for n={n}, k={k}"
    )


@pytest.mark.parametrize(
    "n, k",
    [(5, -1), (-5, 2), (4, 5)],
    ids=["Negative k", "Negative n", "k > n"],
)
def test_logcomb_invalid(n, k):
    with pytest.raises(ValueError):
        logcomb(n, k)


@pytest.mark.parametrize("unbiased", [False, True], ids=["Population", "Sample"])
def test_running_moments(unbiased: bool):
    """Test that running moments are correct."""

    ddof = 1 if unbiased else 0
    n_samples, n_values = 15, 1000
    max_init_values = 100

    # Generate sequences of varying lengths and compute their moments
    values = [
        np.random.randn(np.random.randint(0, max_init_values)) for _ in range(n_samples)
    ]
    means = np.array([np.mean(v) for v in values])
    # With ddof=1 and an empty sequence, var() returns NaN -> replace by 0
    variances = np.nan_to_num(np.array([np.var(v, ddof=ddof) for v in values]), nan=0.0)
    # Each of the n_samples values has been computed from a sequence of length counts[i]
    counts = np.array([len(v) for v in values], dtype=np.int_)

    # successively add values to the running moments
    data = np.random.randn(n_samples, n_values).T
    for new_values in data:
        new_means = np.zeros_like(means)
        new_variances = np.zeros_like(variances)
        for i in range(n_samples):
            new_means[i], new_variances[i] = running_moments(
                means[i], variances[i], counts[i], new_values[i], unbiased
            )
        means, variances = new_means, new_variances
        counts += 1

        # If running_moments were an ufunc:
        # new_means, new_variances = running_moments(
        #     means, variances, counts, new_values, unbiased
        # )
        # means, variances = new_means, new_variances
        # counts += 1

        values = [
            np.concatenate([values[j], [new_values[j]]]) for j in range(n_samples)
        ]

        true_means = [np.mean(vv) for vv in values]
        true_variances = [np.var(vv, ddof=ddof) for vv in values]
        np.testing.assert_allclose(means, true_means, atol=1e-5)
        np.testing.assert_allclose(variances, true_variances, atol=1e-5)


@pytest.mark.parametrize("unbiased", [False, True], ids=["Population", "Sample"])
def test_running_moment_initialization(unbiased: bool):
    """We often use running moments for updates in ValuationResult where the means
    and variances are initialized to zero. This test makes sure that case is handled
    correctly.

    """
    ddof = 1 if unbiased else 0
    got_mean, got_var = running_moments(
        previous_avg=0.0,
        previous_variance=0.0,
        count=0,
        new_value=1.0,
        unbiased=unbiased,
    )

    assert got_mean == 1.0
    assert got_var == 0.0

    got_mean, got_var = running_moments(
        previous_avg=got_mean,
        previous_variance=got_var,
        count=1,
        new_value=2.0,
        unbiased=unbiased,
    )

    np.testing.assert_allclose(got_mean, 1.5)
    np.testing.assert_allclose(got_var, np.var([1.0, 2.0], ddof=ddof))


@pytest.mark.parametrize("unbiased", [False, True], ids=["Population", "Sample"])
def test_running_moments_log_initialization(unbiased):
    """
    Mimics the running moments test case for log_running_moments.

    For the first update with a single value (1.0), since we are using the
    unbiased (ddof=1) estimator, the variance is undefined (nan). On the second
    update, with values 1.0 and 2.0, the unbiased variance should equal 0.5.
    """

    # First update: use x = 1.0 (so log(x) = log(1) = 0.0).
    # When count==0, the previous log accumulators are not used.
    mean, var, log_sum_pos, log_sum_neg, log_sum2 = log_running_moments(
        previous_log_sum_pos=-np.inf,
        previous_log_sum_neg=-np.inf,
        previous_log_sum2=-np.inf,
        count=0,
        new_log_value=0.0,  # log(1.0)
        new_sign=0,
        unbiased=unbiased,
    )

    np.testing.assert_allclose(mean, 1.0)
    assert var == 0.0

    # Second update: use x = 2.0 (so log(x) = log(2)).
    mean, var, log_sum_pos, log_sum_neg, log_sum2 = log_running_moments(
        previous_log_sum_pos=log_sum_pos,
        previous_log_sum_neg=log_sum_neg,
        previous_log_sum2=log_sum2,
        count=1,
        new_log_value=np.log(2.0),
        new_sign=1,
        unbiased=unbiased,
    )
    # For samples [1.0, 2.0], the mean is (1+2)/2 = 1.5.
    np.testing.assert_allclose(mean, 1.5)
    # The population variance would be 0.25, but the unbiased estimator (ddof=1)
    # scales this by n/(n-1) = 2/1, so the expected variance is 0.5.
    if unbiased:
        np.testing.assert_allclose(var, 0.5)
    else:
        np.testing.assert_allclose(var, 0.25)


@pytest.mark.parametrize(
    "prev_log_sum_pos, prev_log_sum_neg, prev_log_sum2, count, new_log_value, new_sign, unbiased, expected_mean, expected_variance",
    [
        (-np.inf, -np.inf, -np.inf, 0, np.log(2.0), 1, True, 2.0, 0.0),
        (-np.inf, -np.inf, -np.inf, 0, np.log(3.0), -1, True, -3.0, 0.0),
        (-np.inf, -np.inf, -np.inf, 0, np.log(0.0), 0, True, 0.0, 0.0),
        (np.log(2.0), -np.inf, np.log(4.0), 1, np.log(3.0), 1, True, 2.5, 0.5),
        (np.log(2.0), -np.inf, np.log(4.0), 1, np.log(1.0), -1, True, 0.5, 4.5),
        (
            np.log(2.0),
            np.log(1.0),
            np.log(5.0),
            2,
            np.log(3.0),
            1,
            True,
            4 / 3,
            39 / 9,
        ),
        (
            np.log(2.0),
            np.log(1.0),
            np.log(5.0),
            2,
            np.log(3.0),
            1,
            False,
            4 / 3,
            26 / 9,
        ),
    ],
    ids=[
        "First update with a positive value",
        "First update with a negative value",
        "First update with a zero",
        "Second update, adding a positive number",
        "Second update, adding a negative number",
        "Unbiased variance correction (with at least two elements)",
        "Without unbiased correction (population variance)",
    ],
)
def test_log_running_moments(
    prev_log_sum_pos: float,
    prev_log_sum_neg: float,
    prev_log_sum2: float,
    count: int,
    new_log_value: float,
    new_sign: int,
    unbiased: bool,
    expected_mean: float,
    expected_variance: float,
):
    mean, variance, _, _, _ = log_running_moments(
        prev_log_sum_pos,
        prev_log_sum_neg,
        prev_log_sum2,
        count,
        new_log_value,
        new_sign,
        unbiased,
    )

    np.testing.assert_allclose(
        mean,
        expected_mean,
        atol=1e-8,
        err_msg=f"Expected mean {expected_mean}, got {mean}",
    )
    np.testing.assert_allclose(
        variance,
        expected_variance,
        atol=1e-8,
        err_msg=f"Expected variance {expected_variance}, got {variance}",
    )


@pytest.mark.parametrize("unbiased", [False, True], ids=["Population", "Sample"])
@pytest.mark.parametrize(
    "weight, coeff, logweight, logcoeff, expected_mean_and_variance",
    [
        (
            lambda n, k: 2 ** int(-(n - 1)),
            lambda n, k: n * math.comb(n - 1, k),
            lambda n, k: logexp(2, -(n - 1)),
            lambda n, k: math.log(n) + logcomb(n - 1, k),
            "compute",
        ),
        (
            lambda n, k: 1 / (n * math.comb(n - 1, k)),
            lambda n, k: n * math.comb(n - 1, k),
            lambda n, k: -math.log(n) - logcomb(n - 1, k),
            lambda n, k: math.log(n) + logcomb(n - 1, k),
            (1.0, 0.0),
        ),
    ],
    ids=["Large quotients", "Cancellations"],
)
@pytest.mark.parametrize(
    "max_n, num_n_values, num_k_values",
    [(20, 10, 10), (1000, 100, 10), (10000, 100, 20)],
    ids=["Small n", "Medium n", "Large n"],
)
def test_log_running_moments_comb(
    weight: Callable[[int, int], float],
    coeff: Callable[[int, int], float],
    logweight: Callable[[int, int], float],
    logcoeff: Callable[[int, int], float],
    expected_mean_and_variance: Literal["compute"] | tuple[float, float],
    max_n: int,
    num_n_values: int,
    num_k_values: int,
    unbiased: bool,
):
    ddof = 1 if unbiased else 0
    n_values = np.random.randint(1, max_n, size=num_n_values)

    for n in n_values:
        log_sum_pos = -np.inf  # log(0)
        log_sum_neg = -np.inf  # log(0)
        log_sum2 = -np.inf
        means = []
        variances = []
        for count, k in enumerate(
            k_values := np.random.randint(0, n, size=num_k_values)
        ):  # type: int, int
            new_log_val = logweight(n, k) + logcoeff(n, k)
            mean, var, log_sum_pos, log_sum_neg, log_sum2 = log_running_moments(
                log_sum_pos,
                log_sum_neg,
                log_sum2,
                count,
                new_log_val,
                new_sign=1,
                unbiased=unbiased,
            )
            means.append(mean)
            variances.append(var)

        # Compare with moments using direct summation
        def safe_coeff(n, k):
            with np.errstate(over="raise"):
                try:
                    return coeff(n, k)
                except FloatingPointError:
                    raise OverflowError(f"Overflow in coeff({n=}, {k=})")

        if expected_mean_and_variance == "compute":
            try:
                xs = np.array(
                    [weight(n, k) * safe_coeff(n, k) for k in k_values],
                    dtype=np.float64,
                )
                true_mean = np.mean(xs)
                true_variance = np.var(xs, ddof=ddof)
            except OverflowError as e:
                pytest.skip(f"Overflow in direct computation: {str(e)} for {n=}")
                continue
        else:
            true_mean, true_variance = expected_mean_and_variance

        np.testing.assert_allclose(means[-1], true_mean, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(variances[-1], true_variance, rtol=1e-10, atol=1e-10)


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
