import logging
from functools import partial

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from tests.conftest import check_rank_correlation, check_total_value, check_values
from valuation.shapley import (
    combinatorial_exact_shapley,
    combinatorial_montecarlo_shapley,
    permutation_montecarlo_shapley,
    truncated_montecarlo_shapley,
)
from valuation.utils import MemcachedConfig, Utility
from valuation.utils.numeric import lower_bound_hoeffding
from valuation.utils.parallel import MapReduceJob, available_cpus, map_reduce


# noinspection PyTestParametrized
@pytest.mark.parametrize(
    "num_samples, fun, rtol, max_iterations",
    [
        (12, permutation_montecarlo_shapley, 0.1, 1),
        (8, combinatorial_montecarlo_shapley, 0.15, 1e3),
    ],
)
def test_analytic_montecarlo_shapley(analytic_shapley, fun, rtol, max_iterations):
    u, exact_values = analytic_shapley
    num_jobs = min(8, available_cpus())

    values, _ = fun(u, max_iterations=max_iterations, progress=False, num_jobs=num_jobs)

    check_values(values, exact_values, rtol=rtol)


@pytest.mark.parametrize(
    "num_samples, fun, delta, eps",
    [
        (12, permutation_montecarlo_shapley, 1e-2, 1e-2),
        (12, combinatorial_montecarlo_shapley, 1e-2, 1e-2),
    ],
)
def test_hoeffding_bound_montecarlo(analytic_shapley, fun, delta, eps, tolerate):
    """FIXME: This test passes but there are several unclear points.
    For example, map_reduce is called with num_jobs=num_runs. Is this correct?
    If I put num_jobs=jobs_per_run, map_reduce encounters errors since a utility
    is passed.
    Before coming back to this test, fix map_reduce interface."""
    u, exact_values = analytic_shapley
    jobs_per_run = min(6, available_cpus(), len(u.data))
    num_runs = min(3, available_cpus() // jobs_per_run)

    max_iterations = None
    max_iterations = lower_bound_hoeffding(delta=delta, eps=eps, score_range=1)

    print(
        f"test_montecarlo_shapley running for {max_iterations} iterations "
        f"per each of {jobs_per_run} jobs. Repeated {num_runs} times."
    )

    _fun = partial(
        fun,
        max_iterations=max_iterations // jobs_per_run,
        progress=False,
        num_jobs=jobs_per_run,
    )
    job = MapReduceJob.from_fun(_fun, lambda r: r[0][0])
    results = map_reduce(job, u, num_jobs=num_runs, num_runs=num_runs)

    max_failures = max(1, int(delta * len(results)))
    for values in results:
        with tolerate(max_failures=max_failures):
            # Trivial bound on total error using triangle inequality
            check_total_value(u, values, atol=len(u.data) * eps)
            check_rank_correlation(values, exact_values, threshold=0.9)


@pytest.mark.parametrize(
    "a, b, num_points, fun, score_type, rtol, max_iterations",
    [
        (2, 0, 10, permutation_montecarlo_shapley, "explained_variance", 0.5, 1000),
        (
            2,
            2,
            10,
            permutation_montecarlo_shapley,
            "neg_median_absolute_error",
            0.5,
            1000,
        ),
        (2, 0, 10, combinatorial_montecarlo_shapley, "explained_variance", 1, 2000),
        (
            2,
            2,
            10,
            combinatorial_montecarlo_shapley,
            "neg_median_absolute_error",
            1,
            2000,
        ),
    ],
)
def test_linear_montecarlo_shapley(
    linear_dataset, fun, score_type, rtol, max_iterations, memcache_client_config
):
    num_jobs = min(8, available_cpus())
    linear_utility = Utility(
        LinearRegression(),
        data=linear_dataset,
        scoring=score_type,
        cache_options=MemcachedConfig(client_config=memcache_client_config),
    )

    values, _ = fun(
        linear_utility, max_iterations=max_iterations, progress=False, num_jobs=num_jobs
    )
    exact_values = combinatorial_exact_shapley(linear_utility, progress=False)
    check_values(values, exact_values, rtol=rtol)


@pytest.mark.parametrize(
    "a, b, num_points, fun, score_type, max_iterations",
    [
        (2, 2, 20, permutation_montecarlo_shapley, "r2", 1000),
        (2, 3, 20, permutation_montecarlo_shapley, "explained_variance", 1000),
        (2, 3, 20, permutation_montecarlo_shapley, "neg_median_absolute_error", 1000),
    ],
)
def test_linear_montecarlo_with_outlier(
    linear_dataset,
    fun,
    score_type,
    max_iterations,
    memcache_client_config,
    total_atol=1e-2,
):
    outlier_idx = np.random.randint(len(linear_dataset.y_train))
    num_jobs = min(8, available_cpus())
    linear_dataset.y_train[outlier_idx] *= 100
    linear_utility = Utility(
        LinearRegression(),
        data=linear_dataset,
        scoring=score_type,
        cache_options=MemcachedConfig(client_config=memcache_client_config),
    )
    shapley_values, _ = fun(
        linear_utility, max_iterations=max_iterations, progress=False, num_jobs=num_jobs
    )
    check_total_value(linear_utility, shapley_values, atol=total_atol)

    assert int(list(shapley_values.keys())[0]) == outlier_idx


@pytest.mark.parametrize(
    "n_points, n_features, regressor, score_type, max_iterations",
    [
        (10, 3, RandomForestRegressor(n_estimators=2), "r2", 2000),
    ],
)
def test_random_forest(boston_dataset, regressor, score_type, max_iterations, rtol=1):
    num_jobs = min(8, available_cpus())
    rf_utility = Utility(
        regressor,
        data=boston_dataset,
        scoring=score_type,
        enable_cache=False,
    )
    permutation_values, _ = permutation_montecarlo_shapley(
        rf_utility, max_iterations=max_iterations, progress=False, num_jobs=num_jobs
    )
    combinatorial_values, _ = combinatorial_montecarlo_shapley(
        rf_utility, max_iterations=max_iterations, progress=False, num_jobs=num_jobs
    )
    check_values(permutation_values, combinatorial_values, rtol=rtol)


# noinspection PyTestParametrized
@pytest.mark.skip("Truncation not yet fully implemented")
@pytest.mark.parametrize("num_samples", [200])
def test_truncated_montecarlo_shapley(analytic_shapley, tolerate):
    u, exact_values = analytic_shapley
    num_cpus = min(available_cpus(), len(u.data))
    num_runs = 10
    delta = 0.01  # Sample bound holds with probability 1-𝛿
    eps = 0.05

    min_permutations = lower_bound_hoeffding(delta=delta, eps=eps, score_range=1)

    print(
        f"test_truncated_montecarlo_shapley running for {num_runs} runs "
        f" of max. {min_permutations} iterations each"
    )

    fun = partial(
        truncated_montecarlo_shapley,
        u=u,
        bootstrap_iterations=10,
        min_scores=5,
        score_tolerance=0.1,
        min_values=10,
        value_tolerance=eps,
        max_iterations=min_permutations,
        num_workers=num_cpus,
        progress=False,
    )
    results = []
    for i in range(num_runs):
        results.append(fun(run_id=i))

    max_failures = max(1, int(delta * len(results)))
    for values, _ in results:
        with tolerate(max_failures=max_failures):
            # Trivial bound on total error using triangle inequality
            check_total_value(u, values, atol=len(u.data) * eps)
            check_rank_correlation(values, exact_values, threshold=0.8)
