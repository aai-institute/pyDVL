import logging
from typing import List, cast

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from pydvl.shapley import (
    combinatorial_exact_shapley,
    combinatorial_montecarlo_shapley,
    permutation_montecarlo_shapley,
    truncated_montecarlo_shapley,
)
from pydvl.utils import GroupedDataset, MemcachedConfig, Utility
from pydvl.utils.numeric import lower_bound_hoeffding
from tests.conftest import check_rank_correlation, check_total_value, check_values

log = logging.getLogger(__name__)


# noinspection PyTestParametrized
@pytest.mark.parametrize(
    "num_samples, fun, rtol, max_iterations",
    [
        (12, permutation_montecarlo_shapley, 0.1, 10),
        (8, combinatorial_montecarlo_shapley, 0.15, 3e3),
    ],
)
def test_analytic_montecarlo_shapley(
    analytic_shapley, fun, rtol, max_iterations, n_jobs
):
    u, exact_values = analytic_shapley

    values, _ = fun(
        u, max_iterations=int(max_iterations), progress=False, n_jobs=n_jobs
    )

    check_values(values, exact_values, rtol=rtol)


@pytest.mark.parametrize(
    "num_samples, delta, eps",
    [
        (12, 1e-2, 1e-1),
    ],
)
@pytest.mark.parametrize("n_jobs", [4])
@pytest.mark.parametrize(
    "fun", [permutation_montecarlo_shapley, combinatorial_montecarlo_shapley]
)
def test_hoeffding_bound_montecarlo(
    analytic_shapley, fun, n_jobs, delta, eps, tolerate
):
    u, exact_values = analytic_shapley

    max_iterations = lower_bound_hoeffding(delta=delta, eps=eps, score_range=1)

    values, _ = fun(
        u=u,
        max_iterations=max_iterations,
        n_jobs=n_jobs,
    )

    with tolerate(max_failures=1):
        # Trivial bound on total error using triangle inequality
        check_total_value(u, values, atol=len(u.data) * eps)
        check_rank_correlation(values, exact_values, threshold=0.9)


@pytest.mark.parametrize(
    "a, b, num_points, fun, score_type, rtol, max_iterations",
    [
        (2, 0, 20, permutation_montecarlo_shapley, "explained_variance", 0.2, 5000),
        (
            2,
            2,
            12,
            truncated_montecarlo_shapley,
            "r2",
            0.2,
            5000,
        ),
        (
            2,
            0,
            12,
            combinatorial_montecarlo_shapley,
            "explained_variance",
            0.2,
            2**11,
        ),
    ],
)
def test_linear_montecarlo_shapley(
    linear_dataset,
    fun,
    score_type,
    rtol,
    max_iterations,
    memcache_client_config,
    n_jobs,
    total_atol=1,
):
    linear_utility = Utility(
        LinearRegression(),
        data=linear_dataset,
        scoring=score_type,
        cache_options=MemcachedConfig(client_config=memcache_client_config),
    )
    values, _ = fun(
        linear_utility,
        max_iterations=int(max_iterations),
        progress=False,
        n_jobs=n_jobs,
    )
    exact_values = combinatorial_exact_shapley(linear_utility, progress=False)
    log.debug(f"These are the exact values: {exact_values}")
    log.debug(f"These are the predicted values: {values}")
    # PyCharm seems to believe that the dictview converts to List[str], so we cast
    exact_values_list = cast(List[float], list(exact_values.values()))
    atol = (exact_values_list[-1] - exact_values_list[0]) / 10
    check_values(values, exact_values, rtol=rtol, atol=atol)
    check_total_value(linear_utility, values, atol=total_atol)


@pytest.mark.parametrize(
    "a, b, num_points, fun, score_type, max_iterations, total_atol",
    [
        (2, 3, 20, permutation_montecarlo_shapley, "r2", 500, 1),
        (
            2,
            3,
            20,
            truncated_montecarlo_shapley,
            "neg_median_absolute_error",
            500,
            3,
        ),
        (2, 3, 20, truncated_montecarlo_shapley, "r2", 500, 200),
    ],
)
def test_linear_montecarlo_with_outlier(
    linear_dataset,
    fun,
    score_type,
    max_iterations,
    memcache_client_config,
    total_atol,
    n_jobs,
):
    outlier_idx = np.random.randint(len(linear_dataset.y_train))
    linear_dataset.y_train[outlier_idx] -= 100
    linear_utility = Utility(
        LinearRegression(),
        data=linear_dataset,
        scoring=score_type,
        cache_options=MemcachedConfig(client_config=memcache_client_config),
    )
    shapley_values, sval_std = fun(
        linear_utility,
        max_iterations=max_iterations,
        progress=False,
        n_jobs=n_jobs,
    )
    log.debug(f"{shapley_values=}")
    log.debug(f"{outlier_idx=}")
    check_total_value(linear_utility, shapley_values, atol=total_atol)

    assert int(list(shapley_values.keys())[0]) == outlier_idx


@pytest.mark.parametrize(
    "a, b, num_points, num_groups, fun, score_type, rtol, max_iterations",
    [
        (
            2,
            2,
            20,
            4,
            permutation_montecarlo_shapley,
            "r2",
            0.2,
            5000,
        ),
        (2, 0, 200, 5, truncated_montecarlo_shapley, "explained_variance", 0.2, 1000),
        (2, 0, 200, 5, truncated_montecarlo_shapley, "r2", 0.2, 1000),
    ],
)
def test_grouped_linear_montecarlo_shapley(
    linear_dataset,
    num_groups,
    fun,
    score_type,
    rtol,
    max_iterations,
    memcache_client_config,
    n_jobs,
):
    data_groups = np.random.randint(0, num_groups, len(linear_dataset))
    grouped_linear_dataset = GroupedDataset.from_dataset(linear_dataset, data_groups)
    grouped_linear_utility = Utility(
        LinearRegression(),
        data=grouped_linear_dataset,
        scoring=score_type,
        cache_options=MemcachedConfig(client_config=memcache_client_config),
    )

    values, _ = fun(
        grouped_linear_utility,
        max_iterations=max_iterations,
        progress=False,
        n_jobs=n_jobs,
    )
    exact_values = combinatorial_exact_shapley(grouped_linear_utility, progress=False)
    log.debug(f"These are the exact values: {exact_values}")
    log.debug(f"These are the predicted values: {values}")
    # PyCharm seems to believe that the dictview converts to List[str], so we cast
    exact_values_list = cast(List[float], list(exact_values.values()))
    atol = (exact_values_list[-1] - exact_values_list[0]) / 30
    check_values(values, exact_values, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "n_points, n_features, regressor, score_type, max_iterations",
    [
        (10, 3, RandomForestRegressor(n_estimators=2), "r2", 20),
        (10, 3, DecisionTreeRegressor(), "r2", 20),
    ],
)
def test_random_forest(
    boston_dataset,
    regressor,
    score_type,
    max_iterations,
    memcache_client_config,
    n_jobs,
):
    """This test checks that random forest can be trained in our library.
    Originally, it would also check that the returned values match between
    permutation and combinatorial Monte Carlo, but this was taking too long in the
    pipeline and was removed."""
    rf_utility = Utility(
        regressor,
        data=boston_dataset,
        scoring=score_type,
        enable_cache=True,
        cache_options=MemcachedConfig(
            client_config=memcache_client_config,
            allow_repeated_evaluations=True,
            rtol_stderr=1,
            time_threshold=0,
        ),
    )

    _, _ = truncated_montecarlo_shapley(
        rf_utility,
        max_iterations=max_iterations,
        progress=False,
        n_jobs=n_jobs,
    )
