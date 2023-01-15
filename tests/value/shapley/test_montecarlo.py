import logging
from typing import Union

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from pydvl.utils import GroupedDataset, MemcachedConfig, Scorer, Utility
from pydvl.utils.numeric import (
    num_samples_permutation_hoeffding,
    squashed_r2,
    squashed_variance,
)
from pydvl.value.shapley.montecarlo import (
    combinatorial_montecarlo_shapley,
    owen_sampling_shapley,
    permutation_montecarlo_shapley,
    truncated_montecarlo_shapley,
)
from pydvl.value.shapley.naive import combinatorial_exact_shapley

from .. import check_rank_correlation, check_total_value, check_values

log = logging.getLogger(__name__)


# noinspection PyTestParametrized
@pytest.mark.parametrize(
    "num_samples, fun, rtol, n_iterations, kwargs",
    [
        (12, permutation_montecarlo_shapley, 0.1, 10, {}),
        # FIXME! it should be enough with 2**(len(data)-1) samples
        (8, combinatorial_montecarlo_shapley, 0.2, 2**10, {}),
        (12, truncated_montecarlo_shapley, 0.1, 10, {"coordinator_update_period": 1}),
        (12, owen_sampling_shapley, 0.1, 4, {"max_q": 200, "method": "antithetic"}),
        (12, owen_sampling_shapley, 0.1, 4, {"max_q": 200, "method": "standard"}),
    ],
)
def test_analytic_montecarlo_shapley(
    num_samples, analytic_shapley, fun, rtol, n_iterations, kwargs, parallel_config
):
    u, exact_values = analytic_shapley

    values = fun(
        u,
        n_iterations=int(n_iterations),
        config=parallel_config,
        progress=False,
        n_jobs=1,
        **kwargs,
    )

    check_values(values, exact_values, rtol=rtol)


@pytest.mark.parametrize("num_samples, delta, eps", [(8, 0.1, 0.1)])
@pytest.mark.parametrize(
    "fun", [permutation_montecarlo_shapley, combinatorial_montecarlo_shapley]
)
def test_hoeffding_bound_montecarlo(
    num_samples, analytic_shapley, fun, delta: float, eps: float, tolerate
):
    u, exact_values = analytic_shapley

    n_iterations = num_samples_permutation_hoeffding(delta=delta, eps=eps, u_range=1)

    for _ in range(10):
        with tolerate(max_failures=int(10 * delta)):
            values = fun(u=u, n_iterations=n_iterations, n_jobs=1)
            # Trivial bound on total error using triangle inequality
            check_total_value(u, values, atol=len(u.data) * eps)
            check_rank_correlation(values, exact_values, threshold=0.8)


@pytest.mark.parametrize(
    "a, b, num_points", [(2, 0, 21)]  # training set will have 0.3 * 21 = 6 samples
)
@pytest.mark.parametrize(
    "scorer, rtol", [(squashed_r2, 0.25), (squashed_variance, 0.25)]
)
@pytest.mark.parametrize(
    "fun, n_iterations, kwargs",
    [
        # FIXME: Hoeffding says 400 should be enough
        (permutation_montecarlo_shapley, 600, {}),
        (truncated_montecarlo_shapley, 500, {"coordinator_update_period": 1}),
        (combinatorial_montecarlo_shapley, 2**11, {}),
        (owen_sampling_shapley, 4, {"max_q": 300, "method": "standard"}),
        # FIXME: antithetic breaks for non-deterministic u
        # (owen_sampling_shapley, 4, {"max_q": 300, "method": "antithetic"}),
    ],
)
def test_linear_montecarlo_shapley(
    linear_dataset,
    scorer: Scorer,
    rtol: float,
    fun,
    n_iterations: float,
    memcache_client_config,
    kwargs: dict,
):
    """Tests values for all methods using a linear dataset.

    For permutation and truncated montecarlo, the rtol for each scorer is chosen
    so that the number of samples selected is just above the (ε,δ) bound for ε =
    rtol, δ=0.001 and the range corresponding to each score. This means that
    roughly once every 1000/num_methods runs the test will fail.

    FIXME:
     - For permutation, we must increase the number of samples above that what
       is done for truncated, this is probably due to the averaging done by the
       latter to reduce variance
     - We don't have a bound for Owen.
    NOTE:
     - The variance in the combinatorial method is huge, so we need lots of
       samples

    """
    u = Utility(
        LinearRegression(),
        data=linear_dataset,
        scoring=scorer,
        cache_options=MemcachedConfig(client_config=memcache_client_config),
    )

    exact_values = combinatorial_exact_shapley(u, progress=False)
    values = fun(u, n_iterations=int(n_iterations), progress=False, n_jobs=1, **kwargs)

    check_values(values, exact_values, rtol=rtol)
    check_total_value(u, values, rtol=rtol)  # FIXME, could be more than rtol


@pytest.mark.parametrize(
    "a, b, num_points", [(2, 0, 21)]  # training set will have 0.3 * 21 ~= 6 samples
)
@pytest.mark.parametrize(
    "scorer, total_atol", [(squashed_r2, 0.1), (squashed_variance, 0.1)]
)
@pytest.mark.parametrize(
    "fun, n_iterations, kwargs",
    [
        (permutation_montecarlo_shapley, 500, {}),
        (truncated_montecarlo_shapley, 500, {"coordinator_update_period": 1}),
        (owen_sampling_shapley, 4, {"max_q": 400, "method": "standard"}),
        # FIXME: antithetic breaks for non-deterministic u
        # (owen_sampling_shapley, 4, {"max_q": 400, "method": "antithetic"}),
    ],
)
def test_linear_montecarlo_with_outlier(
    linear_dataset,
    scorer: Union[str, Scorer],
    total_atol: float,
    fun,
    n_iterations: float,
    kwargs: dict,
    memcache_client_config,
):
    """Tests whether valuation methods are able to detect an obvious outlier.

    A point is selected at random from a linear dataset and the dependent
    variable is set to 10 standard deviations.

    Note that this implies that the whole dataset will have very low utility:
    e.g. for R^2 it will be very negative. The larger the range of the utility,
    the more samples are required for the Monte Carlo approximations to converge,
    as indicated by the Hoeffding bound.
    """
    outlier_idx = np.random.randint(len(linear_dataset.y_train))
    linear_dataset.y_train[outlier_idx] = np.std(linear_dataset.y_train) * 10
    linear_utility = Utility(
        LinearRegression(),
        data=linear_dataset,
        scoring=scorer,
        cache_options=MemcachedConfig(client_config=memcache_client_config),
    )
    values = fun(
        linear_utility,
        n_iterations=int(n_iterations),
        progress=False,
        n_jobs=1,
        **kwargs,
    )

    check_total_value(linear_utility, values, atol=total_atol)
    assert values[0].index == outlier_idx


@pytest.mark.parametrize(
    "a, b, num_points, num_groups", [(2, 0, 21, 2)]  # 24*0.3=6 samples in 2 groups
)
@pytest.mark.parametrize("scorer, rtol", [(squashed_r2, 0.1), (squashed_variance, 0.1)])
@pytest.mark.parametrize(
    "fun, n_iterations, kwargs",
    [
        (permutation_montecarlo_shapley, 700, {}),
        (truncated_montecarlo_shapley, 500, {"coordinator_update_period": 1}),
        (owen_sampling_shapley, 4, {"max_q": 300, "method": "standard"}),
        # FIXME: antithetic breaks for non-deterministic u
        # (owen_sampling_shapley, 4, {"max_q": 300, "method": "antithetic"}),
    ],
)
def test_grouped_linear_montecarlo_shapley(
    linear_dataset,
    num_groups,
    fun,
    scorer: str,
    rtol: float,
    n_iterations: float,
    kwargs: dict,
    memcache_client_config: "MemcachedClientConfig",
):
    """
    For permutation and truncated montecarlo, the rtol for each scorer is chosen
    so that the number of samples selected is just above the (ε,δ) bound for ε =
    rtol, δ=0.001 and the range corresponding to each score. This means that
    roughly once every 1000/num_methods runs the test will fail.

    """
    data_groups = np.random.randint(0, num_groups, len(linear_dataset))
    grouped_linear_dataset = GroupedDataset.from_dataset(linear_dataset, data_groups)
    grouped_linear_utility = Utility(
        LinearRegression(),
        data=grouped_linear_dataset,
        scoring=scorer,
        cache_options=MemcachedConfig(client_config=memcache_client_config),
    )
    exact_values = combinatorial_exact_shapley(grouped_linear_utility, progress=False)

    values = fun(
        grouped_linear_utility,
        n_iterations=int(n_iterations),
        progress=False,
        n_jobs=1,
        **kwargs,
    )

    check_values(values, exact_values, rtol=rtol)


@pytest.mark.skip("What is the point testing random forest training?")
@pytest.mark.parametrize(
    "num_points, num_features, regressor, scorer, n_iterations",
    [
        (10, 3, RandomForestRegressor(n_estimators=2), "r2", 20),
        (10, 3, DecisionTreeRegressor(), "r2", 20),
    ],
)
def test_random_forest(
    housing_dataset,
    regressor,
    scorer: str,
    n_iterations: float,
    memcache_client_config: "MemcachedClientConfig",
    n_jobs: int,
):
    """This test checks that random forest can be trained in our library.
    Originally, it would also check that the returned values match between
    permutation and combinatorial Monte Carlo, but this was taking too long in the
    pipeline and was removed."""
    rf_utility = Utility(
        regressor,
        data=housing_dataset,
        scoring=scorer,
        enable_cache=True,
        cache_options=MemcachedConfig(
            client_config=memcache_client_config,
            allow_repeated_evaluations=True,
            rtol_stderr=1,
            time_threshold=0,
        ),
    )

    _, _ = truncated_montecarlo_shapley(
        rf_utility, n_iterations=int(n_iterations), progress=False, n_jobs=n_jobs
    )
