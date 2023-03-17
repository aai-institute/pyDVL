import logging

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from pydvl.utils import MemcachedConfig, Status, Utility
from pydvl.utils.score import Scorer, squashed_r2
from pydvl.value import compute_shapley_values
from pydvl.value.shapley import ShapleyMode
from pydvl.value.shapley.truncated import NoTruncation
from pydvl.value.stopping import HistoryDeviation, MaxUpdates

from .. import check_total_value, check_values

log = logging.getLogger(__name__)


# noinspection PyTestParametrized
@pytest.mark.parametrize(
    "num_samples, fun, rtol, atol, kwargs",
    [
        (
            12,
            ShapleyMode.TruncatedMontecarlo,
            0.1,
            1e-5,
            dict(
                done=MaxUpdates(500),
                truncation=NoTruncation(),
                n_concurrent_computations=10,
            ),
        ),
    ],
)
def test_tmcs_analytic_montecarlo_shapley(
    num_samples,
    analytic_shapley,
    parallel_config,
    n_jobs,
    fun: ShapleyMode,
    rtol: float,
    atol: float,
    kwargs: dict,
):
    u, exact_values = analytic_shapley

    values = compute_shapley_values(
        u, mode=fun, n_jobs=n_jobs, config=parallel_config, progress=False, **kwargs
    )

    check_values(values, exact_values, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "a, b, num_points", [(2, 0, 21)]  # training set will have 0.3 * 21 = 6 samples
)
@pytest.mark.parametrize("scorer, rtol", [(squashed_r2, 0.25)])
@pytest.mark.parametrize(
    "fun, kwargs",
    [
        (
            ShapleyMode.TruncatedMontecarlo,
            dict(
                done=MaxUpdates(500),
                truncation=NoTruncation(),
                n_concurrent_computations=10,
            ),
        ),
    ],
)
def test_tmcs_linear_montecarlo_shapley(
    linear_shapley,
    n_jobs,
    memcache_client_config,
    scorer: Scorer,
    rtol: float,
    fun: ShapleyMode,
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
    u, exact_values = linear_shapley
    check_total_value(u, exact_values, rtol=rtol)

    values = compute_shapley_values(
        u, mode=fun, progress=False, n_jobs=n_jobs, **kwargs
    )

    check_values(values, exact_values, rtol=rtol)
    check_total_value(u, values, rtol=rtol)  # FIXME, could be more than rtol


@pytest.mark.parametrize(
    "a, b, num_points", [(2, 0, 21)]  # training set will have 0.3 * 21 ~= 6 samples
)
@pytest.mark.parametrize("scorer, total_atol", [(squashed_r2, 0.2)])
@pytest.mark.parametrize(
    "fun, kwargs",
    [
        (
            ShapleyMode.TruncatedMontecarlo,
            dict(
                done=HistoryDeviation(n_steps=10, rtol=0.1) | MaxUpdates(500),
                truncation=NoTruncation(),
                n_concurrent_computations=10,
            ),
        ),
    ],
)
def test_tmcs_linear_montecarlo_with_outlier(
    linear_dataset,
    n_jobs,
    memcache_client_config,
    scorer: Scorer,
    total_atol: float,
    fun,
    kwargs: dict,
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
        scorer=scorer,
        cache_options=MemcachedConfig(client_config=memcache_client_config),
    )
    values = compute_shapley_values(
        linear_utility, mode=fun, progress=False, n_jobs=n_jobs, **kwargs
    )
    values.sort()

    assert values.status == Status.Converged
    check_total_value(linear_utility, values, atol=total_atol)
    assert values[0].index == outlier_idx
