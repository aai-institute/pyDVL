import logging

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from pydvl.utils import GroupedDataset, MemcachedConfig, Utility
from pydvl.utils.numeric import num_samples_permutation_hoeffding
from pydvl.utils.score import Scorer, squashed_r2, squashed_variance
from pydvl.value import compute_shapley_values
from pydvl.value.shapley import ShapleyMode
from pydvl.value.shapley.naive import combinatorial_exact_shapley
from pydvl.value.shapley.truncated import NoTruncation
from pydvl.value.stopping import HistoryDeviation, MaxChecks, MaxUpdates

from .. import check_rank_correlation, check_total_value, check_values

log = logging.getLogger(__name__)


# noinspection PyTestParametrized
@pytest.mark.parametrize(
    "num_samples, fun, rtol, atol, kwargs",
    [
        (12, ShapleyMode.PermutationMontecarlo, 0.1, 1e-5, {"done": MaxUpdates(10)}),
        # FIXME! it should be enough with 2**(len(data)-1) samples
        (
            8,
            ShapleyMode.CombinatorialMontecarlo,
            0.2,
            1e-4,
            {"done": MaxUpdates(2**10)},
        ),
        (
            12,
            ShapleyMode.TruncatedMontecarlo,
            0.1,
            1e-5,
            dict(
                coordinator_update_period=1,
                worker_update_period=0.5,
                done=MaxUpdates(500),
                truncation=NoTruncation(),
            ),
        ),
        (12, ShapleyMode.Owen, 0.1, 1e-4, dict(n_samples=4, max_q=200)),
        (12, ShapleyMode.OwenAntithetic, 0.1, 1e-4, dict(n_samples=4, max_q=200)),
        (
            4,
            ShapleyMode.GroupTesting,
            0.1,
            # Because of the inaccuracy of GTS, a high atol is required for the
            # value 0, for which the rtol has no effect.
            1e-2,
            dict(n_iterations=int(4e5), epsilon=0.2, delta=0.01),
        ),
    ],
)
def test_analytic_montecarlo_shapley(
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


@pytest.mark.parametrize("num_samples, delta, eps", [(8, 0.1, 0.1)])
@pytest.mark.parametrize(
    "fun", [ShapleyMode.PermutationMontecarlo, ShapleyMode.CombinatorialMontecarlo]
)
def test_hoeffding_bound_montecarlo(
    num_samples,
    analytic_shapley,
    tolerate,
    n_jobs,
    fun: ShapleyMode,
    delta: float,
    eps: float,
):
    u, exact_values = analytic_shapley

    n_samples = num_samples_permutation_hoeffding(delta=delta, eps=eps, u_range=1)

    for _ in range(10):
        with tolerate(max_failures=int(10 * delta)):
            values = compute_shapley_values(
                u=u, mode=fun, done=MaxChecks(n_samples), n_jobs=n_jobs
            )
            # Trivial bound on total error using triangle inequality
            check_total_value(u, values, atol=len(u.data) * eps)
            check_rank_correlation(values, exact_values, threshold=0.8)


@pytest.mark.parametrize(
    "a, b, num_points", [(2, 0, 21)]  # training set will have 0.3 * 21 = 6 samples
)
@pytest.mark.parametrize("scorer, rtol", [(squashed_r2, 0.25)])
@pytest.mark.parametrize(
    "fun, kwargs",
    [
        # FIXME: Hoeffding says 400 should be enough
        (ShapleyMode.PermutationMontecarlo, dict(done=MaxUpdates(600))),
        (
            ShapleyMode.TruncatedMontecarlo,
            dict(
                coordinator_update_period=0.2,
                worker_update_period=0.1,
                done=MaxUpdates(500),
                truncation=NoTruncation(),
            ),
        ),
        (ShapleyMode.CombinatorialMontecarlo, dict(done=MaxUpdates(2**11))),
        (ShapleyMode.Owen, dict(n_samples=4, max_q=300)),
        (ShapleyMode.OwenAntithetic, dict(n_samples=4, max_q=300)),
        (
            ShapleyMode.GroupTesting,
            dict(n_iterations=int(1e5), epsilon=0.2, delta=0.01),
        ),
    ],
)
def test_linear_montecarlo_shapley(
    linear_dataset,
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
    u = Utility(
        LinearRegression(),
        data=linear_dataset,
        scorer=scorer,
        cache_options=MemcachedConfig(client_config=memcache_client_config),
    )

    exact_values = combinatorial_exact_shapley(u, progress=False)
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
        (ShapleyMode.PermutationMontecarlo, {"done": MaxUpdates(500)}),
        (
            ShapleyMode.TruncatedMontecarlo,
            dict(
                coordinator_update_period=0.2,
                worker_update_period=0.1,
                done=HistoryDeviation(n_steps=10, rtol=0.1) | MaxUpdates(500),
                truncation=NoTruncation(),
            ),
        ),
        (ShapleyMode.Owen, dict(n_samples=4, max_q=400)),
        (ShapleyMode.OwenAntithetic, dict(n_samples=4, max_q=400)),
        (
            ShapleyMode.GroupTesting,
            dict(n_iterations=int(1e5), epsilon=0.2, delta=0.01),
        ),
    ],
)
def test_linear_montecarlo_with_outlier(
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
    from pydvl.utils import Status

    assert values.status == Status.Converged
    check_total_value(linear_utility, values, atol=total_atol)
    assert values[0].index == outlier_idx


@pytest.mark.parametrize(
    "a, b, num_points, num_groups", [(2, 0, 21, 2)]  # 24*0.3=6 samples in 2 groups
)
@pytest.mark.parametrize("scorer, rtol", [(squashed_r2, 0.1)])
@pytest.mark.parametrize(
    "fun, kwargs",
    [
        (ShapleyMode.PermutationMontecarlo, dict(done=MaxUpdates(700))),
    ],
)
def test_grouped_linear_montecarlo_shapley(
    linear_dataset,
    n_jobs,
    memcache_client_config: "MemcachedClientConfig",
    num_groups: int,
    fun: ShapleyMode,
    scorer: Scorer,
    rtol: float,
    kwargs: dict,
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
        scorer=scorer,
        cache_options=MemcachedConfig(client_config=memcache_client_config),
    )
    exact_values = combinatorial_exact_shapley(grouped_linear_utility, progress=False)

    values = compute_shapley_values(
        grouped_linear_utility, mode=fun, progress=False, n_jobs=n_jobs, **kwargs
    )

    check_values(values, exact_values, rtol=rtol)
