import logging
from copy import deepcopy

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from pydvl.parallel.config import ParallelConfig
from pydvl.utils import GroupedDataset, Status, Utility
from pydvl.utils.numeric import num_samples_permutation_hoeffding
from pydvl.utils.score import Scorer, squashed_r2
from pydvl.utils.types import Seed
from pydvl.value import compute_shapley_values
from pydvl.value.shapley import ShapleyMode
from pydvl.value.shapley.naive import combinatorial_exact_shapley
from pydvl.value.stopping import MaxChecks, MaxUpdates

from .. import check_rank_correlation, check_total_value, check_values
from ..utils import call_with_seeds

log = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "test_game",
    [
        ("symmetric-voting", {"n_players": 6}),
        ("shoes", {"left": 3, "right": 4}),
    ],
    indirect=["test_game"],
)
@pytest.mark.parametrize(
    "fun, rtol, atol, kwargs",
    [
        (ShapleyMode.PermutationMontecarlo, 0.2, 1e-4, dict(done=MaxUpdates(500))),
        (
            ShapleyMode.CombinatorialMontecarlo,
            0.2,
            1e-4,
            dict(done=MaxUpdates(2**10)),
        ),
        (ShapleyMode.Owen, 0.2, 1e-4, dict(n_samples=5, max_q=200)),
        (ShapleyMode.OwenAntithetic, 0.1, 1e-4, dict(n_samples=5, max_q=200)),
        # Because of the inaccuracy of GroupTesting, a high atol is required for the
        # value 0, for which the rtol has no effect.
        (
            ShapleyMode.GroupTesting,
            0.1,
            1e-2,
            dict(n_samples=int(4e4), epsilon=0.2, delta=0.01),
        ),
    ],
)
def test_games(
    test_game,
    parallel_backend,
    n_jobs,
    fun: ShapleyMode,
    rtol: float,
    atol: float,
    kwargs: dict,
    seed,
):
    """Tests values for all methods using a toy games.

    For permutation, the rtol for each scorer is chosen
    so that the number of samples selected is just above the (ε,δ) bound for ε =
    rtol, δ=0.001 and the range corresponding to each score. This means that
    roughly once every 1000/num_methods runs the test will fail.

    FIXME:
     - We don't have a bound for Owen.
    NOTE:
     - The variance in the combinatorial method is huge, so we need lots of
       samples

    """
    values = compute_shapley_values(
        test_game.u,
        mode=fun,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        seed=seed,
        progress=True,
        **kwargs,
    )

    exact_values = test_game.shapley_values()
    check_values(values, exact_values, rtol=rtol, atol=atol)


@pytest.mark.slow
@pytest.mark.parametrize(
    "test_game",
    [
        ("symmetric-voting", {"n_players": 12}),
    ],
    indirect=["test_game"],
)
@pytest.mark.parametrize(
    "fun, kwargs",
    [
        # TODO Add once issue #416 is closed.
        # (ShapleyMode.PermutationMontecarlo, dict(done=MaxChecks(1))),
        (ShapleyMode.CombinatorialMontecarlo, dict(done=MaxChecks(4))),
        (ShapleyMode.Owen, dict(n_samples=4, max_q=200)),
        (ShapleyMode.OwenAntithetic, dict(n_samples=4, max_q=200)),
        (ShapleyMode.GroupTesting, dict(n_samples=21, epsilon=0.2, delta=0.01)),
    ],
)
def test_seed(
    test_game,
    parallel_backend: ParallelConfig,
    n_jobs: int,
    fun: ShapleyMode,
    kwargs: dict,
    seed: Seed,
    seed_alt: Seed,
):
    values_1, values_2, values_3 = call_with_seeds(
        compute_shapley_values,
        test_game.u,
        mode=fun,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        seeds=(seed, seed, seed_alt),
        **deepcopy(kwargs),
    )
    np.testing.assert_equal(values_1.values, values_2.values)
    with pytest.raises(AssertionError):
        np.testing.assert_equal(values_1.values, values_3.values)


@pytest.mark.skip(
    "This test is brittle and the bound isn't sharp. "
    "We should at least document the bound in the documentation."
)
@pytest.mark.slow
@pytest.mark.parametrize("num_samples, delta, eps", [(6, 0.1, 0.1)])
@pytest.mark.parametrize(
    "fun",
    [
        ShapleyMode.PermutationMontecarlo,
        ShapleyMode.CombinatorialMontecarlo,
    ],
)
@pytest.mark.flaky(reruns=1)
def test_hoeffding_bound_montecarlo(
    num_samples,
    analytic_shapley,
    n_jobs,
    fun: ShapleyMode,
    delta: float,
    eps: float,
):
    u, exact_values = analytic_shapley

    n_samples = num_samples_permutation_hoeffding(delta=delta, eps=eps, u_range=1)

    for _ in range(10):
        values = compute_shapley_values(
            u=u, mode=fun, done=MaxChecks(n_samples), n_jobs=n_jobs
        )
        # Trivial bound on total error using triangle inequality
        check_total_value(u, values, atol=len(u.data) * eps)
        check_rank_correlation(values, exact_values, threshold=0.8)


@pytest.mark.slow
@pytest.mark.parametrize(
    "a, b, num_points",
    [(2, 0, 21)],  # training set will have 0.3 * 21 ~= 6 samples
)
@pytest.mark.parametrize("scorer, total_atol", [(squashed_r2, 0.2)])
@pytest.mark.parametrize(
    "fun, kwargs",
    [
        (ShapleyMode.PermutationMontecarlo, {"done": MaxUpdates(500)}),
        (ShapleyMode.Owen, dict(n_samples=4, max_q=400)),
        (ShapleyMode.OwenAntithetic, dict(n_samples=4, max_q=400)),
        (
            ShapleyMode.GroupTesting,
            dict(n_samples=int(5e4), epsilon=0.25, delta=0.1),
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
    cache_backend,
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
        cache_backend=cache_backend,
    )
    values = compute_shapley_values(
        linear_utility, mode=fun, progress=False, n_jobs=n_jobs, **kwargs
    )
    values.sort()

    assert values.status == Status.Converged
    check_total_value(linear_utility, values, atol=total_atol)
    assert values[0].index == outlier_idx


@pytest.mark.parametrize(
    "a, b, num_points, num_groups",
    [(2, 0, 21, 2)],  # 24*0.3=6 samples in 2 groups
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
    num_groups: int,
    fun: ShapleyMode,
    scorer: Scorer,
    rtol: float,
    kwargs: dict,
    cache_backend,
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
        cache_backend=cache_backend,
    )
    exact_values = combinatorial_exact_shapley(grouped_linear_utility, progress=False)

    values = compute_shapley_values(
        grouped_linear_utility, mode=fun, progress=False, n_jobs=n_jobs, **kwargs
    )

    check_values(values, exact_values, rtol=rtol)
