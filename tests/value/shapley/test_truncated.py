import logging

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from pydvl.utils import Status, Utility
from pydvl.utils.score import Scorer, squashed_r2
from pydvl.value import compute_shapley_values
from pydvl.value.shapley import ShapleyMode
from pydvl.value.shapley.truncated import FixedTruncation, NoTruncation
from pydvl.value.stopping import HistoryDeviation, MaxUpdates

from .. import check_total_value, check_values

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
    "done, truncation_cls, truncation_kwargs",
    [
        (MaxUpdates(600), NoTruncation, dict()),
        (MaxUpdates(600), FixedTruncation, dict(fraction=0.9)),
    ],
)
def test_games(
    test_game,
    parallel_backend,
    n_jobs,
    done,
    truncation_cls,
    truncation_kwargs,
    seed,
):
    try:
        truncation = truncation_cls(test_game.u, **truncation_kwargs)
    except TypeError:
        # The NoTruncation class's constructor doesn't take any arguments
        truncation = truncation_cls(**truncation_kwargs)

    values = compute_shapley_values(
        test_game.u,
        mode=ShapleyMode.TruncatedMontecarlo,
        done=done,
        truncation=truncation,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        seed=seed,
        progress=True,
    )

    exact_values = test_game.shapley_values()
    check_values(values, exact_values, rtol=0.2, atol=1e-4)


@pytest.mark.parametrize(
    "a, b, num_points",
    [(2, 0, 21)],  # training set will have 0.3 * 21 ~= 6 samples
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
            ),
        ),
    ],
)
def test_tmcs_linear_montecarlo_with_outlier(
    linear_dataset,
    n_jobs,
    memcache_client_config,
    scorer: Scorer,
    cache_backend,
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
        cache_backend=cache_backend,
    )
    values = compute_shapley_values(
        linear_utility, mode=fun, progress=False, n_jobs=n_jobs, **kwargs
    )
    values.sort()

    assert values.status == Status.Converged
    check_total_value(linear_utility, values, atol=total_atol)
    assert values[0].index == outlier_idx
