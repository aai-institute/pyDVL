import math
from typing import Any, Type

import numpy as np
import pytest
from joblib import parallel_config

from pydvl.valuation import TruncatedUniformStratifiedSampler
from pydvl.valuation.games import Game
from pydvl.valuation.methods import (
    BetaShapleyValuation,
    DataBanzhafValuation,
    SemivalueValuation,
    ShapleyValuation,
)
from pydvl.valuation.samplers import IndexSampler, LOOSampler, UniformSampler
from pydvl.valuation.stopping import HistoryDeviation, MaxUpdates, MinUpdates

from .. import check_values, recursive_make
from ..samplers.test_sampler import deterministic_samplers, random_samplers
from ..utils import timed


@pytest.mark.parametrize("n", [10, 100])
@pytest.mark.parametrize(
    "valuation_cls, kwargs",
    [
        (BetaShapleyValuation, {"alpha": 1, "beta": 1}),
        (BetaShapleyValuation, {"alpha": 1, "beta": 16}),
        (BetaShapleyValuation, {"alpha": 4, "beta": 1}),
        (DataBanzhafValuation, {}),
        (ShapleyValuation, {}),
    ],
)
def test_coefficients(n, valuation_cls, kwargs):
    r"""Coefficients for semi-values must fulfill:

    $$ \sum_{i=1}^{n}\choose{n-1}{j-1}w^{(n)}(j) = 1 $$

    Note that we depart from the usual definitions by including the factor $1/n$
    in the shapley and beta coefficients.
    """
    valuation = valuation_cls(
        utility=None,
        sampler=UniformSampler(),
        is_done=MaxUpdates(50),
        progress=False,
        **kwargs,
    )

    s = [
        valuation.coefficient(n, j - 1, math.comb(n - 1, j - 1))
        for j in range(1, n + 1)
    ]
    assert np.isclose(1, np.sum(s))


@pytest.mark.flaky(reruns=1)
@pytest.mark.parametrize(
    "test_game",
    [
        ("symmetric-voting", {"n_players": 4}),
        ("shoes", {"left": 3, "right": 2}),
    ],
    indirect=["test_game"],
)
@pytest.mark.parametrize(
    "sampler_cls, sampler_kwargs", deterministic_samplers() + random_samplers()
)
@pytest.mark.parametrize(
    "valuation_cls, valuation_kwargs, exact_values_attr",
    [
        # DataShapley is already tested in test_montecarlo_shapley.py
        # (ShapleyValuation, {}, "shapley_values"),
        (BetaShapleyValuation, {"alpha": 1, "beta": 1}, "shapley_values"),
        (DataBanzhafValuation, {}, "banzhaf_values"),
    ],
)
def test_games(
    test_game: Game,
    sampler_cls: Type[IndexSampler],
    sampler_kwargs: dict,
    valuation_cls: Type[SemivalueValuation],
    valuation_kwargs: dict[str, Any],
    exact_values_attr: str,
    seed: int,
):
    if issubclass(sampler_cls, LOOSampler):
        pytest.skip("LOOSampler does not apply to Shapley and Banzhaf")

    # The games have too few players for the bounds in random_samplers(), so we reset
    # them to the limits
    if sampler_cls == TruncatedUniformStratifiedSampler:
        sampler_kwargs = {"lower_bound": None, "upper_bound": None}

    # history = HistoryDeviation(n_steps=1000 * len(test_game.data) ** 2, rtol=1e-3)
    sampler = recursive_make(sampler_cls, sampler_kwargs, seed)
    valuation = valuation_cls(
        utility=test_game.u,
        sampler=sampler,
        is_done=MinUpdates(1000 * len(test_game.data)),  # | history,
        progress=False,
        **valuation_kwargs,
    )
    valuation.fit(test_game.data)
    result = valuation.values()
    exact_result = test_game.__getattribute__(exact_values_attr)()

    # import matplotlib.pyplot as plt
    # from pydvl.valuation.games import ShoesGame, SymmetricVotingGame
    # data = history.memory[:, -history.count :]  # Grab the last `count` values
    # for vv in data:  # each row is one value series
    #     fraction = len(vv) // 2
    #     plt.plot(range(fraction), vv[-fraction:], alpha=0.7)
    # if isinstance(test_game, SymmetricVotingGame):
    #     y = exact_result.values[0]
    #     plt.axhline(y, color="black", linestyle="--")
    # elif isinstance(test_game, ShoesGame):
    #     y1 = exact_result.values[0]
    #     y2 = exact_result.values[-1]
    #     plt.axhline(y1, color="black", linestyle="--")
    #     plt.axhline(y2, color="black", linestyle="--")
    # plt.title(f"{test_game} - {valuation_cls.__name__} - {sampler_cls.__name__}")
    # plt.show()

    check_values(result, exact_result, atol=0.1)


@pytest.mark.flaky(reruns=1)
@pytest.mark.parametrize(
    "test_game",
    [("shoes", {"left": 3, "right": 2})],
    indirect=["test_game"],
)
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_batch_size(test_game, n_jobs, seed):
    def compute_semivalues(batch_size, n_jobs=n_jobs, seed=seed):
        valuation = BetaShapleyValuation(
            utility=test_game.u,
            sampler=UniformSampler(batch_size=batch_size, seed=seed),
            is_done=MaxUpdates(100),
            progress=False,
            alpha=1,
            beta=1,
        )
        with parallel_config(n_jobs=n_jobs):
            valuation.fit(test_game.data)
        return valuation.values()

    timed_fn = timed(compute_semivalues)
    result_single_batch = timed_fn(batch_size=1)
    result_multi_batch = timed_fn(batch_size=5)

    # Occasionally, batch_2 arrives before batch_1, so rtol isn't always 0.
    check_values(result_single_batch, result_multi_batch, rtol=1e-4)
