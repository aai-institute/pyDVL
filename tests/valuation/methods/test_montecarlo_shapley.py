import logging
from typing import Any, Type

import numpy as np
import pytest
from joblib import parallel_config
from sklearn.linear_model import LinearRegression

from pydvl.utils import SupervisedModel
from pydvl.utils.numeric import num_samples_permutation_hoeffding
from pydvl.utils.status import Status
from pydvl.valuation.dataset import GroupedDataset
from pydvl.valuation.methods import (
    GroupTestingShapleyValuation,
    ShapleyValuation,
    StratifiedShapleyValuation,
)
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.samplers import (
    AntitheticOwenSampler,
    AntitheticSampler,
    ConstantSampleSize,
    DeterministicUniformSampler,
    HarmonicSampleSize,
    MSRSampler,
    OwenSampler,
    PermutationSampler,
    PowerLawSampleSize,
    RandomIndexIteration,
    RandomSizeIteration,
    SequentialIndexIteration,
    StratifiedSampler,
    UniformOwenStrategy,
    UniformSampler,
)
from pydvl.valuation.samplers.utils import StochasticSamplerMixin
from pydvl.valuation.scorers import SupervisedScorer, compose_score, sigmoid
from pydvl.valuation.stopping import MaxChecks, MaxUpdates, MinUpdates, NoStopping
from pydvl.valuation.types import Sample
from pydvl.valuation.utility import ModelUtility

from .. import check_rank_correlation, check_total_value, check_values, recursive_make

log = logging.getLogger(__name__)


def shapley_methods(fudge_factor: int):
    return [
        (
            UniformSampler,
            {"seed": lambda seed: seed},
            ShapleyValuation,
            {"is_done": (MinUpdates, {"n_updates": fudge_factor * 2})},
        ),
        (
            PermutationSampler,
            {"seed": lambda seed: seed},
            ShapleyValuation,
            {"is_done": (MinUpdates, {"n_updates": fudge_factor})},
        ),
        (
            AntitheticSampler,
            {"seed": lambda seed: seed},
            ShapleyValuation,
            {"is_done": (MinUpdates, {"n_updates": fudge_factor})},
        ),
        (
            MSRSampler,
            {"seed": lambda seed: seed},
            ShapleyValuation,
            {"is_done": (MinUpdates, {"n_updates": fudge_factor * 12})},
        ),
        (
            StratifiedSampler,
            {
                "sample_sizes": (ConstantSampleSize, {"n_samples": lambda n=32: n}),
                "sample_sizes_iteration": RandomSizeIteration,
                "index_iteration": RandomIndexIteration,
                "seed": lambda seed: seed,
            },
            ShapleyValuation,
            {"is_done": (MinUpdates, {"n_updates": fudge_factor})},
        ),
        (
            StratifiedSampler,
            {
                "sample_sizes": (
                    ConstantSampleSize,
                    {
                        "n_samples": lambda n=32: n,
                        "lower_bound": lambda l=1: l,
                        "upper_bound": lambda u=None: u,
                    },
                ),
                "sample_sizes_iteration": RandomSizeIteration,
                "index_iteration": RandomIndexIteration,
                "seed": lambda seed: seed,
            },
            ShapleyValuation,
            {"is_done": (MinUpdates, {"n_updates": fudge_factor})},
        ),
        (
            StratifiedSampler,
            {
                "sample_sizes": (
                    HarmonicSampleSize,
                    {"n_samples": fudge_factor},
                ),
                "index_iteration": SequentialIndexIteration,
                "seed": lambda seed: seed,
            },
            ShapleyValuation,
            {"is_done": (MinUpdates, {"n_updates": fudge_factor})},
        ),
        (
            StratifiedSampler,
            {
                "sample_sizes": (
                    PowerLawSampleSize,
                    {
                        "exponent": lambda e=-0.5: e,
                        "n_samples": fudge_factor,
                    },
                ),
                "index_iteration": SequentialIndexIteration,
                "seed": lambda seed: seed,
            },
            ShapleyValuation,
            {"is_done": (MinUpdates, {"n_updates": fudge_factor})},
        ),
        (
            OwenSampler,
            {
                "outer_sampling_strategy": (
                    UniformOwenStrategy,
                    {"n_samples_outer": 200, "seed": lambda seed: seed},
                ),
                "seed": lambda seed: seed,
            },
            ShapleyValuation,
            {"is_done": (MinUpdates, {"n_updates": fudge_factor})},
        ),
        (
            AntitheticOwenSampler,
            {
                "outer_sampling_strategy": (
                    UniformOwenStrategy,
                    {"n_samples_outer": 100, "seed": lambda seed: seed},
                ),
                "seed": lambda seed: seed,
            },
            ShapleyValuation,
            {"is_done": (MinUpdates, {"n_updates": fudge_factor // 2})},
        ),
        (
            None,
            {},
            GroupTestingShapleyValuation,
            {
                "n_samples": fudge_factor * 100,
                "epsilon": 0.2,
                "seed": lambda seed: seed,
            },
        ),
        (
            None,
            {},
            StratifiedShapleyValuation,
            {
                "seed": lambda seed: seed,
                "is_done": (MinUpdates, {"n_updates": fudge_factor}),
            },
        ),
    ]


@pytest.mark.flaky(reruns=1)
@pytest.mark.parametrize(
    "test_game, rtol, atol",
    [
        (("symmetric-voting", {"n_players": 4}), 0.2, 1e-2),
        # (("asymmetric-voting", {}), 0.2, 1e-2), # Too many players for some methods
        (("shoes", {"left": 3, "right": 4}), 0.2, 1e-4),
    ],
    indirect=["test_game"],
)
@pytest.mark.parametrize(
    "sampler_cls, sampler_kwargs, valuation_cls, valuation_kwargs", shapley_methods(500)
)
def test_games(
    test_game,
    sampler_cls: Type,
    sampler_kwargs: dict[str, Any],
    valuation_cls,
    valuation_kwargs: dict[str, Any],
    rtol: float,
    atol: float,
    seed: int,
):
    """Tests Shapley values for all methods using toy games.

    For permutation, the rtol for each scorer is chosen
    so that the number of samples selected is just above the (ε,δ) bound for ε =
    rtol, δ=0.001 and the range corresponding to each score. This means that
    roughly once every 1000/num_methods runs the test will fail.

    NOTE:
     - The variance in the combinatorial method is huge, so we need lots of
       samples

    """
    if sampler_cls is not None:
        valuation_kwargs["sampler"] = recursive_make(
            sampler_cls, sampler_kwargs, seed=seed
        )

    valuation_kwargs["utility"] = test_game.u
    valuation_kwargs["progress"] = False

    valuation = recursive_make(valuation_cls, valuation_kwargs, seed=seed)

    valuation.fit(test_game.data)

    result = valuation.result
    expected = test_game.shapley_values()

    check_total_value(
        test_game.u.with_dataset(test_game.data), result, rtol=rtol, atol=atol
    )
    check_values(result, expected, rtol=rtol, atol=atol)


# @pytest.mark.slow
@pytest.mark.parametrize(
    "test_game",
    [("shoes", {"left": 3, "right": 4})],
    indirect=["test_game"],
)
@pytest.mark.parametrize(
    "sampler_cls, sampler_kwargs, valuation_cls, valuation_kwargs", shapley_methods(10)
)
def test_seed(
    test_game,
    sampler_cls: Type,
    sampler_kwargs: dict[str, Any],
    valuation_cls: Type,
    valuation_kwargs: dict[str, Any],
    seed: int,
    seed_alt: int,
):
    values: list[ValuationResult] = []
    if sampler_cls is None or not issubclass(sampler_cls, StochasticSamplerMixin):
        pytest.skip("This test is only for stochastic samplers")

    for s in [seed, seed, seed_alt]:
        if valuation_cls is ShapleyValuation:
            sampler = recursive_make(sampler_cls, sampler_kwargs, seed=s)
            valuation_kwargs["sampler"] = sampler

        valuation_kwargs["utility"] = test_game.u
        valuation_kwargs["progress"] = False
        valuation = recursive_make(valuation_cls, valuation_kwargs, seed=s)

        valuation.fit(test_game.data)
        values.append(valuation.result)

    values_1, values_2, values_3 = values

    np.testing.assert_equal(values_1.values, values_2.values)
    with pytest.raises(AssertionError):
        np.testing.assert_equal(values_1.values, values_3.values)


@pytest.mark.flaky(reruns=1)
@pytest.mark.parametrize("num_samples, delta, eps", [(6, 0.1, 0.1)])
@pytest.mark.parametrize(
    "sampler_cls",
    [PermutationSampler, UniformSampler],
)
def test_hoeffding_bound_montecarlo(
    analytic_shapley, dummy_train_data, sampler_cls: Type, delta: float, eps: float
):
    u, exact_values = analytic_shapley

    n_samples = num_samples_permutation_hoeffding(delta=delta, eps=eps, u_range=1)

    for _ in range(10):
        sampler = sampler_cls()
        valuation = ShapleyValuation(
            utility=u, sampler=sampler, progress=False, is_done=MaxChecks(n_samples)
        )

        result = valuation.fit(dummy_train_data).result

        check_total_value(
            u.with_dataset(dummy_train_data), result, atol=len(dummy_train_data) * eps
        )
        check_rank_correlation(result, exact_values, threshold=0.8)


@pytest.mark.slow
# training set will have 0.3 * 21 ~= 6 samples
@pytest.mark.flaky(reruns=1)
@pytest.mark.parametrize("a, b, num_points", [(2, 0, 21)])
@pytest.mark.parametrize(
    "sampler_cls, sampler_kwargs, valuation_cls, valuation_kwargs", shapley_methods(500)
)
def test_linear_montecarlo_with_outlier(
    linear_dataset,
    linear_shapley,
    n_jobs: int,
    sampler_cls: Type,
    sampler_kwargs: dict[str, Any],
    valuation_cls: Type,
    valuation_kwargs: dict[str, Any],
    seed: int,
):
    """Tests whether valuation methods are able to detect an obvious outlier.

    A point is selected at random from a linear dataset and the dependent
    variable is set to 10 standard deviations.

    Note that this implies that the whole dataset will have very low utility:
    e.g. for R^2 it will be very negative. The larger the range of the utility,
    the more samples are required for the Monte Carlo approximations to converge,
    as indicated by the Hoeffding bound.
    """
    train, test = linear_dataset
    utility, exact_result = linear_shapley

    # TODO: we'd have to precompute this in a fixture
    # outlier_idx = np.random.randint(len(train))
    # train.data().y[outlier_idx] -= 100

    if sampler_cls is not None:
        valuation_kwargs["sampler"] = recursive_make(
            sampler_cls, sampler_kwargs, seed=seed, lower_bound=0, upper_bound=None
        )

    valuation_kwargs["utility"] = utility
    valuation_kwargs["progress"] = False
    valuation = recursive_make(valuation_cls, valuation_kwargs, seed=seed)

    with parallel_config(n_jobs=n_jobs):
        valuation.fit(train)
    result = valuation.result
    result.sort()

    if sampler_cls is not OwenSampler:
        assert result.status == Status.Converged

    # Did we detect the outlier?
    # assert result[0].index == outlier_idx

    check_values(result, exact_result, rtol=0.2)
    check_total_value(utility.with_dataset(train), result, atol=0.1)


@pytest.mark.parametrize(
    "a, b, num_points, num_groups",
    [(2, 0, 21, 2)],  # 24*0.3=6 samples in 2 groups
)
@pytest.mark.parametrize(
    "sampler_cls, valuation_kwargs",
    [(PermutationSampler, dict(is_done=MaxUpdates(700)))],
)
def test_grouped_linear_montecarlo_shapley(
    linear_dataset,
    n_jobs: int,
    num_groups: int,
    sampler_cls: Type,
    valuation_kwargs: dict[str, Any],
):
    """
    For permutation and truncated montecarlo, the rtol for each scorer is chosen
    so that the number of samples selected is just above the (ε,δ) bound for ε =
    rtol, δ=0.001 and the range corresponding to each score. This means that
    roughly once every 1000/num_methods runs the test will fail.
    """
    data_train, data_test = linear_dataset

    scorer = compose_score(
        SupervisedScorer("r2", data_test, default=-np.inf),
        sigmoid,
        name="squashed r2",
    )

    rtol = 0.1

    data_groups = np.random.randint(0, num_groups, len(data_train))
    grouped_linear_dataset = GroupedDataset.from_dataset(data_train, data_groups)
    utility = ModelUtility[Sample, SupervisedModel](LinearRegression(), scorer=scorer)

    valuation = ShapleyValuation(
        utility=utility,
        sampler=sampler_cls(),
        progress=False,
        **valuation_kwargs,
    )
    with parallel_config(n_jobs=n_jobs):
        valuation.fit(grouped_linear_dataset)

    exact_valuation = ShapleyValuation(
        utility=utility,
        sampler=DeterministicUniformSampler(),
        progress=False,
        is_done=NoStopping(),
    )
    exact_valuation.fit(grouped_linear_dataset)

    check_values(valuation.result, exact_valuation.result, rtol=rtol)
