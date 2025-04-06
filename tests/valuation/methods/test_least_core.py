import logging

import pytest
from joblib import parallel_config
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from pydvl.valuation.methods.least_core import (
    ExactLeastCoreValuation,
    LeastCoreValuation,
    MonteCarloLeastCoreValuation,
    create_least_core_problem,
)
from pydvl.valuation.samplers import (
    AntitheticSampler,
    ConstantSampleSize,
    DeterministicUniformSampler,
    FiniteNoIndexIteration,
    HarmonicSampleSize,
    NoIndexIteration,
    RandomSizeIteration,
    StratifiedSampler,
    UniformSampler,
)
from tests.valuation import check_total_value, check_values, recursive_make

logger = logging.getLogger(__name__)


@pytest.mark.flaky(reruns=1)
@pytest.mark.parametrize(
    "test_game, n_samples",
    [
        (("miner", {"n_players": 8}), 128),
        (("shoes", {"left": 10, "right": 5}), 10000),
    ],
    indirect=["test_game"],
)
@pytest.mark.parametrize(
    "sampler_cls, sampler_kwargs",
    [
        (UniformSampler, {"index_iteration": NoIndexIteration}),
        (AntitheticSampler, {"index_iteration": NoIndexIteration}),
        (
            StratifiedSampler,
            {
                "sample_sizes": (
                    ConstantSampleSize,
                    {
                        "n_samples": lambda n=32: n,
                    },
                ),
                "sample_sizes_iteration": RandomSizeIteration,
                "index_iteration": NoIndexIteration,
            },
        ),
        (
            StratifiedSampler,
            {
                "sample_sizes": (
                    ConstantSampleSize,
                    {
                        "n_samples": lambda n=32: n,
                        "lower_bound": lambda l=1: l,
                        "upper_bound": lambda u=2: u,
                    },
                ),
                "sample_sizes_iteration": RandomSizeIteration,
                "index_iteration": NoIndexIteration,
            },
        ),
        (
            StratifiedSampler,
            {
                "sample_sizes": (HarmonicSampleSize, {"n_samples": lambda n: n}),
                "index_iteration": FiniteNoIndexIteration,
            },
        ),
    ],
)
@pytest.mark.parametrize("non_negative_subsidy", (True, False))
def test_randomized_least_core_methods(
    test_game, n_samples, sampler_cls, sampler_kwargs, non_negative_subsidy, seed
):
    valuation = LeastCoreValuation(
        utility=test_game.u,
        sampler=recursive_make(
            sampler_cls, sampler_kwargs, seed=seed, n_samples=n_samples
        ),
        n_samples=n_samples,
        non_negative_subsidy=non_negative_subsidy,
        progress=False,
    )
    valuation.fit(data=test_game.data)
    values = valuation.values()
    exact_values = test_game.least_core_values()
    if non_negative_subsidy:
        check_values(values, exact_values)
        # Sometimes the subsidy is negative but really close to zero
        # Due to numerical errors
        if values.subsidy < 0:
            assert_almost_equal(values.subsidy, 0.0, decimal=5)
    else:
        check_values(values, exact_values, extra_values_names=["subsidy"])


@pytest.mark.flaky(reruns=1)
@pytest.mark.parametrize(
    "test_game, max_samples",
    [
        (("miner", {"n_players": 8}), 128),
        (("shoes", {"left": 10, "right": 5}), 10000),
    ],
    indirect=["test_game"],
)
@pytest.mark.parametrize("non_negative_subsidy", (True, False))
def test_montecarlo_least_core(test_game, max_samples, non_negative_subsidy, seed):
    valuation = MonteCarloLeastCoreValuation(
        utility=test_game.u,
        n_samples=max_samples,
        non_negative_subsidy=non_negative_subsidy,
        progress=False,
    )
    valuation.fit(data=test_game.data)
    values = valuation.values()
    exact_values = test_game.least_core_values()
    if non_negative_subsidy:
        check_values(values, exact_values)
        # Sometimes the subsidy is negative but really close to zero
        # Due to numerical errors
        if values.subsidy < 0:
            assert_almost_equal(values.subsidy, 0.0, decimal=5)
    else:
        check_values(values, exact_values, extra_values_names=["subsidy"])


@pytest.mark.parametrize(
    "test_game",
    [
        ("miner", {"n_players": 3}),
        ("miner", {"n_players": 4}),
        ("shoes", {"left": 1, "right": 1}),
        ("shoes", {"left": 2, "right": 1}),
        ("shoes", {"left": 1, "right": 2}),
    ],
    indirect=True,
)
@pytest.mark.parametrize("non_negative_subsidy", (True, False))
def test_exact_least_core_via_general_least_core_valuation(
    test_game, non_negative_subsidy
):
    sampler = DeterministicUniformSampler(
        index_iteration=FiniteNoIndexIteration,
    )

    powerset_size = 2 ** len(test_game.data)

    valuation = LeastCoreValuation(
        utility=test_game.u,
        sampler=sampler,
        non_negative_subsidy=non_negative_subsidy,
        n_samples=powerset_size,
        progress=False,
    )
    valuation.fit(data=test_game.data)
    values = valuation.values()
    check_total_value(test_game.u.with_dataset(test_game.data), values)
    exact_values = test_game.least_core_values()
    if non_negative_subsidy:
        check_values(values, exact_values)
        # Sometimes the subsidy is negative but really close to zero
        # Due to numerical errors
        if values.subsidy < 0:
            assert_almost_equal(values.subsidy, 0.0, decimal=5)
    else:
        check_values(values, exact_values, extra_values_names=["subsidy"])


@pytest.mark.parametrize(
    "test_game",
    [
        ("miner", {"n_players": 3}),
        ("miner", {"n_players": 4}),
        ("shoes", {"left": 1, "right": 1}),
        ("shoes", {"left": 2, "right": 1}),
        ("shoes", {"left": 1, "right": 2}),
    ],
    indirect=True,
)
@pytest.mark.parametrize("non_negative_subsidy", (True, False))
def test_exact_least_core(test_game, non_negative_subsidy):
    valuation = ExactLeastCoreValuation(
        utility=test_game.u,
        non_negative_subsidy=non_negative_subsidy,
        progress=False,
    )
    valuation.fit(data=test_game.data)
    values = valuation.values()
    check_total_value(test_game.u.with_dataset(test_game.data), values)
    exact_values = test_game.least_core_values()
    if non_negative_subsidy:
        check_values(values, exact_values)
        # Sometimes the subsidy is negative but really close to zero
        # Due to numerical errors
        if values.subsidy < 0:
            assert_almost_equal(values.subsidy, 0.0, decimal=5)
    else:
        check_values(values, exact_values, extra_values_names=["subsidy"])


@pytest.mark.parametrize(
    "test_game",
    [
        ("miner", {"n_players": 3}),
        ("miner", {"n_players": 4}),
        ("shoes", {"left": 1, "right": 1}),
        ("shoes", {"left": 2, "right": 1}),
        ("shoes", {"left": 1, "right": 2}),
    ],
    indirect=True,
)
@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("n_cores", [1, 3])
def test_prepare_problem_for_exact_least_core(test_game, batch_size, n_cores):
    sampler = DeterministicUniformSampler(
        index_iteration=FiniteNoIndexIteration,
        batch_size=batch_size,
    )
    utility = test_game.u.with_dataset(test_game.data)
    powerset_size = 2 ** len(utility.training_data)

    with parallel_config(n_jobs=n_cores):
        problem = create_least_core_problem(
            u=utility,
            sampler=sampler,
            n_samples=powerset_size,
            progress=False,
        )

    expected = test_game.least_core_problem()
    assert_array_almost_equal(problem.utility_values, expected.utility_values)
    assert_array_almost_equal(problem.A_lb, expected.A_lb)
