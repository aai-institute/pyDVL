import logging

import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from pydvl.valuation.methods._naive_least_core import lc_prepare_problem
from pydvl.valuation.methods.least_core import (
    LeastCoreMode,
    LeastCoreValuation,
    create_least_core_problem,
)
from pydvl.valuation.samplers import DeterministicUniformSampler
from pydvl.valuation.samplers.powerset import NoIndexIteration
from tests.valuation import check_total_value, check_values

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "test_game, n_iterations",
    [
        (("miner", {"n_players": 8}), 128),
        (("shoes", {"left": 10, "right": 5}), 10000),
    ],
    indirect=["test_game"],
)
@pytest.mark.parametrize("n_jobs", [1, -1])
@pytest.mark.parametrize("non_negative_subsidy", (True, False))
def test_montecarlo_least_core(
    test_game, n_iterations, n_jobs, non_negative_subsidy, seed
):
    valuation = LeastCoreValuation(
        utility=test_game.u,
        n_jobs=n_jobs,
        n_iterations=n_iterations,
        non_negative_subsidy=non_negative_subsidy,
        progress=False,
        seed=seed,
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
def test_naive_least_core(test_game, non_negative_subsidy):
    valuation = LeastCoreValuation(
        utility=test_game.u,
        non_negative_subsidy=non_negative_subsidy,
        progress=False,
        mode=LeastCoreMode.Exact,
    )
    valuation.fit(data=test_game.data)
    values = valuation.values()
    # HACK because check_total_value expects u with data
    test_game.u = test_game.u.with_dataset(test_game.data)
    check_total_value(test_game.u, values)
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
def test_prepare_problem_for_exact_least_core(test_game):
    problem = lc_prepare_problem(test_game.u.with_dataset(test_game.data))
    expected = test_game.least_core_problem()
    assert_array_almost_equal(problem.utility_values, expected.utility_values)
    assert_array_almost_equal(problem.A_lb, expected.A_lb)


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
def test_prepare_problem_for_exact_least_core_using_samplers(test_game):
    sampler = DeterministicUniformSampler(
        index_iteration=NoIndexIteration,
    )
    utility = test_game.u.with_dataset(test_game.data)
    powerset_size = 2 ** len(utility.training_data)

    problem = create_least_core_problem(
        u=utility,
        sampler=sampler,
        n_iterations=powerset_size,
    )

    expected = test_game.least_core_problem()
    assert_array_almost_equal(problem.utility_values, expected.utility_values)
    assert_array_almost_equal(problem.A_lb, expected.A_lb)
