from __future__ import annotations

import logging

import pytest
import torch
from joblib import parallel_config
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from pydvl.utils.array import try_torch_import
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.games import (
    DummyGameDataset,
    MinerGame,
    ShoesGame,
)
from pydvl.valuation.methods.least_core import (
    ExactLeastCoreValuation,
    MonteCarloLeastCoreValuation,
    create_least_core_problem,
)
from pydvl.valuation.samplers import (
    DeterministicUniformSampler,
    FiniteNoIndexIteration,
)
from tests.valuation import check_total_value, check_values, recursive_make

logger = logging.getLogger(__name__)

torch = try_torch_import()
if torch is None:
    pytest.skip("PyTorch not available", allow_module_level=True)


class TensorDummyGameDataset(DummyGameDataset):
    """Extends DummyGameDataset to use PyTorch tensors instead of NumPy arrays."""

    def __init__(self, n_players: int, description: str = ""):
        x = torch.arange(0, n_players, 1).reshape(-1, 1).float()
        nil = torch.zeros_like(x)
        (
            Dataset.__init__(
                self,
                x,
                nil.clone(),
                feature_names=["x"],
                target_names=["y"],
                description=description,
            ),
        )


class TensorMinerGame(MinerGame):
    """Extends MinerGame to use PyTorch tensors."""

    def __init__(self, n_players: int):
        super().__init__(n_players)
        self.data = TensorDummyGameDataset(self.n_players, "Tensor Miner Game dataset")


class TensorShoesGame(ShoesGame):
    """Extends ShoesGame to use PyTorch tensors."""

    def __init__(self, left: int, right: int):
        super().__init__(left, right)
        self.data = TensorDummyGameDataset(self.n_players, "Tensor Shoes Game dataset")


@pytest.mark.torch
@pytest.mark.parametrize(
    "game_cls, game_kwargs, n_samples",
    [
        (TensorMinerGame, {"n_players": 8}, 128),
        (TensorShoesGame, {"left": 10, "right": 5}, 10000),
    ],
)
@pytest.mark.parametrize("non_negative_subsidy", (True, False))
def test_tensor_montecarlo_least_core(
    game_cls, game_kwargs, n_samples, non_negative_subsidy, seed
):
    """Test that MonteCarloLeastCoreValuation works with tensor inputs."""

    game = recursive_make(game_cls, game_kwargs, seed=seed)

    valuation = MonteCarloLeastCoreValuation(
        utility=game.u,
        n_samples=n_samples,
        non_negative_subsidy=non_negative_subsidy,
        progress=False,
    )
    result = valuation.fit(data=game.data).result
    exact_values = game.least_core_values()

    if non_negative_subsidy:
        check_values(result, exact_values)
        # Sometimes the subsidy is negative but really close to zero
        # Due to numerical errors
        if result.subsidy < 0:
            assert_almost_equal(result.subsidy, 0.0, decimal=5)
    else:
        check_values(result, exact_values, extra_values_names=["subsidy"])


@pytest.mark.torch
@pytest.mark.parametrize(
    "game_cls, game_kwargs",
    [
        (TensorMinerGame, {"n_players": 3}),
        (TensorMinerGame, {"n_players": 4}),
        (TensorShoesGame, {"left": 1, "right": 1}),
        (TensorShoesGame, {"left": 2, "right": 1}),
        (TensorShoesGame, {"left": 1, "right": 2}),
    ],
)
@pytest.mark.parametrize("non_negative_subsidy", (True, False))
def test_tensor_exact_least_core(game_cls, game_kwargs, non_negative_subsidy):
    """Test that ExactLeastCoreValuation works with tensor inputs."""
    game = recursive_make(game_cls, game_kwargs)

    valuation = ExactLeastCoreValuation(
        utility=game.u,
        non_negative_subsidy=non_negative_subsidy,
        progress=False,
    )

    result = valuation.fit(data=game.data).result

    check_total_value(game.u.with_dataset(game.data), result)
    exact_values = game.least_core_values()

    if non_negative_subsidy:
        check_values(result, exact_values)
        # Sometimes the subsidy is negative but really close to zero
        # Due to numerical errors
        if result.subsidy < 0:
            assert_almost_equal(result.subsidy, 0.0, decimal=5)
    else:
        check_values(result, exact_values, extra_values_names=["subsidy"])


@pytest.mark.torch
@pytest.mark.parametrize(
    "game_cls, game_kwargs",
    [
        (TensorMinerGame, {"n_players": 3}),
        (TensorMinerGame, {"n_players": 4}),
        (TensorShoesGame, {"left": 1, "right": 1}),
        (TensorShoesGame, {"left": 2, "right": 1}),
        (TensorShoesGame, {"left": 1, "right": 2}),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_prepare_problem_for_tensor_exact_least_core(game_cls, game_kwargs, batch_size):
    """Test that create_least_core_problem works with tensor inputs."""
    game = recursive_make(game_cls, game_kwargs)

    sampler = DeterministicUniformSampler(
        index_iteration=FiniteNoIndexIteration,
        batch_size=batch_size,
    )
    utility = game.u.with_dataset(game.data)
    powerset_size = 2 ** len(utility.training_data)

    problem = create_least_core_problem(
        u=utility,
        sampler=sampler,
        n_samples=powerset_size,
        progress=False,
    )

    expected = game.least_core_problem()
    assert_array_almost_equal(problem.utility_values, expected.utility_values)
    assert_array_almost_equal(problem.A_lb, expected.A_lb)
