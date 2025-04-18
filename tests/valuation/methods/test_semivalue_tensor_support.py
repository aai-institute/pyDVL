"""
Tests for tensor support in SemivalueValuation subclasses.
"""

from __future__ import annotations

import pytest
import numpy as np
from numpy.testing import assert_allclose

from pydvl.utils.array import try_torch_import
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.methods.semivalue import SemivalueValuation
from pydvl.valuation.methods.shapley import ShapleyValuation, TMCShapleyValuation, StratifiedShapleyValuation
from pydvl.valuation.methods.banzhaf import BanzhafValuation, MSRBanzhafValuation
from pydvl.valuation.methods.beta_shapley import BetaShapleyValuation
from pydvl.valuation.methods.delta_shapley import DeltaShapleyValuation
from pydvl.valuation.methods.loo import LOOValuation
from pydvl.valuation.samplers import DeterministicUniformSampler, PermutationSampler, MSRSampler, RandomIndexIteration
from pydvl.valuation.samplers import StratifiedSampler, ConstantSampleSize, RandomSizeIteration
from pydvl.valuation.stopping import MaxUpdates, HistoryDeviation, MinUpdates
from tests.valuation.methods.conftest import TensorMinerGame, TensorShoesGame
from tests.valuation import recursive_make, check_values

torch = try_torch_import()
if torch is None:
    pytest.skip("PyTorch not available", allow_module_level=True)


@pytest.mark.torch
@pytest.mark.parametrize(
    "game_cls, game_kwargs", 
    [
        (TensorMinerGame, {"n_players": 4}),
        (TensorShoesGame, {"left": 2, "right": 1})
    ]
)
def test_shapley_valuation_tensor_support(game_cls, game_kwargs, seed):
    """Test that ShapleyValuation works with tensor inputs."""
    # Create the game with tensor data
    game = recursive_make(game_cls, game_kwargs, seed=seed)
    
    # Create the sampler and valuation
    sampler = DeterministicUniformSampler(batch_size=1)
    valuation = ShapleyValuation(
        utility=game.u,
        sampler=sampler,
        is_done=MaxUpdates(16),  # Small number of updates for testing
        progress=False
    )
    
    # Fit the valuation
    result = valuation.fit(game.data).result
    
    # Check that the result exists and has the right shape
    assert result is not None
    assert isinstance(result.values, np.ndarray)
    assert len(result.values) == len(game.data)
    assert np.all(np.isfinite(result.values))


@pytest.mark.torch
@pytest.mark.parametrize(
    "game_cls, game_kwargs", 
    [
        (TensorMinerGame, {"n_players": 4}),
        (TensorShoesGame, {"left": 2, "right": 1})
    ]
)
def test_tmc_shapley_valuation_tensor_support(game_cls, game_kwargs, seed):
    """Test that TMCShapleyValuation works with tensor inputs."""
    # Create the game with tensor data
    game = recursive_make(game_cls, game_kwargs, seed=seed)
    
    # Create the valuation with default settings
    valuation = TMCShapleyValuation(
        utility=game.u,
        is_done=MaxUpdates(16),  # Small number of updates for testing
        progress=False
    )
    
    # Fit the valuation
    result = valuation.fit(game.data).result
    
    # Check that the result exists and has the right shape
    assert result is not None
    assert isinstance(result.values, np.ndarray)
    assert len(result.values) == len(game.data)
    assert np.all(np.isfinite(result.values))


@pytest.mark.torch
@pytest.mark.parametrize(
    "game_cls, game_kwargs", 
    [
        (TensorMinerGame, {"n_players": 4}),
        (TensorShoesGame, {"left": 2, "right": 1})
    ]
)
def test_stratified_shapley_valuation_tensor_support(game_cls, game_kwargs, seed):
    """Test that StratifiedShapleyValuation works with tensor inputs."""
    # Create the game with tensor data
    game = recursive_make(game_cls, game_kwargs, seed=seed)
    
    # Create the valuation
    valuation = StratifiedShapleyValuation(
        utility=game.u,
        is_done=MaxUpdates(16),  # Small number of updates for testing
        batch_size=1,
        progress=False
    )
    
    # Fit the valuation
    result = valuation.fit(game.data).result
    
    # Check that the result exists and has the right shape
    assert result is not None
    assert isinstance(result.values, np.ndarray)
    assert len(result.values) == len(game.data)
    assert np.all(np.isfinite(result.values))


@pytest.mark.torch
@pytest.mark.parametrize(
    "game_cls, game_kwargs", 
    [
        (TensorMinerGame, {"n_players": 4}),
        (TensorShoesGame, {"left": 2, "right": 1})
    ]
)
def test_banzhaf_valuation_tensor_support(game_cls, game_kwargs, seed):
    """Test that BanzhafValuation works with tensor inputs."""
    # Create the game with tensor data
    game = recursive_make(game_cls, game_kwargs, seed=seed)
    
    # Create the sampler and valuation
    sampler = DeterministicUniformSampler(batch_size=1)
    valuation = BanzhafValuation(
        utility=game.u,
        sampler=sampler,
        is_done=MaxUpdates(16),  # Small number of updates for testing
        progress=False
    )
    
    # Fit the valuation
    result = valuation.fit(game.data).result
    
    # Check that the result exists and has the right shape
    assert result is not None
    assert isinstance(result.values, np.ndarray)
    assert len(result.values) == len(game.data)
    assert np.all(np.isfinite(result.values))


@pytest.mark.torch
@pytest.mark.parametrize(
    "game_cls, game_kwargs", 
    [
        (TensorMinerGame, {"n_players": 4}),
        (TensorShoesGame, {"left": 2, "right": 1})
    ]
)
def test_msr_banzhaf_valuation_tensor_support(game_cls, game_kwargs, seed):
    """Test that MSRBanzhafValuation works with tensor inputs."""
    # Create the game with tensor data
    game = recursive_make(game_cls, game_kwargs, seed=seed)
    
    # Create the valuation
    valuation = MSRBanzhafValuation(
        utility=game.u,
        is_done=MaxUpdates(16),  # Small number of updates for testing
        batch_size=1,
        progress=False
    )
    
    # Fit the valuation
    result = valuation.fit(game.data).result
    
    # Check that the result exists and has the right shape
    assert result is not None
    assert isinstance(result.values, np.ndarray)
    assert len(result.values) == len(game.data)
    assert np.all(np.isfinite(result.values))


@pytest.mark.torch
@pytest.mark.parametrize(
    "game_cls, game_kwargs, alpha, beta", 
    [
        (TensorMinerGame, {"n_players": 4}, 1.0, 1.0),
        (TensorShoesGame, {"left": 2, "right": 1}, 0.5, 0.5)
    ]
)
def test_beta_shapley_valuation_tensor_support(game_cls, game_kwargs, alpha, beta, seed):
    """Test that BetaShapleyValuation works with tensor inputs."""
    # Create the game with tensor data
    game = recursive_make(game_cls, game_kwargs, seed=seed)
    
    # Create the sampler and valuation
    sampler = DeterministicUniformSampler(batch_size=1)
    valuation = BetaShapleyValuation(
        utility=game.u,
        sampler=sampler,
        is_done=MaxUpdates(16),  # Small number of updates for testing
        alpha=alpha,
        beta=beta,
        progress=False
    )
    
    # Fit the valuation
    result = valuation.fit(game.data).result
    
    # Check that the result exists and has the right shape
    assert result is not None
    assert isinstance(result.values, np.ndarray)
    assert len(result.values) == len(game.data)
    assert np.all(np.isfinite(result.values))


@pytest.mark.torch
@pytest.mark.parametrize(
    "game_cls, game_kwargs", 
    [
        (TensorMinerGame, {"n_players": 4}),
        (TensorShoesGame, {"left": 2, "right": 1})
    ]
)
def test_delta_shapley_valuation_tensor_support(game_cls, game_kwargs, seed):
    """Test that DeltaShapleyValuation works with tensor inputs."""
    # Create the game with tensor data
    game = recursive_make(game_cls, game_kwargs, seed=seed)
    
    # Create the sampler and valuation
    sampler = StratifiedSampler(
        sample_sizes=ConstantSampleSize(),
        sample_sizes_iteration=RandomSizeIteration,
        index_iteration=RandomIndexIteration,
        batch_size=1
    )
    
    valuation = DeltaShapleyValuation(
        utility=game.u,
        sampler=sampler,
        is_done=MaxUpdates(16),  # Small number of updates for testing
        progress=False
    )
    
    # Fit the valuation
    result = valuation.fit(game.data).result
    
    # Check that the result exists and has the right shape
    assert result is not None
    assert isinstance(result.values, np.ndarray)
    assert len(result.values) == len(game.data)
    assert np.all(np.isfinite(result.values))


@pytest.mark.torch
@pytest.mark.parametrize(
    "game_cls, game_kwargs", 
    [
        (TensorMinerGame, {"n_players": 4}),
        (TensorShoesGame, {"left": 2, "right": 1})
    ]
)
def test_loo_valuation_tensor_support(game_cls, game_kwargs):
    """Test that LOOValuation works with tensor inputs."""
    # Create the game with tensor data
    game = recursive_make(game_cls, game_kwargs)
    
    # Create the valuation
    valuation = LOOValuation(
        utility=game.u,
        progress=False
    )
    
    # Fit the valuation
    result = valuation.fit(game.data).result
    
    # Check that the result exists and has the right shape
    assert result is not None
    assert isinstance(result.values, np.ndarray)
    assert len(result.values) == len(game.data)
    assert np.all(np.isfinite(result.values))


@pytest.mark.torch
def test_compare_numpy_tensor_results_deterministic():
    """Test that tensor and numpy implementations produce equivalent results with deterministic settings."""
    # Create a small game with exactly the same data for both tensor and numpy
    n_players = 3
    
    # Create tensor version of the game
    x_tensor = torch.arange(0, n_players, 1).reshape(-1, 1).float()
    nil_tensor = torch.zeros_like(x_tensor)
    tensor_data = Dataset(x_tensor, nil_tensor.clone(), feature_names=["x"], target_names=["y"])
    
    # Create numpy version of the game with identical data
    x_numpy = x_tensor.numpy()
    nil_numpy = nil_tensor.numpy()
    numpy_data = Dataset(x_numpy, nil_numpy.copy(), feature_names=["x"], target_names=["y"])
    
    # Create identical utility objects for both
    from pydvl.valuation.games import MinerGame
    
    tensor_game = MinerGame(n_players)
    tensor_game.data = tensor_data
    
    numpy_game = MinerGame(n_players)
    numpy_game.data = numpy_data
    
    # Use deterministic sampler
    sampler = DeterministicUniformSampler(batch_size=1)
    stopping = MaxUpdates(2**n_players)  # Exhaust all possible combinations
    
    # Create valuations
    tensor_valuation = ShapleyValuation(
        utility=tensor_game.u,
        sampler=sampler,
        is_done=stopping,
        progress=False
    )
    
    numpy_valuation = ShapleyValuation(
        utility=numpy_game.u,
        sampler=sampler,
        is_done=stopping,
        progress=False
    )
    
    # Fit both valuations
    tensor_result = tensor_valuation.fit(tensor_data).result
    numpy_result = numpy_valuation.fit(numpy_data).result
    
    # Results should be identical since we use deterministic sampler and constant game
    assert_allclose(tensor_result.values, numpy_result.values, rtol=1e-6, atol=1e-6)