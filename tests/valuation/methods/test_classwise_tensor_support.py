r"""
Tests for tensor support in ClasswiseShapleyValuation
"""
from __future__ import annotations

import numpy as np
import pytest

from pydvl.utils.array import try_torch_import
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.methods import ClasswiseShapleyValuation
from pydvl.valuation.samplers import ClasswiseSampler, DeterministicPermutationSampler, DeterministicUniformSampler
from pydvl.valuation.scorers import ClasswiseSupervisedScorer
from pydvl.valuation.stopping import MaxUpdates
from pydvl.valuation.utility import ClasswiseModelUtility
from pydvl.valuation import FiniteNoIndexIteration

torch = try_torch_import()
pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch not installed")


class TorchLinearClassifier:
    def __init__(self):
        self._beta = None

    def fit(self, x, y):
        # Extract the first feature and convert to 1D if needed
        x_1d = x[:, 0].reshape(-1)
        
        # Calculate beta using tensor operations
        self._beta = torch.dot(x_1d, y.float()) / torch.dot(x_1d, x_1d)
        return self

    def predict(self, x):
        if self._beta is None:
            raise AttributeError("Model not fitted")
        
        # Predict using the fitted beta parameter
        x_1d = x[:, 0].reshape(-1)
        probs = self._beta * x_1d
        return torch.clamp(torch.round(probs + 1e-10), 0, 1).to(torch.int64)

    def score(self, x, y):
        pred_y = self.predict(x)
        return float((pred_y == y).sum().item() / len(y))


@pytest.fixture
def tensor_train_dataset():
    """Create a tensor-based training dataset."""
    x_train = torch.arange(1, 5).reshape(-1, 1).float()
    y_train = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    return Dataset(x_train, y_train)


@pytest.fixture
def tensor_test_dataset(tensor_train_dataset):
    """Create a tensor-based test dataset."""
    x_test, _ = tensor_train_dataset.data()
    y_test = torch.tensor([0, 0, 0, 1], dtype=torch.int64)
    return Dataset(x_test, y_test)


@pytest.fixture
def tensor_classwise_utility(tensor_test_dataset):
    """Create a tensor-based utility function for ClasswiseShapleyValuation."""
    return ClasswiseModelUtility(
        TorchLinearClassifier(),
        ClasswiseSupervisedScorer("accuracy", tensor_test_dataset),
        catch_errors=False,
    )


def test_classwise_shapley_tensor_support(tensor_classwise_utility, tensor_train_dataset):
    """Test that ClasswiseShapleyValuation works with tensor inputs."""
    in_class_sampler = DeterministicPermutationSampler()
    out_of_class_sampler = DeterministicUniformSampler(index_iteration=FiniteNoIndexIteration)
    sampler = ClasswiseSampler(
        in_class=in_class_sampler,
        out_of_class=out_of_class_sampler,
        batch_size=1,
    )
    
    # Create the valuation with tensor inputs
    valuation = ClasswiseShapleyValuation(
        tensor_classwise_utility,
        sampler=sampler,
        is_done=MaxUpdates(100),
        progress=False,
        normalize_values=True,
    )
    
    # Fit the valuation
    valuation.fit(tensor_train_dataset)
    
    # Check result
    assert valuation.result is not None
    assert isinstance(valuation.result.values, np.ndarray)  # Result values are always numpy arrays
    assert len(valuation.result.values) == 4  # Should have a value for each training point
    
    # Ensure all values are finite
    assert np.all(np.isfinite(valuation.result.values))


def test_compare_tensor_and_numpy_results():
    """Test that tensor and numpy inputs produce equivalent results."""
    # Create numpy datasets
    x_train_np = np.arange(1, 5).reshape(-1, 1).astype(np.float32)
    y_train_np = np.array([0, 0, 1, 1], dtype=np.int64)
    np_train_dataset = Dataset(x_train_np, y_train_np)
    
    x_test_np = x_train_np.copy()
    y_test_np = np.array([0, 0, 0, 1], dtype=np.int64)
    np_test_dataset = Dataset(x_test_np, y_test_np)
    
    # Create numpy-based utility
    from tests.valuation.methods.test_classwise_shapley import ClosedFormLinearClassifier
    np_utility = ClasswiseModelUtility(
        ClosedFormLinearClassifier(),
        ClasswiseSupervisedScorer("accuracy", np_test_dataset),
        catch_errors=False,
    )
    
    # Create tensor datasets
    x_train_tensor = torch.tensor(x_train_np)
    y_train_tensor = torch.tensor(y_train_np)
    tensor_train_dataset = Dataset(x_train_tensor, y_train_tensor)
    
    x_test_tensor = torch.tensor(x_test_np)
    y_test_tensor = torch.tensor(y_test_np)
    tensor_test_dataset = Dataset(x_test_tensor, y_test_tensor)
    
    # Create tensor-based utility
    tensor_utility = ClasswiseModelUtility(
        TorchLinearClassifier(),
        ClasswiseSupervisedScorer("accuracy", tensor_test_dataset),
        catch_errors=False,
    )
    
    # Create samplers
    in_class_sampler = DeterministicPermutationSampler()
    out_of_class_sampler = DeterministicUniformSampler(index_iteration=FiniteNoIndexIteration)
    sampler = ClasswiseSampler(
        in_class=in_class_sampler,
        out_of_class=out_of_class_sampler,
        batch_size=1,
    )
    
    # Create valuations
    np_valuation = ClasswiseShapleyValuation(
        np_utility,
        sampler=sampler,
        is_done=MaxUpdates(100),
        progress=False,
        normalize_values=True,
    )
    
    tensor_valuation = ClasswiseShapleyValuation(
        tensor_utility,
        sampler=sampler,
        is_done=MaxUpdates(100),
        progress=False,
        normalize_values=True,
    )
    
    # Fit valuations
    np_valuation.fit(np_train_dataset)
    tensor_valuation.fit(tensor_train_dataset)
    
    # Compare results - they should be close but not identical due to floating point differences
    np.testing.assert_allclose(
        np_valuation.result.values, 
        tensor_valuation.result.values, 
        rtol=1e-5, atol=1e-5
    )