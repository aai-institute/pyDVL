"""
Tests for tensor support in DataOOBValuation
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import BaggingClassifier

from pydvl.utils.array import is_numpy, try_torch_import
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.methods.data_oob import (
    DataOOBValuation,
    neg_l2_distance,
    point_wise_accuracy,
)
from tests.valuation.methods.conftest import TorchBaggingClassifier

torch = try_torch_import()
if torch is None:
    pytest.skip("PyTorch not available", allow_module_level=True)


@pytest.mark.torch
def test_point_wise_accuracy_tensor():
    """Test that point_wise_accuracy works with tensor inputs."""
    y_true = torch.tensor([0, 1, 0, 1, 1])
    y_pred = torch.tensor([0, 1, 1, 1, 0])

    # Get accuracy scores
    scores = point_wise_accuracy(y_true, y_pred)

    # Should be numpy array with [1, 1, 0, 1, 0]
    expected = np.array([1.0, 1.0, 0.0, 1.0, 0.0])
    assert is_numpy(scores)
    assert np.array_equal(scores, expected)


@pytest.mark.torch
def test_neg_l2_distance_tensor():
    """Test that neg_l2_distance works with tensor inputs."""
    y_true = torch.tensor([0.0, 1.0, 2.0, 3.0])
    y_pred = torch.tensor([0.1, 0.9, 2.2, 2.8])

    # Get distance scores
    scores = neg_l2_distance(y_true, y_pred)

    # Should be negative squared distances: -[(0.1)², (0.1)², (0.2)², (0.2)²]
    expected = np.array([-0.01, -0.01, -0.04, -0.04])
    assert isinstance(scores, np.ndarray)
    np.testing.assert_allclose(scores, expected, rtol=1e-6)


@pytest.mark.torch
def test_data_oob_valuation_tensor_support(tensor_dataset, seed):
    """Test that DataOOBValuation works with tensor inputs."""
    model = TorchBaggingClassifier(n_estimators=5, max_samples=0.5, random_state=seed)

    # Train the model on the full dataset
    x_data, y_data = tensor_dataset.data()
    model.fit(x_data, y_data)

    # Create the valuation
    valuation = DataOOBValuation(model=model)

    # Fit the valuation
    valuation.fit(tensor_dataset)

    # Check result
    assert valuation.result is not None
    assert isinstance(
        valuation.result.values, np.ndarray
    )  # Result values are always numpy arrays
    assert len(valuation.result.values) == len(
        tensor_dataset
    )  # Should have a value for each point

    # Ensure all values are finite
    assert np.all(np.isfinite(valuation.result.values))


@pytest.mark.torch
def test_data_oob_sklearn_with_tensor_inputs(tensor_dataset, seed):
    """Test that DataOOBValuation works with sklearn models and tensor inputs."""
    # Convert tensor data to numpy for sklearn
    x_np, y_np = tensor_dataset.data()
    x_np = x_np.detach().numpy()
    y_np = y_np.detach().numpy()

    # Create and train a sklearn BaggingClassifier
    from sklearn.linear_model import LogisticRegression

    model = BaggingClassifier(
        estimator=LogisticRegression(random_state=seed),
        n_estimators=5,
        max_samples=0.5,
        random_state=seed,
    ).fit(x_np, y_np)

    valuation = DataOOBValuation(model=model)

    # Fit the valuation
    valuation.fit(tensor_dataset)

    # Check result
    assert valuation.result is not None
    assert isinstance(valuation.result.values, np.ndarray)
    assert len(valuation.result.values) == len(tensor_dataset)
    assert np.all(np.isfinite(valuation.result.values))


@pytest.mark.torch
def test_compare_tensor_and_numpy_data_oob(seed):
    """Test that tensor and numpy inputs produce equivalent results for
    DataOOBValuation."""
    # Create numpy dataset
    x_np = np.linspace(0, 10, 20).reshape(-1, 1)
    y_np = np.where(x_np.squeeze() > 5, 1, 0)
    np_dataset = Dataset(x_np, y_np)

    # Create tensor dataset with the same data
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.int64)
    tensor_dataset = Dataset(x_tensor, y_tensor)

    # Create and train a sklearn BaggingClassifier for both tests
    from sklearn.linear_model import LogisticRegression

    model = BaggingClassifier(
        estimator=LogisticRegression(random_state=seed),
        n_estimators=5,
        max_samples=0.5,
        random_state=42,
    ).fit(x_np, y_np)

    # Create valuations
    np_valuation = DataOOBValuation(model=model)
    tensor_valuation = DataOOBValuation(model=model)

    # Fit valuations
    np_valuation.fit(np_dataset)
    tensor_valuation.fit(tensor_dataset)

    # Results should be identical since the model is the same and data is identical
    np.testing.assert_allclose(
        np_valuation.result.values, tensor_valuation.result.values
    )
