"""
Tests for tensor support in DataOOBValuation
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import BaggingClassifier
from typing_extensions import Self

from pydvl.utils.array import array_equal, try_torch_import
from pydvl.utils.types import Seed
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.methods.data_oob import (
    DataOOBValuation,
    neg_l2_distance,
    point_wise_accuracy,
)

torch = try_torch_import()
if torch is None:
    pytest.skip("PyTorch not available", allow_module_level=True)


class TorchLinearClassifier:
    """Simple torch linear classifier for testing purposes."""

    def __init__(self):
        self._beta = None

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Self:
        # Extract the first feature and convert to 1D if needed
        x_1d = x[:, 0].reshape(-1)
        self._beta = torch.dot(x_1d, y.float()) / torch.dot(x_1d, x_1d)
        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if self._beta is None:
            raise AttributeError("Model not fitted")

        # Predict using the fitted beta parameter
        x_1d = x[:, 0].reshape(-1)
        probs = self._beta * x_1d
        return torch.clamp(torch.round(probs + 1e-10), 0, 1).to(torch.int64)

    def score(self, x: torch.Tensor, y: torch.Tensor) -> float:
        pred_y = self.predict(x)
        return float((pred_y == y).sum().item() / len(y))


class TorchBaggingClassifier:
    """A simple implementation of BaggingClassifier for torch tensors."""

    def __init__(self, n_estimators: int, max_samples: float, random_state: Seed):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.estimators_ = []
        self.estimators_samples_ = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        sample_size = (
            int(n_samples * self.max_samples)
            if isinstance(self.max_samples, float)
            else self.max_samples
        )

        torch.manual_seed(self.random_state)

        for i in range(self.n_estimators):
            # Sample with replacement
            indices = torch.randint(0, n_samples, (sample_size,))
            X_sampled = X[indices]
            y_sampled = y[indices]

            # Create and fit a base estimator
            estimator = TorchLinearClassifier()
            estimator.fit(X_sampled, y_sampled)

            # Store the estimator and the sample indices
            self.estimators_.append(estimator)
            self.estimators_samples_.append(indices.cpu().numpy())

        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        # Make predictions with each estimator
        all_predictions = []
        for estimator in self.estimators_:
            preds = estimator.predict(X)
            all_predictions.append(preds)

        # Stack predictions and take majority vote
        stacked = torch.stack(all_predictions)
        return torch.mode(stacked, dim=0).values


@pytest.fixture
def tensor_dataset():
    """Create a tensor-based dataset."""
    # Create a simple dataset with 20 points
    x = torch.linspace(0, 10, 20).reshape(-1, 1)
    # Simple linear relationship with some noise
    y = torch.where(
        x.squeeze() > 5,
        torch.ones(x.shape[0], dtype=torch.int64),
        torch.zeros(x.shape[0], dtype=torch.int64),
    )
    return Dataset(x, y)


@pytest.mark.torch
def test_point_wise_accuracy_tensor():
    """Test that point_wise_accuracy works with tensor inputs."""
    y_true = torch.tensor([0, 1, 0, 1, 1])
    y_pred = torch.tensor([0, 1, 1, 1, 0])

    # Get accuracy scores
    scores = point_wise_accuracy(y_true, y_pred)

    # Should be numpy array with [1, 1, 0, 1, 0]
    expected = np.array([1.0, 1.0, 0.0, 1.0, 0.0])
    assert isinstance(scores, np.ndarray)
    assert array_equal(scores, expected)


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
