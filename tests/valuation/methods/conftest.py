from __future__ import annotations

from typing import Self

import pytest

from pydvl.utils import Seed, try_torch_import
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.games import DummyGameDataset, MinerGame, ShoesGame
from pydvl.valuation.scorers import ClasswiseSupervisedScorer
from pydvl.valuation.utility.classwise import ClasswiseModelUtility

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
