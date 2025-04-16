from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from pydvl.utils.array import try_torch_import
from pydvl.utils.types import SupervisedModel
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.scorers import ClasswiseSupervisedScorer

torch = try_torch_import()


class ThresholdClassifier:
    def fit(self, x: NDArray, y: NDArray | None) -> ThresholdClassifier:
        return self

    def predict(self, x: NDArray) -> NDArray:
        y = x > 0.5
        return y[:, 0].astype(int)

    def score(self, x: NDArray, y: NDArray | None) -> float:
        assert y is not None
        return float(np.equal(self.predict(x), y).mean())


@pytest.fixture
def model() -> SupervisedModel:
    model = ThresholdClassifier()
    return model


@pytest.mark.parametrize(
    "test_data, expected_scores",
    [
        (
            Dataset(
                x=np.asarray([0.0, 0.5, 1.0]).reshape(-1, 1),
                y=np.asarray([0, 0, 0]),
            ),
            {0: np.nan, 1: 0.0},
        ),
        (
            Dataset(
                x=np.asarray([0.0, 0.5, 1.0]).reshape(-1, 1),
                y=np.asarray([1, 1, 1]),
            ),
            {0: 0.0, 1: np.nan},
        ),
        (
            Dataset(
                x=np.asarray([0.0, 0.5, 1.0]).reshape(-1, 1),
                y=np.asarray([0, 0, 1]),
            ),
            {0: 0.93, 1: 0.64},
        ),
    ],
)
def test_classwise_scorer(
    model: SupervisedModel, test_data: Dataset, expected_scores: dict[int, float]
):
    scorer = ClasswiseSupervisedScorer("accuracy", test_data)

    for label, expected_score in expected_scores.items():
        scorer.label = label
        score = scorer(model)
        np.testing.assert_allclose(score, expected_score, atol=1e-2)


@pytest.mark.torch
def test_classwise_scorer_with_tensor():
    """Test ClasswiseSupervisedScorer with PyTorch tensor inputs."""
    if torch is None:
        pytest.skip("PyTorch not installed")
        
    class TorchThresholdClassifier:
        def fit(self, x, y=None):
            return self
            
        def predict(self, x):
            y = x > 0.5
            return y[:, 0].long()
            
        def score(self, x, y=None):
            assert y is not None
            return float((self.predict(x) == y).float().mean().item())
    
    model = TorchThresholdClassifier()
    
    # Create test data with torch tensors
    test_data = Dataset(
        x=torch.tensor([[0.0], [0.5], [1.0]]),
        y=torch.tensor([0, 0, 1]),
    )
    
    # Expected scores should match the numpy version
    expected_scores = {0: 0.93, 1: 0.64}
    
    scorer = ClasswiseSupervisedScorer("accuracy", test_data)
    
    for label, expected_score in expected_scores.items():
        scorer.label = label
        score = scorer(model)
        np.testing.assert_allclose(score, expected_score, atol=1e-2)
