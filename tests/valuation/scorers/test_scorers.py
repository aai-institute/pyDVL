import numpy as np
import pytest
import sklearn
from numpy.typing import NDArray
from packaging import version

from pydvl.valuation.dataset import Dataset
from pydvl.valuation.scorers import SupervisedScorer, compose_score, sigmoid


class FittedLinearModel:
    def __init__(self, coef: NDArray):
        self.coef = coef

    def predict(self, X):
        return X @ self.coef

    def fit(self, X, y):
        pass

    def score(self, X, y):
        return np.linalg.norm(X @ self.coef - y)


@pytest.fixture
def model():
    coef = np.array([1, 2])
    return FittedLinearModel(coef)


@pytest.fixture
def scorer(model):
    coef = model.coef
    X = np.array([[1, 2], [3, 4]])
    y = X @ coef
    test_data = Dataset(X, y)
    return SupervisedScorer("r2", test_data=test_data, default=0.0)


def test_scorer(model, scorer):
    """Tests the Scorer class."""
    assert str(scorer) == "r2(default=0.0, range=(-inf, inf))"
    if version.parse(sklearn.__version__) >= version.parse("1.4.0"):
        assert (
            repr(scorer)
            == "R2 (scorer=make_scorer(r2_score, response_method='predict'))"
        )
    else:
        assert repr(scorer) == "R2 (scorer=make_scorer(r2_score))"

    assert 1.0 == scorer(model)


def test_compose_score(model, scorer):
    """Tests the compose_score function."""
    composed = compose_score(scorer, sigmoid, name="squashed r2")
    assert str(composed) == "squashed r2(default=0.5, range=(0.0, 1.0))"

    assert composed.range[0] == 0
    assert composed.range[1] == 1
    assert scorer.range[0] == -np.inf
    assert scorer.range[1] == np.inf

    assert composed.default == sigmoid(scorer.default)

    assert sigmoid(1.0) == composed(model)
