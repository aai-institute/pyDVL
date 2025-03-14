import numpy as np
import sklearn
from numpy.typing import NDArray
from packaging import version

from pydvl.utils.score import Scorer, compose_score, squashed_r2, squashed_variance

sigmoid = lambda x: 1 / (1 + np.exp(-x))


class FittedLinearModel:
    def __init__(self, coef: NDArray):
        self.coef = coef

    def predict(self, X: NDArray) -> NDArray:
        return X @ self.coef

    def fit(self, X: NDArray, y: NDArray):
        pass

    def score(self, X: NDArray, y: NDArray) -> float:
        return np.linalg.norm(X @ self.coef - y)


def test_scorer():
    """Tests the Scorer class."""
    scorer = Scorer("r2")
    assert str(scorer) == "r2"
    if version.parse(sklearn.__version__) >= version.parse("1.4.0"):
        assert (
            repr(scorer)
            == "R2 (scorer=make_scorer(r2_score, response_method='predict'))"
        )
    else:
        assert repr(scorer) == "R2 (scorer=make_scorer(r2_score))"

    coef = np.array([1, 2])
    X = np.array([[1, 2], [3, 4]])
    model = FittedLinearModel(coef)
    assert 1.0 == scorer(model, X, X @ coef)


def test_compose_score():
    """Tests the compose_score function."""
    composed = compose_score(Scorer("r2"), sigmoid, (0, 1), "squashed r2")
    assert str(composed) == "squashed r2"
    assert repr(composed) == "SquashedR2 (scorer=r2)"

    coef = np.array([1, 2])
    X = np.array([[1, 2], [3, 4]])
    model = FittedLinearModel(coef)
    assert sigmoid(1.0) == composed(model, X, X @ coef)


def test_squashed_r2():
    """Tests the squashed_r2 scorer."""
    assert str(squashed_r2) == "squashed r2"
    assert repr(squashed_r2) == "SquashedR2 (scorer=r2)"
    np.testing.assert_allclose(squashed_r2.range, (0, 1))

    coef = np.array([1, 2])
    X = np.array([[1, 2], [3, 4]])
    model = FittedLinearModel(coef)
    assert sigmoid(1.0) == squashed_r2(model, X, X @ coef)


def test_squashed_variance():
    """Tests the squashed_variance scorer."""
    assert str(squashed_variance) == "squashed explained variance"
    assert (
        repr(squashed_variance)
        == "SquashedExplainedVariance (scorer=explained_variance)"
    )
    np.testing.assert_allclose(squashed_variance.range, (0, 1))

    coef = np.array([1, 2])
    X = np.array([[1, 2], [3, 4]])
    model = FittedLinearModel(coef)
    assert sigmoid(1.0) == squashed_variance(model, X, X @ coef)
