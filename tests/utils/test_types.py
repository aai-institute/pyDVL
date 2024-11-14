import pytest
from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from pydvl.utils.types import is_bagging_model


@pytest.mark.parametrize(
    "model_class",
    [
        BaggingClassifier,
        BaggingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
        ExtraTreesClassifier,
        ExtraTreesRegressor,
    ],
)
def test_is_bagging_model(model_class):
    model = model_class()
    assert is_bagging_model(
        model
    ), f"{model_class.__name__} should be recognized as a bagging model"


def test_is_not_bagging_model():
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    assert not is_bagging_model(
        model
    ), "LinearRegression should not be recognized as a bagging model"
