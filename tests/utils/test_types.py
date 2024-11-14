import pytest
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)

from pydvl.utils.types import BaggingModel


@pytest.mark.parametrize(
    "model_class",
    [
        BaggingClassifier,
        BaggingRegressor,
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        IsolationForest,
        RandomForestClassifier,
        RandomForestRegressor,
    ],
)
def test_is_bagging_model(model_class):
    model = model_class()
    assert isinstance(
        model, BaggingModel
    ), f"{model_class.__name__} should be recognized as a bagging model"


@pytest.mark.parametrize(
    "model_class",
    [
        AdaBoostClassifier,
        AdaBoostRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    ],
)
def test_is_not_bagging_model(model_class):
    model = model_class()
    assert not isinstance(
        model, BaggingModel
    ), f"{model_class.__name__} should not be recognized as a bagging model"


@pytest.mark.parametrize(
    "model_class",
    [
        StackingClassifier,
        StackingRegressor,
        VotingClassifier,
        VotingRegressor,
    ],
)
def test_is_not_bagging_model_other(model_class):
    model = model_class(estimators=[("est", RandomForestClassifier())])
    assert not isinstance(
        model, BaggingModel
    ), f"{model_class.__name__} should not be recognized as a bagging model"
