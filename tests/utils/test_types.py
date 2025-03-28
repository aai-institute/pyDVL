import numpy as np
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

from pydvl.utils.types import BaggingModel, validate_number


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
    assert isinstance(model, BaggingModel), (
        f"{model_class.__name__} should be recognized as a bagging model"
    )


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
    assert not isinstance(model, BaggingModel), (
        f"{model_class.__name__} should not be recognized as a bagging model"
    )


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
    assert not isinstance(model, BaggingModel), (
        f"{model_class.__name__} should not be recognized as a bagging model"
    )


@pytest.mark.parametrize(
    "value, dtype, lower, upper",
    [
        (42, int, None, None),
        (3.14, float, None, None),
        # With bounds
        (5, int, 0, 10),
        (3.14, float, 0.0, 5.0),
        (0, int, 0, 10),
        (10, int, 0, 10),
        # Type conversions
        (np.int32(42), int, None, None),
        (np.int64(42), int, -np.inf, None),
        (np.float32(3.14), float, None, None),
        (np.float64(3.14), float, None, np.inf),
        (np.float16(123.456), float, None, None),
        # Type conversion cases - only float to int
        (42.0, int, None, None),
        # Especial cases
        (float("nan"), float, None, None),
        (float("inf"), float, None, None),
        (float("-inf"), float, None, None),
    ],
)
def test_validate_number(value, dtype, lower, upper):
    result = validate_number("test", value, dtype, lower=lower, upper=upper)
    if np.isnan(value) and np.isnan(result):
        assert True
    else:
        assert isinstance(result, dtype)
        np.testing.assert_allclose(result, value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "value, dtype, exception",
    [
        ("not_a_number", int, TypeError),
        ("invalid", float, TypeError),
        (None, int, TypeError),
        (42.3, int, ValueError),
        (np.pi, np.float16, ValueError),
    ],
)
def test_validate_number_type_error(value, dtype, exception):
    with pytest.raises(exception):
        validate_number("test", value, dtype)


def test_validate_number_bound_error():
    with pytest.raises(ValueError, match=r"'test' is -1, but it should be >= 0"):
        validate_number("test", -1, int, lower=0)

    with pytest.raises(ValueError, match=r"'test' is 11, but it should be <= 10"):
        validate_number("test", 11, int, upper=10)

    with pytest.raises(ValueError, match=r"'test' is -5, but it should be >= 0"):
        validate_number("test", -5, int, lower=0, upper=10)

    with pytest.raises(ValueError, match=r"'test' is 15, but it should be <= 10"):
        validate_number("test", 15, int, lower=0, upper=10)
