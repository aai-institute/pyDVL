import numpy as np
import pytest

from sklearn.linear_model import LinearRegression
from typing import Type
from valuation.shapley import combinatorial_exact_shapley
from valuation.utils import Dataset


@pytest.fixture(scope="module")
def boston_dataset():
    from sklearn import datasets
    return Dataset.from_sklearn(datasets.load_boston())


@pytest.fixture(scope="module")
def linear_dataset():
    from sklearn.utils import Bunch
    a = 2
    b = 0
    x = np.arange(-1, 1, .15)
    y = np.random.normal(loc=a * x + b, scale=1)
    db = Bunch()
    db.data, db.target = x.reshape(-1, 1), y.reshape(-1, 1)
    db.DESCR = f"y~N({a}*x + {b}, 1)"
    db.feature_names = ["x"]
    db.target_names = ["y"]
    return Dataset.from_sklearn(data=db, train_size=0.66)


@pytest.fixture()
def scoring():
    return 'r2'


@pytest.fixture()
def exact_shapley(linear_dataset, scoring):
    model = LinearRegression()
    values_c = combinatorial_exact_shapley(
            model, linear_dataset, scoring=scoring, progress=False)
    return model, linear_dataset, values_c, scoring


class TolerateErrors:
    def __init__(self, max_errors: int, exception_cls: Type[BaseException]):
        self.max_errors = max_errors
        self.Exception = exception_cls
        self.error_count = 0

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error_count += 1
        if self.error_count > self.max_errors:
            raise self.Exception(
                f"Maximum number of {self.max_errors} error(s) reached")
        return True
