import numpy as np
import pytest

from sklearn.linear_model import LinearRegression
from valuation.shapley import combinatorial_exact_shapley
from valuation.utils import Dataset


@pytest.fixture(scope="module")
def boston_dataset():
    from sklearn import datasets
    return Dataset(datasets.load_boston())


@pytest.fixture(scope="module")
def linear_dataset():
    from sklearn.utils import Bunch
    a = 2
    b = 0
    x = np.arange(-1, 1, .2)
    y = np.random.normal(loc=a * x + b, scale=1)
    db = Bunch()
    db.data, db.target = x.reshape(-1, 1), y.reshape(-1, 1)
    db.DESCR = f"y~{a}*x + {b}"
    db.feature_names = ["x"]
    return Dataset(data=db, train_size=0.66)


@pytest.fixture(scope="module")
def exact_shapley(linear_dataset):
    model = LinearRegression()
    values_c = combinatorial_exact_shapley(model, linear_dataset,
                                           progress=False)
    return model, linear_dataset, np.array(list(values_c.values()))
