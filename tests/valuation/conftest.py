import numpy as np
import pytest
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
    # x = np.array([-1., -0.6, -0.2, 0.7, 1.0])
    y = np.random.normal(loc=a * x + b, scale=1)
    # y = a*x + b
    db = Bunch()
    db.data, db.target = x.reshape(-1, 1), y.reshape(-1, 1)
    db.DESCR = f"y~{a}*x + {b}"
    db.feature_names = ["x"]
    db.target_names = ["y"]
    return Dataset(data=db, train_size=0.66)
