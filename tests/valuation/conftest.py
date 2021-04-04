import pytest
from valuation.utils import Dataset


@pytest.fixture(scope="module")
def dataset():
    from sklearn import datasets
    return Dataset(datasets.load_boston())
