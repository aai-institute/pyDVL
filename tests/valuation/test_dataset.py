import numpy as np
import pytest
from sklearn.datasets import load_wine, make_classification

from pydvl.valuation.dataset import Dataset, GroupedDataset
from pydvl.valuation.result import ValuationResult


@pytest.fixture(scope="module", params=[0.1, 0.5, 0.8])
def train_size(request):
    return request.param


def test_creating_dataset_from_sklearn(train_size):
    data = load_wine()
    train, test = Dataset.from_sklearn(data, train_size=train_size)
    assert len(train) == int(train_size * len(data.data))


def test_creating_dataset_subsclassfrom_sklearn(train_size):
    data = load_wine()

    class TestDataset(Dataset):
        ...

    train, test = TestDataset.from_sklearn(data, train_size=train_size)
    assert isinstance(train, TestDataset)
    assert isinstance(test, TestDataset)
    assert len(train) == int(train_size * len(data.data))


@pytest.mark.parametrize("kwargs", ({}, {"description": "Test Dataset"}))
def test_creating_dataset_from_x_y_arrays(train_size, kwargs):
    X, y = make_classification()
    train, test = Dataset.from_arrays(X, y, train_size=train_size, **kwargs)
    assert len(train) == int(train_size * len(X))
    for k, v in kwargs.items():
        assert getattr(train, k) == v


def test_creating_grouped_dataset_from_sklearn(train_size):
    data = load_wine()
    data_groups = np.random.randint(low=0, high=3, size=len(data.data)).flatten()
    n_groups = len(np.unique(data_groups))
    train, test = GroupedDataset.from_sklearn(
        data, data_groups=data_groups, train_size=train_size
    )
    assert len(train) <= n_groups
    assert len(test) <= n_groups
    assert len(train.get_data()[0]) == int(train_size * len(data.data))


def test_creating_grouped_dataset_from_sklearn_failure(train_size):
    with pytest.raises(ValueError):
        data = load_wine()
        # The length of data groups should be equal to that of data
        data_groups_length = np.random.randint(low=0, high=len(data.data) - 1)
        data_groups = np.random.randint(
            low=0, high=3, size=data_groups_length
        ).flatten()
        GroupedDataset.from_sklearn(data, data_groups=data_groups)


def test_creating_grouped_dataset_subsclassfrom_sklearn(train_size):
    data = load_wine()

    class TestGroupedDataset(GroupedDataset):
        ...

    data_groups = np.random.randint(low=0, high=3, size=len(data.data)).flatten()
    n_groups = len(np.unique(data_groups))
    train, test = TestGroupedDataset.from_sklearn(
        data, data_groups=data_groups, train_size=train_size
    )
    assert isinstance(train, TestGroupedDataset)
    assert len(train) <= n_groups
    assert len(test) <= n_groups
    assert len(train.get_data()[0]) == int(train_size * len(data.data))


@pytest.mark.parametrize("kwargs", ({}, {"description": "Test Dataset"}))
def test_creating_grouped_dataset_from_x_y_arrays(train_size, kwargs):
    X, y = make_classification()
    data_groups = np.random.randint(low=0, high=3, size=len(X)).flatten()
    n_groups = len(np.unique(data_groups))
    train, test = GroupedDataset.from_arrays(
        X, y, data_groups=data_groups, train_size=train_size, **kwargs
    )
    assert len(train) <= n_groups
    assert len(test) <= n_groups
    assert len(train.get_data()[0]) == int(train_size * len(X))
    for k, v in kwargs.items():
        assert getattr(train, k) == v


def test_creating_grouped_dataset_from_x_y_arrays_failure(train_size):
    with pytest.raises(ValueError):
        X, y = make_classification()
        # The length of data groups should be equal to that of X and y
        data_groups_length = np.random.randint(low=0, high=len(X) - 1)
        data_groups = np.random.randint(
            low=0, high=3, size=data_groups_length
        ).flatten()
        GroupedDataset.from_arrays(X, y, data_groups=data_groups)


def test_grouped_dataset_results():
    """Test that data names are preserved in valuation results"""
    X, y = make_classification()
    train_size = 0.5
    data_groups = np.random.randint(low=0, high=3, size=len(X)).flatten()
    train, test = GroupedDataset.from_arrays(
        X, y, data_groups=data_groups, train_size=train_size
    )

    v = ValuationResult.zeros(indices=train.indices, data_names=train.data_names)
    v2 = ValuationResult(
        indices=train.indices,
        values=np.ones(len(train)),
        variances=np.zeros(len(train)),
        data_names=train.data_names,
    )
    v += v2
    assert np.all(v.values == 1)
    assert np.all(v.names == np.array(train.data_names))
    assert np.all([isinstance(x, np.int_) for x in v.names])
