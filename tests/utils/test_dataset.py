import numpy as np
import pytest
from sklearn.datasets import load_wine, make_classification

from pydvl.utils.dataset import Dataset, GroupedDataset


@pytest.fixture(scope="module", params=[0.1, 0.5, 0.8])
def train_size(request):
    return request.param


def test_creating_dataset_from_sklearn(train_size):
    data = load_wine()
    dataset = Dataset.from_sklearn(data, train_size=train_size)
    assert len(dataset) == int(train_size * len(data.data))


def test_creating_dataset_subsclassfrom_sklearn(train_size):
    data = load_wine()

    class TestDataset(Dataset):
        ...

    dataset = TestDataset.from_sklearn(data, train_size=train_size)
    assert isinstance(dataset, TestDataset)
    assert len(dataset) == int(train_size * len(data.data))


def test_creating_dataset_from_x_y_arrays(train_size):
    X, y = make_classification()
    dataset = Dataset.from_arrays(X, y, train_size=train_size)
    assert len(dataset) == int(train_size * len(X))


def test_creating_grouped_dataset_from_sklearn(train_size):
    data = load_wine()
    data_groups = (
        np.random.randint(low=0, high=3, size=int(train_size * len(data.data)))
        .flat()
        .tolist()
    )
    n_groups = len(np.unique(data_groups))
    dataset = GroupedDataset.from_sklearn(
        data, data_groups=data_groups, train_size=train_size
    )
    assert len(dataset) == n_groups


def test_creating_grouped_dataset_subsclassfrom_sklearn(train_size):
    data = load_wine()

    class TestGroupedDataset(GroupedDataset):
        ...

    data_groups = (
        np.random.randint(low=0, high=3, size=int(train_size * len(data.data)))
        .flat()
        .tolist()
    )
    n_groups = len(np.unique(data_groups))
    dataset = TestGroupedDataset.from_sklearn(
        data, data_groups=data_groups, train_size=train_size
    )
    assert isinstance(dataset, TestGroupedDataset)
    assert len(dataset) == n_groups


def test_creating_grouped_dataset_from_x_y_arrays(train_size):
    X, y = make_classification()
    data_groups = (
        np.random.randint(low=0, high=3, size=int(train_size * len(X))).flat().tolist()
    )
    n_groups = len(np.unique(data_groups))
    dataset = GroupedDataset.from_arrays(
        X, y, data_groups=data_groups, train_size=train_size
    )
    assert len(dataset) == n_groups
