import numpy as np
import pytest
from sklearn.datasets import load_wine, make_classification

from pydvl.valuation.dataset import Dataset, GroupedDataset, RawData
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

    class TestDataset(Dataset): ...

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
    assert len(train.data().x) == int(train_size * len(data.data))


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

    class TestGroupedDataset(GroupedDataset): ...

    data_groups = np.random.randint(low=0, high=3, size=len(data.data)).flatten()
    n_groups = len(np.unique(data_groups))
    train, test = TestGroupedDataset.from_sklearn(
        data, data_groups=data_groups, train_size=train_size
    )
    assert isinstance(train, TestGroupedDataset)
    assert len(train) <= n_groups
    assert len(test) <= n_groups
    assert len(train.data().x) == int(train_size * len(data.data))


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
    assert len(train.data().x) == int(train_size * len(X))
    for k, v in kwargs.items():
        assert getattr(train, k) == v


def test_creating_grouped_dataset_from_x_y_arrays_failure(train_size):
    X, y = make_classification()
    # The length of data groups should be equal to that of X and y
    data_groups_length = np.random.randint(low=0, high=len(X) - 1)
    data_groups = np.random.randint(low=0, high=3, size=data_groups_length).flatten()
    with pytest.raises(ValueError):
        GroupedDataset.from_arrays(X, y, data_groups=data_groups)
    with pytest.raises(ValueError, match="data_groups must be provided"):
        GroupedDataset.from_arrays(X, y, train_size=train_size)


def test_grouped_dataset_results():
    """Test that data names are preserved in valuation results"""
    X, y = make_classification()
    train_size = 0.5
    data_groups = np.random.randint(low=0, high=3, size=len(X)).flatten()
    train, test = GroupedDataset.from_arrays(
        X, y, train_size=train_size, data_groups=data_groups
    )

    v = ValuationResult.zeros(indices=train.indices, data_names=train.names)
    v2 = ValuationResult(
        indices=train.indices,
        values=np.ones(len(train)),
        variances=np.zeros(len(train)),
        data_names=train.names,
    )
    v += v2
    assert np.all(v.values == 1)
    assert np.all(v.names == np.array(train.names))
    assert np.all([isinstance(x, object) for x in v.names])


@pytest.mark.parametrize(
    "idx, expected_x, expected_y, expected_groups, expected_names, expected_group_names",
    [
        (
            0,
            np.array([[1, 2], [5, 6]]),
            np.array([0, -1]),
            [0, 0],
            ["a", "c"],
            ["group1"],
        ),
        (
            slice(0, 2),
            np.array([[1, 2], [5, 6], [3, 4]]),
            np.array([0, -1, 1]),
            [0, 0, 1],
            ["a", "c", "b"],
            ["group1", "group2"],
        ),
        (
            [0, 2],
            np.array([[1, 2], [5, 6], [7, 8]]),
            np.array([0, -1, 1]),
            [0, 0, 2],
            ["a", "c", "d"],
            ["group1", "group3"],
        ),
    ],
)
def test_getitem_returns_correct_grouped_dataset(
    idx, expected_x, expected_y, expected_groups, expected_names, expected_group_names
):
    dataset = GroupedDataset(
        x=np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
        y=np.array([0, 1, -1, 1]),
        data_groups=[0, 1, 0, 2],
        data_names=["a", "b", "c", "d"],
        group_names=["group1", "group2", "group3"],
    )
    sliced_dataset = dataset[idx]
    assert np.array_equal(sliced_dataset._x, expected_x)
    assert np.array_equal(sliced_dataset._y, expected_y)
    assert np.array_equal(sliced_dataset.data_to_group, expected_groups)
    assert np.array_equal(sliced_dataset._data_names, expected_names)
    assert np.array_equal(sliced_dataset.names, expected_group_names)


def test_default_group_names():
    """Test that default group_names are set to the string representations of group ids
    when not provided."""
    x = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    data_groups = [0, 1, 0]
    dataset = GroupedDataset(x=x, y=y, data_groups=data_groups)
    # Default group_names should be created as {group_id: str(group_id)} for each group
    # present.
    expected = ["0", "1"]
    assert all(dataset.names == expected)


def test_incomplete_group_names():
    """Test that providing an incomplete group_names dictionary raise an exception."""
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, -1, 1])
    with pytest.raises(ValueError, match="The number of group names"):
        _ = GroupedDataset(
            x=x,
            y=y,
            data_groups=[0, 1, 0, 2],
            group_names=["g1", "g3"],
        )


@pytest.mark.parametrize(
    "idx, expected_x, expected_y, expected_names",
    [
        (0, np.array([[1, 2]]), np.array([0]), ["a"]),
        (slice(0, 2), np.array([[1, 2], [3, 4]]), np.array([0, 1]), ["a", "b"]),
        ([0, 2], np.array([[1, 2], [5, 6]]), np.array([0, 0]), ["a", "c"]),
    ],
)
def test_getitem_returns_correct_dataset(idx, expected_x, expected_y, expected_names):
    dataset = Dataset(
        x=np.array([[1, 2], [3, 4], [5, 6]]),
        y=np.array([0, 1, 0]),
        data_names=["a", "b", "c"],
    )
    sliced_dataset = dataset[idx]
    assert np.array_equal(sliced_dataset._x, expected_x)
    assert np.array_equal(sliced_dataset._y, expected_y)
    assert np.array_equal(sliced_dataset._data_names, expected_names)


@pytest.mark.parametrize(
    "idx",
    [[0], slice(0, 2), [0, 2]],
)
def test_dataset_slice_data_access(idx):
    dataset = Dataset(
        x=np.array([[1, 2], [3, 4], [5, 6]]),
        y=np.array([0, 1, 0]),
        data_names=["a", "b", "c"],
    )
    assert np.array_equal(dataset[idx].data().x, dataset.data().x[idx])
    assert np.array_equal(dataset[idx].data().y, dataset.data().y[idx])


@pytest.mark.parametrize(
    "x, y, feature_names, target_names, multi_output, expected_exception",
    [
        (
            np.array([[1, 2], [3, 4]]),
            np.array([0, 1]),
            ["f1", "f2"],
            ["t1"],
            False,
            None,
        ),
        (
            np.array([[1, 2], [3, 4]]),
            np.array([0, 1]),
            ["f1"],
            ["t1"],
            False,
            ValueError,
        ),
        (
            np.array([[1, 2], [3, 4]]),
            np.array([[0], [1]]),
            ["f1", "f2"],
            ["t1"],
            False,
            None,
        ),
        (
            np.array([[1, 2], [3, 4]]),
            np.array([[0, 1], [1, 0]]),
            ["f1", "f2"],
            ["t1", "t2"],
            True,
            None,
        ),
        (
            np.array([[1, 2], [3, 4]]),
            np.array([[0, 1], [1, 0]]),
            ["f1", "f2"],
            ["t1"],
            True,
            ValueError,
        ),
    ],
)
def test_mismatching_features_and_names(
    x, y, feature_names, target_names, multi_output, expected_exception
):
    if expected_exception:
        with pytest.raises(expected_exception):
            Dataset(
                x=x,
                y=y,
                feature_names=feature_names,
                target_names=target_names,
                multi_output=multi_output,
            )
    else:
        dataset = Dataset(
            x=x,
            y=y,
            feature_names=feature_names,
            target_names=target_names,
            multi_output=multi_output,
        )
        assert dataset.feature_names == feature_names
        assert dataset.target_names == target_names


@pytest.mark.parametrize(
    "x, feature_names, feature_to_find, expected_slice",
    [
        (
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            ["f1", "f2", "f3"],
            "f2",
            np.array([2, 5, 8]),
        ),
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            ["feature1", "feature2"],
            "feature1",
            np.array([1, 3, 5]),
        ),
        (
            np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
            ["a", "b", "c", "d"],
            "d",
            np.array([4, 8, 12]),
        ),
    ],
)
def test_feature_returns_correct_slice(
    x, feature_names, feature_to_find, expected_slice
):
    dataset = Dataset(
        x=x,
        y=np.array([0, 1, 0]),
        feature_names=feature_names,
    )
    indices = dataset.feature(feature_to_find)
    assert np.array_equal(dataset.data().x[indices], expected_slice)


@pytest.mark.parametrize(
    "x, feature_names, feature_to_find",
    [
        (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), ["f1", "f2", "f3"], "f4"),
        (np.array([[1, 2], [3, 4], [5, 6]]), ["feature1", "feature2"], "feature3"),
        (
            np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
            ["a", "b", "c", "d"],
            "e",
        ),
    ],
)
def test_feature_raises_value_error_for_invalid_feature(
    x, feature_names, feature_to_find
):
    dataset = Dataset(
        x=x,
        y=np.array([0, 1, 0]),
        feature_names=feature_names,
    )
    with pytest.raises(ValueError, match=f"Feature {feature_to_find} is not in"):
        dataset.feature(feature_to_find)


@pytest.mark.parametrize(
    "indices, expected_x, expected_y",
    [
        (None, np.array([[1, 2], [3, 4], [5, 6]]), np.array([0, 1, 0])),
        ([0, 2], np.array([[1, 2], [5, 6]]), np.array([0, 0])),
        (slice(1, 3), np.array([[3, 4], [5, 6]]), np.array([1, 0])),
    ],
)
def test_data_returns_correct_subset(indices, expected_x, expected_y):
    dataset = Dataset(
        x=np.array([[1, 2], [3, 4], [5, 6]]),
        y=np.array([0, 1, 0]),
    )
    result = dataset.data(indices)
    assert np.array_equal(result.x, expected_x)
    assert np.array_equal(result.y, expected_y)


@pytest.mark.parametrize(
    "indices, expected_indices",
    [
        (None, np.array([0, 1, 2])),
        ([0, 2], np.array([0, 2])),
        (slice(1, 3), np.array([1, 2])),
    ],
)
def test_data_indices_returns_correct_indices(indices, expected_indices):
    dataset = Dataset(
        x=np.array([[1, 2], [3, 4], [5, 6]]),
        y=np.array([0, 1, 0]),
    )
    result = dataset.data_indices(indices)
    assert np.array_equal(result, expected_indices)


@pytest.mark.parametrize(
    "indices, expected_indices",
    [
        (None, np.array([0, 1, 2])),
        ([0, 2], np.array([0, 2])),
        (slice(1, 3), np.array([1, 2])),
    ],
)
def test_logical_indices_returns_correct_indices(indices, expected_indices):
    dataset = Dataset(
        x=np.array([[1, 2], [3, 4], [5, 6]]),
        y=np.array([0, 1, 0]),
    )
    result = dataset.logical_indices(indices)
    assert np.array_equal(result, expected_indices)


@pytest.mark.parametrize(
    "target_name, expected_slice",
    [
        ("y1", np.index_exp[:, 0]),
        ("y2", np.index_exp[:, 1]),
    ],
)
def test_target_returns_correct_slice(target_name, expected_slice):
    dataset = Dataset(
        x=np.array([[1, 2], [3, 4], [5, 6]]),
        y=np.array([[0, 1], [1, 0], [0, 1]]),
        target_names=["y1", "y2"],
        multi_output=True,
    )
    result = dataset.target(target_name)
    assert result == expected_slice


@pytest.mark.parametrize(
    "target_name",
    ["y3", "invalid_target"],
)
def test_target_raises_value_error_for_invalid_target(target_name):
    dataset = Dataset(
        x=np.array([[1, 2], [3, 4], [5, 6]]),
        y=np.array([[0, 1], [1, 0], [0, 1]]),
        target_names=["y1", "y2"],
        multi_output=True,
    )
    with pytest.raises(ValueError, match=f"Target {target_name} is not in"):
        dataset.target(target_name)


def test_indices_property_returns_correct_indices():
    dataset = Dataset(
        x=np.array([[1, 2], [3, 4], [5, 6]]),
        y=np.array([0, 1, 0]),
    )
    assert np.array_equal(dataset.indices, np.array([0, 1, 2]))


def test_names_property_returns_correct_names():
    dataset = Dataset(
        x=np.array([[1, 2], [3, 4], [5, 6]]),
        y=np.array([0, 1, 0]),
        data_names=["a", "b", "c"],
    )
    assert np.array_equal(dataset.names, np.array(["a", "b", "c"]))


def test_n_features_property_returns_correct_dimension():
    dataset = Dataset(
        x=np.array([[1, 2], [3, 4], [5, 6]]),
        y=np.array([0, 1, 0]),
    )
    assert dataset.n_features == 2


def test_str_method_returns_description():
    dataset = Dataset(
        x=np.array([[1, 2], [3, 4], [5, 6]]),
        y=np.array([0, 1, 0]),
        description="Test dataset",
    )
    assert str(dataset) == "Test dataset"


@pytest.mark.parametrize(
    "x, data_groups, expected_exception",
    [
        (np.array([[1, 2], [3, 4], [5, 6]]), [0, 1, 0], None),
        (np.array([[1, 2], [3, 4], [5, 6]]), [0, 1], ValueError),
    ],
)
def test_data_groups_and_x_length_check(x, data_groups, expected_exception):
    if expected_exception:
        with pytest.raises(
            expected_exception, match="data_groups and x must have the same length"
        ):
            GroupedDataset(x=x, y=np.array([0, 1, 0]), data_groups=data_groups)
    else:
        dataset = GroupedDataset(x=x, y=np.array([0, 1, 0]), data_groups=data_groups)
        assert len(dataset.data_to_group) == len(x)


@pytest.mark.parametrize(
    "indices, expected_x, expected_y",
    [
        (None, np.array([[1, 2], [5, 6], [3, 4], [7, 8]]), np.array([0, 3, 1, 1])),
        ([0, 2], np.array([[1, 2], [5, 6], [7, 8]]), np.array([0, 3, 1])),
        (slice(1, 3), np.array([[3, 4], [7, 8]]), np.array([1, 1])),
    ],
)
def test_grouped_data_returns_correct_subset(indices, expected_x, expected_y):
    dataset = GroupedDataset(
        x=np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
        y=np.array([0, 1, 3, 1]),
        data_groups=[0, 1, 0, 2],
    )
    result = dataset.data(indices)
    assert np.array_equal(result.x, expected_x)
    assert np.array_equal(result.y, expected_y)


@pytest.mark.parametrize(
    "indices, expected_indices",
    [
        (None, np.array([0, 2, 1, 3])),
        ([0, 2], np.array([0, 2, 3])),
        (slice(1, 3), np.array([1, 3])),
    ],
)
def test_grouped_data_indices_returns_correct_indices(indices, expected_indices):
    dataset = GroupedDataset(
        x=np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
        y=np.array([0, 1, 0, 1]),
        data_groups=[0, 1, 0, 2],
    )
    result = dataset.data_indices(indices)
    assert np.array_equal(result, expected_indices)


@pytest.mark.parametrize(
    "data_indices, expected_indices",
    [
        (None, np.array([0, 1, 0])),
        ([0, 2], np.array([0, 0])),
        (slice(1, 3), np.array([1, 0])),
    ],
)
def test_grouped_logical_indices_returns_correct_indices(
    data_indices, expected_indices
):
    dataset = GroupedDataset(
        x=np.array([[1, 2], [3, 4], [5, 6]]),
        y=np.array([0, 1, 0]),
        data_groups=[0, 1, 0],
    )
    result = dataset.logical_indices(data_indices)
    assert np.array_equal(result, expected_indices)


@pytest.mark.parametrize(
    "x, y",
    [
        (np.array([1, 2]), np.array([1, 2, 3])),
        (np.array([1, 2, 3]), np.array([1, 2])),
        (np.zeros((3, 2)), np.zeros(2)),
    ],
    ids=["x_shorter", "y_shorter", "different_shapes"],
)
def test_rawdata_creation_raises_on_length_mismatch(x, y):
    with pytest.raises(ValueError, match="x and y must have the same length"):
        RawData(x, y)


@pytest.mark.parametrize(
    "x, y", [(np.array([[1]]), 1), (3, np.array([1])), (1, 2), (None, None)]
)
def test_rawdata_creation_raises_on_non_arrays(x, y):
    with pytest.raises(TypeError, match="must be arrays of the same type"):
        RawData(x, y)


def test_rawdata_iteration():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    data = RawData(x, y)
    unpacked_x, unpacked_y = data
    assert np.array_equal(unpacked_x, x)
    assert np.array_equal(unpacked_y, y)


@pytest.mark.parametrize(
    "idx, expected_x, expected_y",
    [
        (
            1,
            np.array([2]),
            np.array([5]),
        ),
        (
            slice(1, None),
            np.array([2, 3]),
            np.array([5, 6]),
        ),
        (
            [0, 2],
            np.array([1, 3]),
            np.array([4, 6]),
        ),
    ],
    ids=["single_index", "slice", "sequence"],
)
def test_rawdata_getitem(idx, expected_x, expected_y):
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    data = RawData(x, y)
    result = data[idx]
    assert isinstance(result, RawData)
    assert np.array_equal(result.x, expected_x)
    assert np.array_equal(result.y, expected_y)


@pytest.mark.parametrize(
    "x, y, length",
    [
        (np.array([1, 2, 3]), np.array([4, 5, 6]), 3),
        (np.zeros((5, 2)), np.zeros(5), 5),
        (np.array([]), np.array([]), 0),
    ],
    ids=["1d_arrays", "2d_arrays", "empty_arrays"],
)
def test_rawdata_len(x, y, length):
    data = RawData(x, y)
    assert len(data) == length
