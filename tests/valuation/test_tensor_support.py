import numpy as np
import pytest
from sklearn.datasets import make_classification

from pydvl.utils.array import array_equal, is_numpy, try_torch_import
from pydvl.valuation.dataset import Dataset, GroupedDataset, RawData

torch = try_torch_import()
pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch not installed")


@pytest.fixture
def numpy_data():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([0, 1, 0])
    return X, y


@pytest.fixture
def torch_data(numpy_data):
    X, y = numpy_data
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.int64)
    return X_tensor, y_tensor


def test_rawdata_creation_with_tensors(torch_data):
    X, y = torch_data
    data = RawData(X, y)
    assert torch.is_tensor(data.x)
    assert torch.is_tensor(data.y)
    assert torch.equal(data.x, X)
    assert torch.equal(data.y, y)


def test_rawdata_type_preservation(torch_data):
    X, y = torch_data
    data = RawData(X, y)
    sliced_data = data[1:3]
    assert torch.is_tensor(sliced_data.x)
    assert torch.is_tensor(sliced_data.y)
    assert sliced_data.x.shape[0] == 2
    assert sliced_data.y.shape[0] == 2


def test_rawdata_mixed_input_types(numpy_data, torch_data):
    X_np, _ = numpy_data
    _, y_tensor = torch_data
    with pytest.raises(TypeError, match="must be arrays of the same type"):
        RawData(X_np, y_tensor)


@pytest.mark.parametrize(
    "accessor",
    [lambda ds: (ds._x, ds._y), lambda ds: (ds.data().x, ds.data().y)],
    ids=["direct", "data_method"],
)
def test_dataset_tensor_creation_and_data_method(torch_data, accessor):
    X, y = torch_data
    ds = Dataset(X, y)
    x_ret, y_ret = accessor(ds)
    assert torch.is_tensor(x_ret)
    assert torch.is_tensor(y_ret)
    assert torch.equal(x_ret, X)
    assert torch.equal(y_ret, y)


@pytest.mark.parametrize(
    "index, expected_length",
    [
        (0, 1),
        (slice(1, 3), 2),
    ],
)
def test_dataset_slicing_with_tensors_param(torch_data, index, expected_length):
    X, y = torch_data
    ds = Dataset(X, y)
    sliced_ds = ds[index]
    assert torch.is_tensor(sliced_ds._x)
    assert torch.is_tensor(sliced_ds._y)
    assert sliced_ds._x.shape[0] == expected_length


def test_dataset_from_arrays_with_tensors(torch_data):
    X, y = torch_data
    train, test = Dataset.from_arrays(X, y, train_size=0.66)
    assert torch.is_tensor(train._x)
    assert torch.is_tensor(train._y)
    assert torch.is_tensor(test._x)
    assert torch.is_tensor(test._y)
    assert len(train) + len(test) == len(X)


def test_dataset_from_arrays_stratified_with_tensors():
    # Create a larger dataset with more samples per class for stratified split to work
    X = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
            [13.0, 14.0],
            [15.0, 16.0],
            [17.0, 18.0],
        ]
    )
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0])

    train, test = Dataset.from_arrays(X, y, train_size=0.66, stratify_by_target=True)
    assert torch.is_tensor(train._x)
    assert torch.is_tensor(train._y)
    assert torch.is_tensor(test._x)
    assert torch.is_tensor(test._y)

    original_class_counts = [torch.sum(y == label).item() for label in torch.unique(y)]
    train_class_counts = [
        torch.sum(train._y == label).item() for label in torch.unique(train._y)
    ]

    for train_count, orig_count in zip(train_class_counts, original_class_counts):
        assert train_count > 0


def test_dataset_feature_method_with_tensors(torch_data):
    X, y = torch_data
    dataset = Dataset(
        X, y, feature_names=["feature1", "feature2"], target_names=["target"]
    )
    feature_slice = dataset.feature("feature1")
    feature_data = dataset.data().x[feature_slice]
    assert torch.is_tensor(feature_data)
    assert feature_data.shape == (3,)
    assert torch.equal(feature_data, X[:, 0])

    with pytest.raises(ValueError, match="Feature .* is not in"):
        dataset.feature("nonexistent_feature")


def test_dataset_target_method_with_tensors(torch_data):
    X, y = torch_data
    # Create a multi-output target (2D tensor) to properly test the target method
    y_2d = torch.stack([y, y + 1], dim=1)  # Create a [3, 2] tensor
    dataset = Dataset(
        X,
        y_2d,
        feature_names=["feature1", "feature2"],
        target_names=["target1", "target2"],
        multi_output=True,
    )

    # Get the slice for the first target
    target_slice = dataset.target("target1")
    target_data = dataset.data().y[target_slice]
    assert torch.is_tensor(target_data)
    assert target_data.shape == (3,)
    assert torch.equal(target_data, y)

    # Get the slice for the second target
    target_slice = dataset.target("target2")
    target_data = dataset.data().y[target_slice]
    assert torch.is_tensor(target_data)
    assert target_data.shape == (3,)
    assert torch.equal(target_data, y + 1)

    with pytest.raises(ValueError, match="Target .* is not in"):
        dataset.target("nonexistent_target")


@pytest.mark.parametrize("data_groups", [[0, 1, 0], torch.tensor([0, 1, 0])])
def test_grouped_dataset_creation_with_tensors_param(torch_data, data_groups):
    X, y = torch_data
    ds = GroupedDataset(X, y, data_groups=data_groups)
    assert torch.is_tensor(ds._x)
    assert torch.is_tensor(ds._y)
    assert is_numpy(ds.data_to_group)
    assert is_numpy(ds._indices)


def test_grouped_dataset_data_method_with_tensors(torch_data):
    X, y = torch_data
    data_groups = [0, 1, 0]
    dataset = GroupedDataset(X, y, data_groups=data_groups)

    group_data = dataset.data([0])
    assert torch.is_tensor(group_data.x)
    assert torch.is_tensor(group_data.y)
    assert group_data.x.shape[0] == 2
    assert torch.equal(group_data.x[0], X[0])
    assert torch.equal(group_data.x[1], X[2])


def test_grouped_dataset_from_arrays_with_tensors(torch_data):
    X, y = torch_data
    data_groups = [0, 1, 0]
    train, test = GroupedDataset.from_arrays(
        X, y, data_groups=data_groups, train_size=0.66
    )
    assert torch.is_tensor(train._x)
    assert torch.is_tensor(train._y)
    assert torch.is_tensor(test._x)
    assert torch.is_tensor(test._y)
    assert is_numpy(train.data_to_group)
    assert is_numpy(train._indices)


@pytest.mark.parametrize(
    "index, expected_length",
    [
        (0, 2),
        ([0, 1], 3),
    ],
)
def test_grouped_dataset_slicing_with_tensors_param(torch_data, index, expected_length):
    X, y = torch_data
    data_groups = [0, 1, 0]
    ds = GroupedDataset(X, y, data_groups=data_groups)
    sliced_ds = ds[index]
    assert torch.is_tensor(sliced_ds._x)
    assert torch.is_tensor(sliced_ds._y)
    assert sliced_ds._x.shape[0] == expected_length
    assert is_numpy(sliced_ds.data_to_group)
    assert is_numpy(sliced_ds._indices)


@pytest.mark.parametrize(
    "sel, expected, expected_length",
    [
        ([0], np.array([0, 2]), None),
        ([0, 1], np.array([0, 2, 1]), None),
        (None, np.array([0, 2, 1]), None),
        (slice(0, 1), None, 2),
    ],
)
def test_data_indices_with_tensors(torch_data, sel, expected, expected_length):
    X, y = torch_data
    data_groups = [0, 1, 0]
    dataset = GroupedDataset(X, y, data_groups=data_groups)
    indices = dataset.data_indices(sel)
    assert is_numpy(indices)
    if expected is not None:
        assert indices.shape[0] == expected.shape[0]
        assert array_equal(indices, expected)
    else:
        assert indices.shape[0] == expected_length


@pytest.mark.parametrize(
    "sel, expected",
    [
        (None, np.array([0, 1, 0])),
        ([0, 2], np.array([0, 0])),
        (torch.tensor([0, 2]), np.array([0, 0])),
    ],
)
def test_logical_indices_with_tensors(torch_data, sel, expected):
    X, y = torch_data
    data_groups = [0, 1, 0]
    dataset = GroupedDataset(X, y, data_groups=data_groups)
    indices = dataset.logical_indices(sel)
    assert is_numpy(indices)
    assert array_equal(indices, expected)


def test_group_names_with_tensors(torch_data):
    X, y = torch_data
    data_groups = [0, 1, 0]
    group_names = ["group_a", "group_b"]

    dataset = GroupedDataset(X, y, data_groups=data_groups, group_names=group_names)
    assert isinstance(
        dataset.names, np.ndarray
    )  # Group names are always numpy arrays for now
    assert np.array_equal(
        dataset.names, np.array(["group_a", "group_b"], dtype=np.str_)
    )


@pytest.fixture
def larger_torch_dataset(seed):
    X_np, y_np = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        random_state=seed,
    )
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.int64)
    return X, y


def test_grouped_dataset_from_arrays_with_larger_tensor_data(larger_torch_dataset):
    X, y = larger_torch_dataset
    data_groups = y.tolist()  # Use class labels as groups

    train, test = GroupedDataset.from_arrays(
        X, y, data_groups=data_groups, train_size=0.75
    )

    assert torch.is_tensor(train._x)
    assert torch.is_tensor(train._y)
    assert is_numpy(train.data_to_group)
    assert is_numpy(train._indices)

    # Check group indices
    unique_groups = torch.unique(y)
    assert len(train._indices) <= len(unique_groups)

    # Test that the groups are preserved correctly
    for group in train._indices:
        group_idx = group.item()
        group_indices = train.data_indices([group_idx])

        assert is_numpy(group_indices)
        assert np.all(train.data_to_group[group_indices] == group_idx)


def test_grouped_dataset_from_dataset_with_tensors(torch_data):
    X, y = torch_data

    # First create a Dataset
    dataset = Dataset(X, y)

    # Then create a GroupedDataset from it
    data_groups = [0, 1, 0]
    grouped_dataset = GroupedDataset.from_dataset(dataset, data_groups=data_groups)

    assert torch.is_tensor(grouped_dataset._x)
    assert torch.is_tensor(grouped_dataset._y)
    assert is_numpy(grouped_dataset.data_to_group)
    assert is_numpy(grouped_dataset._indices)

    # Check that the data is correct
    assert torch.equal(grouped_dataset._x, X)
    assert torch.equal(grouped_dataset._y, y)

    # Check that the group-to-data mapping is correct
    group0_indices = grouped_dataset.data_indices([0])
    assert array_equal(group0_indices, np.array([0, 2]))


def test_stratified_split_with_tensors(larger_torch_dataset):
    X, y = larger_torch_dataset
    train, test = Dataset.from_arrays(X, y, train_size=0.75, stratify_by_target=True)

    assert torch.is_tensor(train._x)
    assert torch.is_tensor(train._y)

    original_class_counts = [torch.sum(y == label).item() for label in torch.unique(y)]
    train_class_counts = [
        torch.sum(train._y == label).item() for label in torch.unique(y)
    ]

    train_ratio = len(train) / len(y)
    for i, (train_count, orig_count) in enumerate(
        zip(train_class_counts, original_class_counts)
    ):
        expected = orig_count * train_ratio
        assert abs(train_count - expected) / expected < 0.1


def test_grouped_dataset_with_larger_tensor_data(larger_torch_dataset):
    X, y = larger_torch_dataset
    data_groups = y.tolist()
    dataset = GroupedDataset(X, y, data_groups=data_groups)

    assert torch.is_tensor(dataset._x)
    assert torch.is_tensor(dataset._y)

    for group in torch.unique(y):
        group_idx = group.item()
        group_data = dataset.data([group_idx])

        assert torch.all(group_data.y == group_idx)
        assert torch.is_tensor(group_data.x)
        assert torch.is_tensor(group_data.y)


def test_check_x_y_handling(numpy_data, torch_data):
    """Test if Dataset and GroupedDataset correctly handle type conversion."""
    X_np, y_np = numpy_data
    X_tensor, y_tensor = torch_data

    # The implementation uses check_X_y which handles type conversion automatically
    # We should check that the final types are consistent internally

    # When using NumPy arrays
    dataset_np = Dataset(X_np, y_np)
    assert isinstance(dataset_np._x, np.ndarray)
    assert isinstance(dataset_np._y, np.ndarray)

    # When using PyTorch tensors
    dataset_tensor = Dataset(X_tensor, y_tensor)
    assert torch.is_tensor(dataset_tensor._x)
    assert torch.is_tensor(dataset_tensor._y)

    # For GroupedDataset
    data_groups_np = np.array([0, 1, 0])
    dataset_np = GroupedDataset(X_np, y_np, data_groups=data_groups_np)
    assert isinstance(dataset_np._x, np.ndarray)
    assert isinstance(dataset_np._y, np.ndarray)
    assert isinstance(dataset_np.data_to_group, np.ndarray)

    # For GroupedDataset with tensors
    data_groups_tensor = torch.tensor([0, 1, 0])
    dataset_tensor = GroupedDataset(X_tensor, y_tensor, data_groups=data_groups_tensor)
    assert torch.is_tensor(dataset_tensor._x)
    assert torch.is_tensor(dataset_tensor._y)
    assert is_numpy(dataset_tensor.data_to_group)


def test_dataset_tensor_mmap():
    """Test error handling when trying to use mmap with tensors."""
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = torch.tensor([0, 1, 0])

    # Test mmap with tensors - should raise an error
    with pytest.raises(TypeError, match="must be numpy arrays in order to use mmap"):
        Dataset(X, y, mmap=True)


def test_dataset_factory_type_consistency():
    """Test that factory methods maintain type consistency."""
    # Create a larger dataset for more reliable stratification
    X = torch.tensor(np.random.rand(100, 5).astype(np.float32))
    y = torch.tensor(np.random.randint(0, 3, 100).astype(np.int64))

    # Test from_arrays
    train_ds, test_ds = Dataset.from_arrays(X, y, train_size=0.7)
    assert torch.is_tensor(train_ds._x)
    assert torch.is_tensor(train_ds._y)
    assert torch.is_tensor(test_ds._x)
    assert torch.is_tensor(test_ds._y)

    # Test from_arrays with stratification
    train_ds, test_ds = Dataset.from_arrays(
        X, y, train_size=0.7, stratify_by_target=True
    )
    assert torch.is_tensor(train_ds._x)
    assert torch.is_tensor(train_ds._y)

    # Check that all classes appear in both train and test
    unique_classes = torch.unique(y)
    for cls in unique_classes:
        assert torch.any(train_ds._y == cls)
        assert torch.any(test_ds._y == cls)


def test_grouped_dataset_factory_type_consistency():
    """Test that GroupedDataset factory methods maintain type consistency."""
    # Create a larger dataset with tensor data
    X = torch.tensor(np.random.rand(100, 5).astype(np.float32))
    y = torch.tensor(np.random.randint(0, 3, 100).astype(np.int64))
    data_groups = torch.tensor(np.random.randint(0, 5, 100).astype(np.int64))

    # Test from_arrays
    train_ds, test_ds = GroupedDataset.from_arrays(
        X, y, data_groups=data_groups, train_size=0.7
    )
    assert torch.is_tensor(train_ds._x)
    assert torch.is_tensor(train_ds._y)
    assert is_numpy(train_ds.data_to_group)
    assert is_numpy(train_ds._indices)

    # Test from_dataset
    dataset = Dataset(X, y)
    grouped_ds = GroupedDataset.from_dataset(dataset, data_groups=data_groups)
    assert torch.is_tensor(grouped_ds._x)
    assert torch.is_tensor(grouped_ds._y)
    assert is_numpy(grouped_ds.data_to_group)
    assert is_numpy(grouped_ds._indices)


def test_edge_case_empty_groups():
    """Test edge case where a group might have no data points."""
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = torch.tensor([0, 1, 2, 3])

    # Create data groups where one group has no data points
    data_groups = torch.tensor([0, 1, 1, 2])  # Group with id=3 has no points
    dataset = GroupedDataset(X, y, data_groups=data_groups)

    # Verify data_to_group mapping
    assert array_equal(dataset.data_to_group, data_groups.cpu().numpy())

    # Verify group_to_data mapping
    # Group 0 should have one data point (index 0)
    assert len(dataset.group_to_data[0]) == 1
    assert dataset.group_to_data[0][0] == 0

    # Group 1 should have two data points (indices 1, 2)
    assert len(dataset.group_to_data[1]) == 2
    assert sorted(dataset.group_to_data[1]) == [1, 2]

    # Group 2 should have one data point (index 3)
    assert len(dataset.group_to_data[2]) == 1
    assert dataset.group_to_data[2][0] == 3


def test_target_method_with_single_dim_tensor():
    """Test target method with single dimension tensor targets."""
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = torch.tensor([0, 1, 0])

    dataset = Dataset(X, y, target_names=["target"])

    target_idx = dataset.target("target")
    assert isinstance(target_idx, slice)

    target_data = dataset.data().y[target_idx]
    assert torch.is_tensor(target_data)
    assert torch.equal(target_data, y)


def test_target_method_with_multi_dim_tensor():
    """Test target method with multidimensional tensor targets."""
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = torch.tensor([[0, 1], [1, 0], [0, 1]])

    dataset = Dataset(X, y, target_names=["t1", "t2"], multi_output=True)

    t1_slice = dataset.target("t1")
    t2_slice = dataset.target("t2")

    assert isinstance(t1_slice, tuple)
    assert t1_slice[1] == 0

    assert isinstance(t2_slice, tuple)
    assert t2_slice[1] == 1

    t1_data = dataset.data().y[t1_slice]
    t2_data = dataset.data().y[t2_slice]

    assert torch.is_tensor(t1_data)
    assert torch.is_tensor(t2_data)
    assert torch.equal(t1_data, torch.tensor([0, 1, 0]))
    assert torch.equal(t2_data, torch.tensor([1, 0, 1]))


def test_tensor_type_preservation_with_operations():
    """Test that tensor types are preserved through sequences of operations."""
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = torch.tensor([0, 1, 0])
    data_groups = torch.tensor([0, 1, 0])

    dataset = GroupedDataset(X, y, data_groups=data_groups)

    # Test slicing operation
    sliced_dataset = dataset[0]  # Get first group
    assert torch.is_tensor(sliced_dataset._x)
    assert torch.is_tensor(sliced_dataset._y)
    assert is_numpy(sliced_dataset.data_to_group)

    # Test data method
    data = sliced_dataset.data()
    assert torch.is_tensor(data.x)
    assert torch.is_tensor(data.y)

    # Test data_indices method
    indices = sliced_dataset.data_indices(None)
    assert is_numpy(indices)

    # Test logical_indices method
    logical_indices = dataset.logical_indices(indices)
    assert is_numpy(logical_indices)

    # Test multiple slice operations
    multi_slice = dataset[[0, 1]]
    assert torch.is_tensor(multi_slice._x)
    assert torch.is_tensor(multi_slice._y)
    assert is_numpy(multi_slice.data_to_group)

    # Test data access from multi_slice
    multi_data = multi_slice.data()
    assert torch.is_tensor(multi_data.x)
    assert torch.is_tensor(multi_data.y)
    assert multi_data.x.shape[0] == len(X)  # All data points should be in the result

    # For direct data access, use raw indices
    direct_indices = torch.tensor([0, 1])  # Just use first two data points directly
    direct_data = dataset.data(direct_indices)
    assert torch.is_tensor(direct_data.x)
    assert torch.is_tensor(direct_data.y)

    # Test type preservation through operations - ensure every output is a tensor
    for output in [
        sliced_dataset._x,
        sliced_dataset._y,
        data.x,
        data.y,
        multi_slice._x,
        multi_slice._y,
        multi_data.x,
        multi_data.y,
        direct_data.x,
        direct_data.y,
    ]:
        assert torch.is_tensor(output), f"Expected tensor, got {type(output)}"
