import numpy as np
import pytest
from sklearn.datasets import make_classification

from pydvl.utils.types import try_torch_import
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


def test_rawdata_type_mismatch(numpy_data, torch_data):
    X_np, _ = numpy_data
    _, y_tensor = torch_data
    with pytest.raises(TypeError, match="must be arrays of the same type"):
        RawData(X_np, y_tensor)


def test_dataset_creation_with_tensors(torch_data):
    X, y = torch_data
    dataset = Dataset(X, y)
    assert torch.is_tensor(dataset._x)
    assert torch.is_tensor(dataset._y)
    assert torch.equal(dataset._x, X)
    assert torch.equal(dataset._y, y)


def test_dataset_data_method_with_tensors(torch_data):
    X, y = torch_data
    dataset = Dataset(X, y)
    data = dataset.data()
    assert torch.is_tensor(data.x)
    assert torch.is_tensor(data.y)
    assert torch.equal(data.x, X)
    assert torch.equal(data.y, y)


def test_dataset_slicing_with_tensors(torch_data):
    X, y = torch_data
    dataset = Dataset(X, y)
    sliced_dataset = dataset[0]
    assert torch.is_tensor(sliced_dataset._x)
    assert torch.is_tensor(sliced_dataset._y)
    assert sliced_dataset._x.shape[0] == 1

    sliced_dataset = dataset[1:3]
    assert torch.is_tensor(sliced_dataset._x)
    assert torch.is_tensor(sliced_dataset._y)
    assert sliced_dataset._x.shape[0] == 2


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
    X = torch.tensor([
        [1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], 
        [11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0]
    ])
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
        X, y_2d, feature_names=["feature1", "feature2"], 
        target_names=["target1", "target2"], multi_output=True
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


def test_grouped_dataset_creation_with_tensors(torch_data):
    X, y = torch_data
    data_groups = [0, 1, 0]
    dataset = GroupedDataset(X, y, data_groups=data_groups)
    assert torch.is_tensor(dataset._x)
    assert torch.is_tensor(dataset._y)
    assert torch.equal(dataset._x, X)
    assert torch.equal(dataset._y, y)
    assert torch.is_tensor(dataset.data_to_group)
    assert torch.is_tensor(dataset._indices)


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
    assert torch.is_tensor(train.data_to_group)
    assert torch.is_tensor(train._indices)


def test_grouped_dataset_slicing_with_tensors(torch_data):
    X, y = torch_data
    data_groups = [0, 1, 0]
    dataset = GroupedDataset(X, y, data_groups=data_groups)

    sliced_dataset = dataset[0]
    assert torch.is_tensor(sliced_dataset._x)
    assert torch.is_tensor(sliced_dataset._y)
    assert sliced_dataset._x.shape[0] == 2
    assert torch.is_tensor(sliced_dataset.data_to_group)
    assert torch.is_tensor(sliced_dataset._indices)

    sliced_dataset = dataset[[0, 1]]
    assert torch.is_tensor(sliced_dataset._x)
    assert torch.is_tensor(sliced_dataset._y)
    assert sliced_dataset._x.shape[0] == 3
    assert torch.is_tensor(sliced_dataset.data_to_group)
    assert torch.is_tensor(sliced_dataset._indices)


def test_grouped_dataset_with_tensor_data_groups(torch_data):
    X, y = torch_data
    data_groups_tensor = torch.tensor([0, 1, 0])
    dataset = GroupedDataset(X, y, data_groups=data_groups_tensor)
    assert torch.is_tensor(dataset._x)
    assert torch.is_tensor(dataset._y)
    assert torch.is_tensor(dataset.data_to_group)
    assert torch.is_tensor(dataset._indices)
    
    # Test data indices method
    indices = dataset.data_indices([0])
    assert torch.is_tensor(indices)
    assert torch.equal(indices, torch.tensor([0, 2], dtype=torch.int64))


def test_data_indices_with_tensors(torch_data):
    X, y = torch_data
    data_groups = [0, 1, 0]
    dataset = GroupedDataset(X, y, data_groups=data_groups)
    
    # Test for a single group
    indices = dataset.data_indices([0])
    assert torch.is_tensor(indices)
    assert indices.shape[0] == 2
    assert torch.equal(indices, torch.tensor([0, 2], dtype=torch.int64))
    
    # Test for multiple groups
    indices = dataset.data_indices([0, 1])
    assert torch.is_tensor(indices)
    assert indices.shape[0] == 3
    assert torch.equal(indices, torch.tensor([0, 2, 1], dtype=torch.int64))
    
    # Test with None
    indices = dataset.data_indices(None)
    assert torch.is_tensor(indices)
    assert indices.shape[0] == 3
    
    # Test with slice
    indices = dataset.data_indices(slice(0, 1))
    assert torch.is_tensor(indices)
    assert indices.shape[0] == 2


def test_logical_indices_with_tensors(torch_data):
    X, y = torch_data
    data_groups = [0, 1, 0]
    dataset = GroupedDataset(X, y, data_groups=data_groups)
    
    # Test return all
    indices = dataset.logical_indices(None)
    assert torch.is_tensor(indices)
    assert torch.equal(indices, torch.tensor([0, 1, 0], dtype=torch.int64))
    
    # Test return subset
    indices = dataset.logical_indices([0, 2])
    assert torch.is_tensor(indices)
    assert torch.equal(indices, torch.tensor([0, 0], dtype=torch.int64))
    
    # Test with tensor indices
    tensor_indices = torch.tensor([0, 2])
    indices = dataset.logical_indices(tensor_indices)
    assert torch.is_tensor(indices)
    assert torch.equal(indices, torch.tensor([0, 0], dtype=torch.int64))


def test_group_names_with_tensors(torch_data):
    X, y = torch_data
    data_groups = [0, 1, 0]
    group_names = ["group_a", "group_b"]
    
    dataset = GroupedDataset(X, y, data_groups=data_groups, group_names=group_names)
    assert isinstance(dataset.names, np.ndarray)  # Group names are always numpy arrays for now
    assert np.array_equal(dataset.names, np.array(["group_a", "group_b"], dtype=np.str_))


@pytest.fixture
def larger_torch_dataset():
    X_np, y_np = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        random_state=42,
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
    assert torch.is_tensor(train.data_to_group)
    assert torch.is_tensor(train._indices)
    
    # Check group indices
    unique_groups = torch.unique(y)
    assert len(train._indices) <= len(unique_groups)
    
    # Test that the groups are preserved correctly
    for group in train._indices:
        group_idx = group.item()
        group_data = train.data([group_idx])
        group_indices = train.data_indices([group_idx])
        
        assert torch.is_tensor(group_indices)
        assert torch.all(train.data_to_group[group_indices] == group_idx)


def test_grouped_dataset_from_dataset_with_tensors(torch_data):
    X, y = torch_data
    
    # First create a Dataset
    dataset = Dataset(X, y)
    
    # Then create a GroupedDataset from it
    data_groups = [0, 1, 0]
    grouped_dataset = GroupedDataset.from_dataset(dataset, data_groups=data_groups)
    
    assert torch.is_tensor(grouped_dataset._x)
    assert torch.is_tensor(grouped_dataset._y)
    assert torch.is_tensor(grouped_dataset.data_to_group)
    assert torch.is_tensor(grouped_dataset._indices)
    
    # Check that the data is correct
    assert torch.equal(grouped_dataset._x, X)
    assert torch.equal(grouped_dataset._y, y)
    
    # Check that the group-to-data mapping is correct
    group0_indices = grouped_dataset.data_indices([0])
    assert torch.is_tensor(group0_indices)
    assert torch.equal(group0_indices, torch.tensor([0, 2], dtype=torch.int64))


@pytest.fixture
def larger_torch_dataset():
    X_np, y_np = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        random_state=42,
    )
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.int64)
    return X, y


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
