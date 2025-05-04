"""Tests for the pydvl.utils.array module."""

import numpy as np
import pytest
from numpy.typing import NDArray

from pydvl.utils.array import (
    array_concatenate,
    array_count_nonzero,
    array_nonzero,
    array_unique,
    atleast1d,
    check_X_y,
    check_X_y_torch,
    is_categorical,
    is_numpy,
    is_tensor,
    stratified_split_indices,
    to_numpy,
    to_tensor,
    try_torch_import,
)

torch = try_torch_import()


@pytest.mark.torch
def test_is_tensor():
    """Test is_tensor function."""
    if torch is not None:
        assert is_tensor(torch.tensor([1, 2, 3]))
    assert not is_tensor(np.array([1, 2, 3]))
    assert not is_tensor([1, 2, 3])


def test_is_numpy():
    """Test is_numpy function."""
    assert is_numpy(np.array([1, 2, 3]))
    assert not is_numpy([1, 2, 3])


@pytest.mark.torch
def test_is_numpy_torch():
    """Test is_numpy function with torch tensors."""
    assert not is_numpy(torch.tensor([1, 2, 3]))


@pytest.mark.torch
def test_as_tensor():
    """Test as_tensor function."""
    np_array = np.array([1, 2, 3])
    tensor = to_tensor(np_array)
    assert is_tensor(tensor)
    assert np.array_equal(to_numpy(tensor), np_array)

    list_array = [1, 2, 3]
    tensor = to_tensor(list_array)
    assert is_tensor(tensor)
    assert np.array_equal(to_numpy(tensor), np.array(list_array))

    tensor1 = torch.tensor([1, 2, 3])
    tensor2 = to_tensor(tensor1)
    assert tensor1 is tensor2  # Should be the same object


def test_to_numpy():
    list_array = [1, 2, 3]
    np_array = to_numpy(list_array)
    assert is_numpy(np_array)
    assert np.array_equal(np_array, np.array(list_array))

    np_array1 = np.array([1, 2, 3])
    np_array2 = to_numpy(np_array1)
    assert np_array1 is np_array2


@pytest.mark.torch
def test_to_numpy_torch():
    tensor = torch.tensor([1, 2, 3])
    np_array = to_numpy(tensor)
    assert is_numpy(np_array)
    assert np.array_equal(np_array, np.array([1, 2, 3]))


def test_array_unique():
    arr = np.array([1, 2, 2, 3, 1])
    result = array_unique(arr)
    assert is_numpy(result)
    assert np.array_equal(result, np.array([1, 2, 3]))

    result, indices = array_unique(arr, return_index=True)
    assert is_numpy(result)
    assert np.array_equal(result, np.array([1, 2, 3]))
    assert np.array_equal(indices, np.array([0, 1, 3]))


@pytest.mark.torch
def test_array_unique_torch():
    arr = torch.tensor([1, 2, 2, 3, 1])
    result = array_unique(arr)
    assert is_tensor(result)
    assert torch.equal(result, torch.tensor([1, 2, 3]))

    # With return_index (slightly different behavior than numpy)
    result, indices = array_unique(arr, return_index=True)
    assert is_tensor(result)
    assert torch.equal(result, torch.tensor([1, 2, 3]))
    np.testing.assert_array_equal(indices, np.array([0, 1, 3]))


def test_array_concatenate():
    arrays = [np.array([1, 2]), np.array([3, 4])]
    result = array_concatenate(arrays)
    assert is_numpy(result)
    assert np.array_equal(result, np.array([1, 2, 3, 4]))

    # With axis
    arrays = [np.array([[1], [2]]), np.array([[3], [4]])]
    result = array_concatenate(arrays, axis=1)
    assert is_numpy(result)
    assert np.array_equal(result, np.array([[1, 3], [2, 4]]))

    with pytest.raises(ValueError):
        array_concatenate([])


@pytest.mark.torch
def test_array_concatenate_torch():
    arrays = [torch.tensor([1, 2]), torch.tensor([3, 4])]
    result = array_concatenate(arrays)
    assert is_tensor(result)
    assert torch.equal(result, torch.tensor([1, 2, 3, 4]))

    # With axis
    arrays = [torch.tensor([[1], [2]]), torch.tensor([[3], [4]])]
    result = array_concatenate(arrays, axis=1)
    assert is_tensor(result)
    assert torch.equal(result, torch.tensor([[1, 3], [2, 4]]))

    # Mixed case (should return torch tensor)
    arrays = [np.array([1, 2]), torch.tensor([3, 4])]
    result = array_concatenate(arrays)
    assert is_tensor(result)
    assert torch.equal(result, torch.tensor([1, 2, 3, 4]))

    with pytest.raises(ValueError):
        array_concatenate([])


def test_array_count_nonzero():
    array = np.array([0, 1, 0, 3, 0])
    result = array_count_nonzero(array)
    assert isinstance(result, int)
    assert result == 2

    # Boolean array
    bool_array = np.array([True, False, True])
    result = array_count_nonzero(bool_array)
    assert result == 2


@pytest.mark.torch
def test_array_count_nonzero_torch():
    array = torch.tensor([0, 1, 0, 3, 0])
    result = array_count_nonzero(array)
    assert isinstance(result, int)
    assert result == 2

    # Boolean tensor
    bool_array = torch.tensor([True, False, True])
    result = array_count_nonzero(bool_array)
    assert result == 2


def test_array_nonzero():
    array = np.array([0, 1, 0, 3, 0])
    result = array_nonzero(array)
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert is_numpy(result[0])
    assert np.array_equal(result[0], np.array([1, 3]))

    # 2D array
    array_2d = np.array([[0, 1], [2, 0]])
    result = array_nonzero(array_2d)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert is_numpy(result[0])
    assert is_numpy(result[1])
    assert np.array_equal(result[0], np.array([0, 1]))
    assert np.array_equal(result[1], np.array([1, 0]))


@pytest.mark.torch
def test_array_nonzero_torch():
    array = torch.tensor([0, 1, 0, 3, 0])
    result = array_nonzero(array)
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert is_numpy(result[0])
    np.testing.assert_array_equal(result[0], np.array([1, 3]))

    # 2D tensor
    array_2d = torch.tensor([[0, 1], [2, 0]])
    result = array_nonzero(array_2d)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert is_numpy(result[0])
    assert is_numpy(result[1])
    np.testing.assert_array_equal(result[0], np.array([0, 1]))
    np.testing.assert_array_equal(result[1], np.array([1, 0]))


def test_stratified_split_indices():
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    train_indices, test_indices = stratified_split_indices(
        y, train_size=0.6, random_state=42
    )
    assert is_numpy(train_indices)
    assert is_numpy(test_indices)
    # Just check that indices are split
    assert len(train_indices) + len(test_indices) == len(y)

    # Check stratification - ensure all classes are represented
    train_labels = y[train_indices]
    unique, counts = np.unique(train_labels, return_counts=True)
    assert np.array_equal(unique, np.array([0, 1, 2]))


@pytest.mark.torch
def test_stratified_split_indices_torch():
    y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    train_indices, test_indices = stratified_split_indices(
        y, train_size=0.6, random_state=42
    )
    assert is_tensor(train_indices)
    assert is_tensor(test_indices)
    assert len(train_indices) + len(test_indices) == len(y)

    # Check stratification
    train_labels = y[train_indices]
    unique, counts = torch.unique(train_labels, return_counts=True)
    assert torch.equal(unique, torch.tensor([0, 1, 2]))


def test_atleast1d():
    result = atleast1d(5)
    assert is_numpy(result)
    assert np.array_equal(result, np.array([5]))

    # 1D array
    arr = np.array([1, 2, 3])
    result = atleast1d(arr)
    assert is_numpy(result)
    assert result is arr  # Should be the same object

    with pytest.raises(TypeError, match="Unsupported array or scalar type"):
        _ = atleast1d([1, 2, 3])


@pytest.mark.torch
def test_atleast1d_torch():
    arr = torch.tensor(5)
    result = atleast1d(arr)
    assert is_tensor(result)
    assert torch.equal(result, torch.tensor([5]))

    # Torch 1D array
    arr = torch.tensor([1, 2, 3])
    result = atleast1d(arr)
    assert is_tensor(result)
    assert result is arr  # Should be the same object


@pytest.mark.torch
def test_check_X_y_torch():
    X = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
    y = torch.tensor([0, 1, 0], dtype=torch.float32)
    X_checked, y_checked = check_X_y_torch(X, y)
    assert is_tensor(X_checked)
    assert is_tensor(y_checked)
    assert X_checked is X  # Should be the same object without copy=True
    assert y_checked is y  # Should be the same object without copy=True

    # With copy=True
    X_checked, y_checked = check_X_y_torch(X, y, copy=True)
    assert X_checked is not X
    assert y_checked is not y

    # 1D X (should be converted to 2D)
    X = torch.tensor([1, 2, 3], dtype=torch.float32)
    y = torch.tensor([0, 1, 0], dtype=torch.float32)
    X_checked, y_checked = check_X_y_torch(X, y)
    assert X_checked.dim() == 2
    assert X_checked.shape == (3, 1)

    # Multi-output y
    X = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
    y = torch.tensor([[0, 1], [1, 0], [0, 0]], dtype=torch.float32)
    with pytest.raises(ValueError):
        check_X_y_torch(X, y)  # Should fail with multi_output=False

    X_checked, y_checked = check_X_y_torch(X, y, multi_output=True)
    assert y_checked.dim() == 2
    assert y_checked.shape == (3, 2)

    # Error cases
    # None y
    with pytest.raises(ValueError):
        check_X_y_torch(X, None)

    # Wrong type
    with pytest.raises(TypeError):
        check_X_y_torch(np.array([[1, 2]]), y)

    with pytest.raises(TypeError):
        check_X_y_torch(X, np.array([0]))

    # Dimension errors
    with pytest.raises(ValueError):
        check_X_y_torch(torch.tensor([[[1]]]), y)  # 3D X

    with pytest.raises(ValueError):
        check_X_y_torch(X, torch.tensor([[[0]]]))  # 3D y

    # Different sample sizes
    with pytest.raises(ValueError):
        check_X_y_torch(X, torch.tensor([0, 1]))

    # Empty array
    with pytest.raises(ValueError):
        check_X_y_torch(
            torch.tensor([[], []], dtype=torch.float32),
            torch.tensor([], dtype=torch.float32),
        )

    # Non-finite values
    with pytest.raises(ValueError):
        check_X_y_torch(torch.tensor([[1, float("inf")], [3, 4]]), y)

    with pytest.raises(ValueError):
        check_X_y_torch(X, torch.tensor([0, float("nan"), 1]))


def test_check_X_y():
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    y = np.array([0, 1, 0], dtype=np.float32)
    X_checked, y_checked = check_X_y(X, y)
    assert is_numpy(X_checked)
    assert is_numpy(y_checked)

    if torch is not None:
        X = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
        y = torch.tensor([0, 1, 0], dtype=torch.float32)
        X_checked, y_checked = check_X_y(X, y)
        assert is_tensor(X_checked)
        assert is_tensor(y_checked)


@pytest.mark.parametrize(
    "array,expected",
    [
        (np.array(["a", "b", "c"], dtype=object), True),
        (np.array(["a", "b", "c"], dtype=str), True),
        (np.array([1, 2, 3], dtype=np.uint8), True),
        (np.array([1, 2, 3], dtype=np.uint32), True),
        (np.array([1, 2, 3], dtype=np.int8), True),
        (np.array([1, 2, 3], dtype=np.int32), True),
        (np.array([True, False, True], dtype=bool), True),
        (np.array([1.0, 2.0, 3.0], dtype=np.float32), False),
        (np.array([1.0, 2.0, 3.0], dtype=np.float64), False),
        (np.array([1 + 2j, 3 + 4j], dtype=complex), False),
    ],
)
def test_is_categorical_numpy(array: NDArray, expected):
    assert is_categorical(array) == expected


@pytest.mark.torch
def test_is_categorical_torch():
    # Boolean tensors
    assert is_categorical(torch.tensor([True, False, True], dtype=torch.bool))

    # Integer tensors
    assert is_categorical(torch.tensor([1, 2, 3], dtype=torch.uint8))
    assert is_categorical(torch.tensor([1, 2, 3], dtype=torch.int8))
    assert is_categorical(torch.tensor([1, 2, 3], dtype=torch.int16))
    assert is_categorical(torch.tensor([1, 2, 3], dtype=torch.int32))
    assert is_categorical(torch.tensor([1, 2, 3], dtype=torch.int64))

    # Non-categorical types
    assert not is_categorical(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
    assert not is_categorical(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))

    # Complex tensors (if available)
    if hasattr(torch, "complex64"):
        assert not is_categorical(torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex64))
    if hasattr(torch, "complex128"):
        assert not is_categorical(
            torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex128)
        )


def test_is_categorical_other():
    """Test is_categorical function with other array-like objects."""
    assert is_categorical([1, 2, 3])
    assert is_categorical((1, 2, 3))
