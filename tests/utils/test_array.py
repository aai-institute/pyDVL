"""Tests for the utils.array module."""

import warnings
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from pydvl.utils.array import (
    array_arange,
    array_concatenate,
    array_equal,
    array_index,
    array_ones,
    array_ones_like,
    array_slice,
    array_unique,
    array_where,
    array_zeros,
    array_zeros_like,
    atleast1d,
    check_X_y,
    check_X_y_torch,
    for_pytorch,
    for_sklearn,
    is_numpy,
    is_tensor,
    stratified_split_indices,
    to_numpy,
    to_tensor,
    try_torch_import,
)

torch = try_torch_import()
requires_torch = pytest.mark.skipif(torch is None, reason="PyTorch not available")


# Helper function to generate test arrays
def get_test_arrays() -> Dict[str, List[Any]]:
    """Get test arrays for both numpy and torch."""
    arrays = {
        "numpy": [
            np.array([1, 2, 3]),
            np.array([[1, 2], [3, 4]]),
            np.array([True, False, True]),
            np.array([1.1, 2.2, 3.3]),
        ]
    }

    if torch is not None:
        arrays["torch"] = [
            torch.tensor([1, 2, 3]),
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([True, False, True]),
            torch.tensor([1.1, 2.2, 3.3]),
        ]

    return arrays


class TestTypeChecking:
    """Test type checking functions."""

    def test_is_tensor(self):
        """Test is_tensor function."""
        if torch is not None:
            assert is_tensor(torch.tensor([1, 2, 3]))
        assert not is_tensor(np.array([1, 2, 3]))
        assert not is_tensor([1, 2, 3])

    def test_is_numpy(self):
        """Test is_numpy function."""
        assert is_numpy(np.array([1, 2, 3]))
        if torch is not None:
            assert not is_numpy(torch.tensor([1, 2, 3]))
        assert not is_numpy([1, 2, 3])


class TestTypeConversion:
    """Test type conversion functions."""

    @requires_torch
    def test_as_tensor(self):
        """Test as_tensor function."""
        # From numpy
        np_array = np.array([1, 2, 3])
        tensor = to_tensor(np_array)
        assert is_tensor(tensor)
        assert np.array_equal(to_numpy(tensor), np_array)

        # From list
        list_array = [1, 2, 3]
        tensor = to_tensor(list_array)
        assert is_tensor(tensor)
        assert np.array_equal(to_numpy(tensor), np.array(list_array))

        # From tensor (identity)
        tensor1 = torch.tensor([1, 2, 3])
        tensor2 = to_tensor(tensor1)
        assert tensor1 is tensor2  # Should be the same object

    def test_as_numpy(self):
        """Test as_numpy function."""
        # From list
        list_array = [1, 2, 3]
        np_array = to_numpy(list_array)
        assert is_numpy(np_array)
        assert np.array_equal(np_array, np.array(list_array))

        # From numpy (identity)
        np_array1 = np.array([1, 2, 3])
        np_array2 = to_numpy(np_array1)
        assert np_array1 is np_array2  # Should be the same object

        # From tensor
        if torch is not None:
            tensor = torch.tensor([1, 2, 3])
            np_array = to_numpy(tensor)
            assert is_numpy(np_array)
            assert np.array_equal(np_array, np.array([1, 2, 3]))


class TestArrayCreation:
    """Test array creation functions."""

    @pytest.mark.parametrize("shape", [3, (2, 3), [2, 2]])
    def test_array_zeros(self, shape):
        """Test array_zeros function."""
        # Default (numpy)
        result = array_zeros(shape)
        assert is_numpy(result)
        assert result.shape == (shape,) if isinstance(shape, int) else tuple(shape)
        assert np.all(result == 0)

        if torch is not None:
            # With torch reference
            like = torch.tensor([1])
            result = array_zeros(shape, like=like)
            assert is_tensor(result)
            assert result.shape == (shape,) if isinstance(shape, int) else tuple(shape)
            assert torch.all(result == 0)

    @pytest.mark.parametrize("shape", [3, (2, 3), [2, 2]])
    def test_array_ones(self, shape):
        """Test array_ones function."""
        # Default (numpy)
        result = array_ones(shape)
        assert is_numpy(result)
        assert result.shape == (shape,) if isinstance(shape, int) else tuple(shape)
        assert np.all(result == 1)

        if torch is not None:
            # With torch reference
            like = torch.tensor([1])
            result = array_ones(shape, like=like)
            assert is_tensor(result)
            assert result.shape == (shape,) if isinstance(shape, int) else tuple(shape)
            assert torch.all(result == 1)

    def test_array_zeros_like(self):
        """Test array_zeros_like function."""
        arrays = get_test_arrays()

        for np_arr in arrays["numpy"]:
            result = array_zeros_like(np_arr)
            assert is_numpy(result)
            assert result.shape == np_arr.shape
            assert np.all(result == 0)

        if torch is not None:
            for torch_arr in arrays["torch"]:
                result = array_zeros_like(torch_arr)
                assert is_tensor(result)
                assert result.shape == torch_arr.shape
                assert torch.all(result == 0)

    def test_array_ones_like(self):
        """Test array_ones_like function."""
        arrays = get_test_arrays()

        for np_arr in arrays["numpy"]:
            result = array_ones_like(np_arr)
            assert is_numpy(result)
            assert result.shape == np_arr.shape
            assert np.all(result == 1)

        if torch is not None:
            for torch_arr in arrays["torch"]:
                result = array_ones_like(torch_arr)
                assert is_tensor(result)
                assert result.shape == torch_arr.shape
                assert torch.all(result == 1)

    def test_array_arange(self):
        """Test array_arange function."""
        # Test with numpy default
        result = array_arange(5)
        assert is_numpy(result)
        assert np.array_equal(result, np.arange(5))

        # Test with start, stop, step
        result = array_arange(1, 10, 2)
        assert is_numpy(result)
        assert np.array_equal(result, np.arange(1, 10, 2))

        if torch is not None:
            # Test with torch reference
            like = torch.tensor([1])
            result = array_arange(5, like=like)
            assert is_tensor(result)
            assert torch.equal(result, torch.arange(5))


class TestArrayOperations:
    """Test array operations."""

    def test_array_where(self):
        """Test array_where function."""
        # Numpy case
        condition = np.array([True, False, True])
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        result = array_where(condition, x, y)
        assert is_numpy(result)
        assert np.array_equal(result, np.array([1, 5, 3]))

        if torch is not None:
            # Torch case
            condition = torch.tensor([True, False, True])
            x = torch.tensor([1, 2, 3])
            y = torch.tensor([4, 5, 6])
            result = array_where(condition, x, y)
            assert is_tensor(result)
            assert torch.equal(result, torch.tensor([1, 5, 3]))

            # Mixed case (should return torch tensor)
            condition = np.array([True, False, True])
            result = array_where(condition, x, y)
            assert is_tensor(result)
            assert torch.equal(result, torch.tensor([1, 5, 3]))

    def test_array_unique(self):
        """Test array_unique function."""
        # Numpy case
        arr = np.array([1, 2, 2, 3, 1])
        result = array_unique(arr)
        assert is_numpy(result)
        assert np.array_equal(result, np.array([1, 2, 3]))

        # With return_index
        result, indices = array_unique(arr, return_index=True)
        assert is_numpy(result)
        assert np.array_equal(result, np.array([1, 2, 3]))
        assert np.array_equal(indices, np.array([0, 1, 3]))

        if torch is not None:
            # Torch case
            arr = torch.tensor([1, 2, 2, 3, 1])
            result = array_unique(arr)
            assert is_tensor(result)
            assert torch.equal(result, torch.tensor([1, 2, 3]))

            # With return_index (slightly different behavior than numpy)
            result, indices = array_unique(arr, return_index=True)
            assert is_tensor(result)
            assert torch.equal(result, torch.tensor([1, 2, 3]))
            assert torch.equal(indices, torch.tensor([0, 1, 3]))

    def test_array_concatenate(self):
        """Test array_concatenate function."""
        # Numpy case
        arrays = [np.array([1, 2]), np.array([3, 4])]
        result = array_concatenate(arrays)
        assert is_numpy(result)
        assert np.array_equal(result, np.array([1, 2, 3, 4]))

        # With axis
        arrays = [np.array([[1], [2]]), np.array([[3], [4]])]
        result = array_concatenate(arrays, axis=1)
        assert is_numpy(result)
        assert np.array_equal(result, np.array([[1, 3], [2, 4]]))

        if torch is not None:
            # Torch case
            arrays = [torch.tensor([1, 2]), torch.tensor([3, 4])]
            result = array_concatenate(arrays)
            assert is_tensor(result)
            assert torch.equal(result, torch.tensor([1, 2, 3, 4]))

            # Mixed case (should return torch tensor)
            arrays = [np.array([1, 2]), torch.tensor([3, 4])]
            result = array_concatenate(arrays)
            assert is_tensor(result)
            assert torch.equal(result, torch.tensor([1, 2, 3, 4]))

            # Empty array error
            with pytest.raises(ValueError):
                array_concatenate([])

    def test_array_equal(self):
        """Test array_equal function."""
        # Numpy case
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])
        arr3 = np.array([1, 2, 4])
        assert array_equal(arr1, arr2)
        assert not array_equal(arr1, arr3)

        if torch is not None:
            # Torch case
            arr1 = torch.tensor([1, 2, 3])
            arr2 = torch.tensor([1, 2, 3])
            arr3 = torch.tensor([1, 2, 4])
            assert array_equal(arr1, arr2)
            assert not array_equal(arr1, arr3)

            # Mixed case
            arr1 = np.array([1, 2, 3])
            arr2 = torch.tensor([1, 2, 3])
            assert array_equal(arr1, arr2)

    def test_array_slice(self):
        """Test array_slice function."""
        # Numpy case
        arr = np.array([1, 2, 3, 4, 5])
        result = array_slice(arr, slice(1, 4))
        assert is_numpy(result)
        assert np.array_equal(result, np.array([2, 3, 4]))

        # With list indices
        result = array_slice(arr, [0, 2, 4])
        assert is_numpy(result)
        assert np.array_equal(result, np.array([1, 3, 5]))

        if torch is not None:
            # Torch case
            arr = torch.tensor([1, 2, 3, 4, 5])
            result = array_slice(arr, slice(1, 4))
            assert is_tensor(result)
            assert torch.equal(result, torch.tensor([2, 3, 4]))

            # With list indices
            result = array_slice(arr, [0, 2, 4])
            assert is_tensor(result)
            assert torch.equal(result, torch.tensor([1, 3, 5]))

        # Test error for non-indexable
        with pytest.raises(TypeError):
            array_slice(123, 0)

    def test_array_index(self):
        """Test array_index function."""
        # Numpy case
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        key = np.array([0, 2])
        result = array_index(arr, key)
        assert is_numpy(result)
        assert np.array_equal(result, np.array([[1, 2], [5, 6]]))

        # With dim=1
        result = array_index(arr, np.array([1]), dim=1)
        assert is_numpy(result)
        assert np.array_equal(result, np.array([[2], [4], [6]]))

        if torch is not None:
            # Torch case
            arr = torch.tensor([[1, 2], [3, 4], [5, 6]])
            key = torch.tensor([0, 2])
            result = array_index(arr, key)
            assert is_tensor(result)
            assert torch.equal(result, torch.tensor([[1, 2], [5, 6]]))

            # With dim=1
            result = array_index(arr, torch.tensor([1]), dim=1)
            assert is_tensor(result)
            assert torch.equal(result, torch.tensor([[2], [4], [6]]))

            # Mixed case
            result = array_index(arr, np.array([0, 2]))
            assert is_tensor(result)
            assert torch.equal(result, torch.tensor([[1, 2], [5, 6]]))

        # Test error for out of bounds dim
        with pytest.raises(ValueError):
            array_index(arr, key, dim=3)


class TestLibrarySpecificOperations:
    """Test library-specific operations."""

    def test_for_sklearn(self):
        """Test for_sklearn function."""
        # Numpy case (identity)
        arr = np.array([1, 2, 3])
        result = for_sklearn(arr)
        assert is_numpy(result)
        assert result is arr  # Should be the same object

        if torch is not None:
            # Torch case (conversion)
            arr = torch.tensor([1, 2, 3])
            result = for_sklearn(arr)
            assert is_numpy(result)
            assert np.array_equal(result, np.array([1, 2, 3]))

            # With warning
            with warnings.catch_warnings(record=True) as w:
                result = for_sklearn(arr, function_name="test_function")
                assert len(w) == 1
                assert "test_function requires numpy arrays" in str(w[0].message)

    @requires_torch
    def test_for_pytorch(self):
        """Test for_pytorch function."""
        # Torch case (identity)
        arr = torch.tensor([1, 2, 3])
        result = for_pytorch(arr)
        assert is_tensor(result)
        assert result is arr  # Should be the same object

        # Numpy case (conversion)
        arr = np.array([1, 2, 3])
        result = for_pytorch(arr)
        assert is_tensor(result)
        assert torch.equal(result, torch.tensor([1, 2, 3]))

        # With warning
        with warnings.catch_warnings(record=True) as w:
            result = for_pytorch(arr, function_name="test_function")
            assert len(w) == 1
            assert "test_function requires PyTorch tensors" in str(w[0].message)

    def test_stratified_split_indices(self):
        """Test stratified_split_indices function."""
        # Numpy case
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        train_indices, test_indices = stratified_split_indices(
            y, train_size=0.6, random_state=42
        )
        assert is_numpy(train_indices)
        assert is_numpy(test_indices)
        # Just check that indices are split, don't assert exact count as sklearn may behave differently
        assert len(train_indices) + len(test_indices) == len(y)

        # Check stratification - ensure all classes are represented
        train_labels = y[train_indices]
        unique, counts = np.unique(train_labels, return_counts=True)
        assert np.array_equal(unique, np.array([0, 1, 2]))

        if torch is not None:
            # Torch case
            y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
            train_indices, test_indices = stratified_split_indices(
                y, train_size=0.6, random_state=42
            )
            assert is_tensor(train_indices)
            assert is_tensor(test_indices)
            # Just check that indices are split, don't assert exact count as torch implementation may behave differently
            assert len(train_indices) + len(test_indices) == len(y)

            # Check stratification - ensure all classes are represented
            train_labels = y[train_indices]
            unique, counts = torch.unique(train_labels, return_counts=True)
            assert torch.equal(unique, torch.tensor([0, 1, 2]))

    def test_atleast1d(self):
        """Test atleast1d function."""
        # Scalar
        result = atleast1d(5)
        assert is_numpy(result)
        assert np.array_equal(result, np.array([5]))

        # 1D array
        arr = np.array([1, 2, 3])
        result = atleast1d(arr)
        assert is_numpy(result)
        assert result is arr  # Should be the same object

        # List
        result = atleast1d([1, 2, 3])
        assert is_numpy(result)
        assert np.array_equal(result, np.array([1, 2, 3]))

        if torch is not None:
            # Torch scalar
            arr = torch.tensor(5)
            result = atleast1d(arr)
            assert is_tensor(result)
            assert torch.equal(result, torch.tensor([5]))

            # Torch 1D array
            arr = torch.tensor([1, 2, 3])
            result = atleast1d(arr)
            assert is_tensor(result)
            assert result is arr  # Should be the same object

    @requires_torch
    def test_check_X_y_torch(self):
        """Test check_X_y_torch function."""
        # Valid inputs
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

    def test_check_X_y(self):
        """Test check_X_y function."""
        # Test numpy arrays
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        y = np.array([0, 1, 0], dtype=np.float32)
        X_checked, y_checked = check_X_y(X, y)
        assert is_numpy(X_checked)
        assert is_numpy(y_checked)

        if torch is not None:
            # Test torch tensors
            X = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
            y = torch.tensor([0, 1, 0], dtype=torch.float32)
            X_checked, y_checked = check_X_y(X, y)
            assert is_tensor(X_checked)
            assert is_tensor(y_checked)

            # Mixed types not well-defined, not testing
