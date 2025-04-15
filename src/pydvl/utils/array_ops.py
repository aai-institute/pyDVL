"""
This module contains utility functions for working with arrays in a type-agnostic way.
It supports both NumPy arrays and PyTorch tensors with consistent interfaces.
"""

from __future__ import annotations

import warnings
from typing import Any, List, Sequence, Tuple, Union, cast

import numpy as np
import sklearn.utils
from numpy.typing import ArrayLike, NDArray

from pydvl.utils.types import Array, ArrayT, require_torch, try_torch_import

__all__ = [
    "is_tensor",
    "is_numpy",
    "as_tensor",
    "as_numpy",
    "array_zeros",
    "array_ones",
    "array_zeros_like",
    "array_ones_like",
    "array_where",
    "array_unique",
    "array_concatenate",
    "array_equal",
    "array_arange",
    "array_slice",
    "array_index",
    "stratified_split_indices",
    "for_sklearn",
    "for_pytorch",
    "atleast1d",
    "check_X_y",
    "check_X_y_torch",
]

torch = try_torch_import()
Tensor = None if torch is None else torch.Tensor


def is_tensor(array: Any) -> bool:
    """Check if an array is a PyTorch tensor."""
    return torch is not None and isinstance(array, torch.Tensor)


def is_numpy(array: Any) -> bool:
    """Check if an array is a NumPy ndarray."""
    return isinstance(array, np.ndarray)


def as_tensor(array: Array | ArrayLike, device: Any | None = None) -> Tensor:
    """
    Convert array to torch.Tensor if it's not already.

    Args:
        array: Input array.
        device: Optional device to place the tensor on.

    Returns:
        A torch.Tensor representation of the input.

    Raises:
        ImportError: If PyTorch is not available.
    """
    if torch is None:
        raise ImportError("PyTorch is not available")
    if isinstance(array, torch.Tensor):
        if device is not None and array.device != device:
            return array.to(device)
        return array
    # Prefer as_tensor to preserve memory layout if possible.
    try:
        return torch.as_tensor(array, device=device)
    except Exception:
        # Fallback to torch.tensor in case the input is not directly convertible by as_tensor.
        return torch.tensor(array, device=device)


def as_numpy(array: Array | ArrayLike) -> NDArray:
    """
    Convert array to a numpy.ndarray if it's not already.

    Args:
        array: Input array.

    Returns:
        A numpy.ndarray representation of the input.
    """
    if isinstance(array, np.ndarray):
        return array
    if is_tensor(array):
        return array.cpu().detach().numpy()
    return np.asarray(array)


def array_zeros(
    shape: Union[int, Tuple[int, ...], List[int]],
    *,
    dtype: Any | None = None,
    like: Array | None = None,
) -> Array:
    """
    Create a zero-filled array with the same type as `like`.

    Args:
        shape: Desired shape for the new array.
        dtype: Data type (optional).
        like: Reference array (numpy array or torch tensor).

    Returns:
        An array of zeros (np.ndarray or torch.Tensor).
    """
    if like is None:
        return np.zeros(shape, dtype=dtype)

    if is_numpy(like):
        return np.zeros(shape, dtype=dtype)
    elif is_tensor(like):
        assert torch is not None
        like_tensor = cast(Tensor, like)
        return torch.zeros(shape, dtype=dtype, device=like_tensor.device)
    else:
        # In case 'like' is an unsupported type, fallback with numpy.
        warnings.warn("Reference object type is not recognized. Falling back to numpy.")
        return np.zeros(shape, dtype=dtype)


def array_ones(
    shape: Union[int, Tuple[int, ...], List[int]],
    *,
    dtype: Any | None = None,
    like: Array | None = None,
) -> Array:
    """
    Create a one-filled array with the same type as `like`.

    Args:
        shape: Desired shape for the new array.
        dtype: Data type (optional).
        like: Reference array (numpy array or torch tensor).

    Returns:
        An array of ones (np.ndarray or torch.Tensor).
    """
    if like is None:
        return np.ones(shape, dtype=dtype)

    if is_numpy(like):
        return np.ones(shape, dtype=dtype)
    elif is_tensor(like):
        assert torch is not None
        like_tensor = cast(Tensor, like)
        return torch.ones(shape, dtype=dtype, device=like_tensor.device)
    else:
        warnings.warn("Reference object type is not recognized. Falling back to numpy.")
        return np.ones(shape, dtype=dtype)


def array_zeros_like(array: ArrayT, dtype: Any | None = None) -> ArrayT:
    """
    Create a zero-filled array with the same shape and type as `array`.

    Args:
        array: Reference array (numpy array or torch tensor).
        dtype: Data type (optional).

    Returns:
        An array of zeros matching the input type.
    """
    if is_tensor(array):
        return cast(Tensor, torch.zeros_like(array, dtype=dtype))
    return cast(NDArray, np.zeros_like(array, dtype=dtype))


def array_ones_like(array: ArrayT, dtype: Any | None = None) -> ArrayT:
    """
    Create a one-filled array with the same shape and type as `array`.

    Args:
        array: Reference array (numpy array or torch tensor).
        dtype: Data type (optional).

    Returns:
        An array of ones matching the input type.
    """
    if is_numpy(array):
        return cast(NDArray, np.ones_like(array, dtype=dtype))
    elif is_tensor(array):
        assert torch is not None
        return cast(torch.Tensor, torch.ones_like(array, dtype=dtype))
    raise TypeError(f"Unsupported array type: {type(array).__name__}")


def array_where(condition: ArrayT, x: ArrayT, y: ArrayT) -> ArrayT:
    """
    Return elements chosen from x or y depending on condition.

    Args:
        condition: Boolean mask.
        x: Values selected where condition is True.
        y: Values selected where condition is False.

    Returns:
        An array with elements from x or y, following the input type.
    """
    # If any of the inputs is a tensor and torch is available, work with torch.
    if any(is_tensor(a) for a in (condition, x, y)):
        device = None
        for a in (condition, x, y):
            if is_tensor(a):
                device = cast(Tensor, a).device
                break

        condition_tensor = (
            condition
            if is_tensor(condition)
            else torch.as_tensor(as_numpy(condition), device=device)
        )
        x_tensor = x if is_tensor(x) else torch.as_tensor(as_numpy(x), device=device)
        y_tensor = y if is_tensor(y) else torch.as_tensor(as_numpy(y), device=device)
        return torch.where(condition_tensor, x_tensor, y_tensor)
    else:
        return np.where(condition, x, y)


def array_unique(
    array: ArrayT, return_index: bool = False, **kwargs: Any
) -> Union[ArrayT, Tuple[ArrayT, NDArray]]:
    """
    Return the unique elements in an array, optionally with indices of their first occurrences.

    Args:
        array: Input array.
        return_index: If True, also return the indices of the unique elements.
        **kwargs: Extra keyword arguments for the underlying unique function.

    Returns:
        A unique set of elements, and optionally the indices (only for numpy arrays;
        for torch tensors indices are computed manually).
    """
    if is_tensor(array):
        result = torch.unique(array, **kwargs)
        if return_index:
            indices = []
            array_cpu = array.cpu()
            result_cpu = result.cpu()
            # For each unique value, find its first occurrence.
            for val in result_cpu:
                nz = (array_cpu == val).nonzero()
                indices.append(nz[0].item())
            indices_tensor = torch.tensor(
                indices, dtype=torch.long, device=array.device
            )
            return cast(Tensor, result), indices_tensor
        return cast(Tensor, result)
    else:  # Fallback to numpy approach.
        return cast(
            NDArray, np.unique(as_numpy(array), return_index=return_index, **kwargs)
        )


def array_concatenate(arrays: Sequence[ArrayT], axis: int = 0) -> ArrayT:
    """
    Join a sequence of arrays along an existing axis.

    Args:
        arrays: Sequence of arrays.
        axis: Axis along which to concatenate.

    Returns:
        Concatenated array of the same type as the inputs.

    Raises:
        ValueError: If the input list is empty.
    """
    if not arrays:
        raise ValueError("Cannot concatenate an empty list of arrays.")
    # If any array is a torch tensor, convert all arrays to tensors.
    if any(is_tensor(a) for a in arrays):
        tensor_arrays = []
        device = None
        for a in arrays:
            if is_tensor(a):
                device = cast(Tensor, a).device
                break
        for a in arrays:
            if is_tensor(a):
                tensor_arrays.append(a)
            else:
                tensor_arrays.append(torch.as_tensor(as_numpy(a), device=device))
        return torch.cat(tensor_arrays, dim=axis)
    # Otherwise, convert all arrays to numpy arrays.
    numpy_arrays = [as_numpy(a) for a in arrays]
    return np.concatenate(numpy_arrays, axis=axis)


def array_equal(array1: Array, array2: Array) -> bool:
    """
    Check if two arrays are element-wise equal.

    Args:
        array1: First array.
        array2: Second array.

    Returns:
        True if arrays are equal, otherwise False.
    """
    if is_numpy(array1) and is_numpy(array2):
        return np.array_equal(array1, array2)
    elif is_tensor(array1) and is_tensor(array2):
        return torch.equal(cast(Tensor, array1), cast(Tensor, array2))
    # Fall back to comparing numpy representations.
    return np.array_equal(as_numpy(array1), as_numpy(array2))


def array_arange(
    start: int,
    stop: int | None = None,
    step: int = 1,
    *,
    dtype: Any | None = None,
    like: Array | None = None,
) -> Array:
    """
    Create an array with evenly spaced values within a given interval.

    Args:
        start: Start of interval, or stop if stop is None.
        stop: End of interval (exclusive).
        step: Step size.
        dtype: Data type (optional).
        like: Reference array to infer the type and device.

    Returns:
        An array (numpy.ndarray or torch.Tensor) of the specified range.
    """
    if stop is None:
        start, stop = 0, start

    # If a reference is provided and is recognized as numpy or tensor, use it.
    if like is not None:
        if is_numpy(like):
            return np.arange(start, stop, step, dtype=dtype)
        elif is_tensor(like):
            assert torch is not None
            like_tensor = cast(Tensor, like)
            return torch.arange(
                start, stop, step, dtype=dtype, device=like_tensor.device
            )
        else:
            warnings.warn(
                "Reference object type not recognized. Falling back to numpy."
            )
    return np.arange(start, stop, step, dtype=dtype)


def array_slice(array: ArrayT, indices: Any) -> ArrayT:
    """
    Slice an array in a type-agnostic way.

    Args:
        array: The array to be sliced.
        indices: The slicing indices (int, slice, list, etc.).

    Returns:
        A sliced array with the same type as the input.

    Raises:
        TypeError: If the input array does not support indexing.
    """
    if not hasattr(array, "__getitem__"):
        raise TypeError(
            f"Provided object of type {type(array).__name__} is not indexable."
        )
    return array[indices]


def array_index(array: Array, key: Array, dim: int = 0) -> Array:
    """
    Index into an array along the specified dimension.

    Args:
        array: The input array.
        key: The indices to select.
        dim: Dimension along which to index.

    Returns:
        An array of the same type as the input with the specified indexing applied.

    Raises:
        ValueError: If the dimension is out of bounds.
    """
    # Verify that dim is within valid range.
    if not (
        0 <= dim < array.shape[0]
        if hasattr(array, "shape") and len(array.shape) > 0
        else 0
    ):
        raise ValueError(
            f"Dimension {dim} is out of bounds for array with shape {getattr(array, 'shape', None)}."
        )

    if is_numpy(array):
        # Handle indexing along specified dimension.
        if dim == 0:
            return array[key]
        elif dim == 1:
            return array[:, key]
        else:
            idx = tuple(slice(None) if i != dim else key for i in range(array.ndim))
            return array[idx]
    elif is_tensor(array) and torch is not None:
        arr_tensor = cast(Tensor, array)
        # Ensure key is a tensor.
        key_tensor = (
            key
            if is_tensor(key)
            else torch.as_tensor(as_numpy(key), device=arr_tensor.device)
        )
        return torch.index_select(arr_tensor, dim, key_tensor.long())
    else:
        raise TypeError("Unsupported array type for indexing.")


def for_sklearn(array: Array, function_name: str | None = None) -> NDArray:
    """
    Convert array to numpy format for scikit-learn compatibility.

    Args:
        array: The input array.
        function_name: Optional name of the calling function (for warning message).

    Returns:
        Numpy array.
    """
    if is_tensor(array):
        if function_name:
            warnings.warn(
                f"{function_name} requires numpy arrays. Converting tensor to "
                "numpy, which may impact performance."
            )
        return as_numpy(array)
    return cast(NDArray, array)


def for_pytorch(
    array: Array, device: Any | None = None, function_name: str | None = None
) -> Tensor:
    """
    Convert array to torch.Tensor for PyTorch compatibility.

    Args:
        array: The input array.
        device: Optional device to place the tensor on.
        function_name: Optional name of the calling function (for warning message).

    Returns:
        A torch.Tensor.

    Raises:
        ImportError: If PyTorch is not available.
    """
    if torch is None:
        raise ImportError("PyTorch is not available")
    if is_numpy(array):
        if function_name:
            warnings.warn(
                f"{function_name} requires PyTorch tensors. Converting numpy "
                "array to tensor, which may impact performance."
            )
        return as_tensor(array, device=device)
    tensor = cast(Tensor, array)
    if device is not None and tensor.device != device:
        return tensor.to(device)
    return tensor


def stratified_split_indices(
    y: Array, train_size: float | int = 0.8, random_state: int | None = None
) -> Tuple[Array, Array]:
    """
    Compute stratified train/test split indices based on labels.

    Args:
        y: Labels array (numpy array or torch tensor).
        train_size: Fraction or absolute number of training samples.
        random_state: Random seed for reproducibility.

    Returns:
        A tuple (train_indices, test_indices) matching the type of y.
    """
    # Torch branch
    if is_tensor(y) and torch is not None:
        if random_state is not None:
            torch.manual_seed(random_state)
        n_samples = len(y)
        train_size_int = (
            int(train_size * n_samples) if isinstance(train_size, float) else train_size
        )
        classes = torch.unique(y)
        class_indices = [torch.where(y == c)[0] for c in classes]
        train_indices = []
        test_indices = []
        for idx in class_indices:
            shuffled = idx[torch.randperm(len(idx))]
            n_train = int(len(idx) * (train_size_int / n_samples))
            train_indices.append(shuffled[:n_train])
            test_indices.append(shuffled[n_train:])
        return torch.cat(train_indices), torch.cat(test_indices)
    else:
        # Numpy branch.
        from sklearn.model_selection import train_test_split

        y_np = as_numpy(y)
        indices = np.arange(len(y_np))
        train_indices, test_indices = train_test_split(
            indices, train_size=train_size, stratify=y_np, random_state=random_state
        )
        # If the original was a tensor, convert back.
        if is_tensor(y) and torch is not None:
            return (
                torch.as_tensor(train_indices, device=for_pytorch(y).device),
                torch.as_tensor(test_indices, device=for_pytorch(y).device),
            )
        return train_indices, test_indices


def atleast1d(a: ArrayT | ArrayLike) -> ArrayT:
    """
    Ensures that the array is at least 1D.

    Args:
        a: Input array-like object.

    Returns:
        The array, as a 1D structure.
    """
    if is_numpy(a):
        return cast(NDArray, np.atleast_1d(a))
    if is_tensor(a):
        return a.unsqueeze(0) if a.ndim == 0 else a
    return cast(NDArray, np.atleast_1d(np.array(a)))


def check_X_y(
    X: ArrayT,
    y: ArrayT,
    *,
    multi_output: bool = False,
    estimator: str | object | None = None,
    copy: bool = False,
):
    """
    Validate X and y mimicking the functionality of sklearn's check_X_y.

    For torch tensors, delegates to check_X_y_torch.

    Args:
        X: Input data (at least 2D).
        y: Target values (1D for single-output or 2D for multi-output if enabled).
        multi_output: Whether multi-output targets are allowed.
        estimator: The name or instance of the estimator (used in error messages).
        copy: If True, a copy of the arrays is made.

    Returns:
        A tuple (X_converted, y_converted).

    Raises:
        ValueError or TypeError if the inputs do not validate.
    """
    if is_tensor(X) and is_tensor(y):
        return check_X_y_torch(
            cast(Tensor, X),
            cast(Tensor, y),
            multi_output=multi_output,
            estimator=estimator,
            copy=copy,
        )
    return sklearn.utils.check_X_y(
        X, y, multi_output=multi_output, estimator=estimator, copy=copy
    )


def check_X_y_torch(
    X: Tensor,
    y: Tensor,
    *,
    multi_output: bool = False,
    estimator: str | object | None = None,
    copy: bool = False,
):
    """
    Validate torch tensors X and y similarly to sklearn's check_X_y.

    Args:
        X: Input tensor (at least 2D).
        y: Target tensor (1D for single-output or 2D for multi-output if allowed).
        multi_output: Whether multi-output targets are allowed.
        estimator: Estimator name or instance (used in error messages).
        copy: If True, clones the inputs.

    Returns:
        A tuple (X_converted, y_converted).

    Raises:
        ValueError or TypeError if the inputs are invalid.
    """
    torch = require_torch()

    estimator_name = (
        estimator.__class__.__name__
        if estimator is not None and not isinstance(estimator, str)
        else estimator or "check_X_y"
    )

    if y is None:
        raise ValueError(f"{estimator_name} requires y to be passed, but y is None.")

    if not isinstance(X, torch.Tensor):
        raise TypeError(f"Expected X to be a torch.Tensor, got {type(X).__name__}.")
    if not isinstance(y, torch.Tensor):
        raise TypeError(f"Expected y to be a torch.Tensor, got {type(y).__name__}.")

    if copy:
        X = X.clone()
        y = y.clone()

    if X.dim() == 1:
        X = X.unsqueeze(1)

    if X.dim() < 2:
        raise ValueError(
            f"Expected at least 2D input for X, got {X.dim()}D with shape {tuple(X.shape)}."
        )

    if multi_output:
        if y.dim() not in (1, 2):
            raise ValueError(
                f"Expected y to be 1D or 2D for multi_output, got {y.dim()}D with shape {tuple(y.shape)}."
            )
    else:
        if y.dim() == 2 and y.size(1) == 1:
            y = y.view(-1)
        elif y.dim() != 1:
            raise ValueError(
                f"Expected 1D input for y, got {y.dim()}D with shape {tuple(y.shape)}."
            )

    if X.size(0) != y.size(0):
        raise ValueError(
            f"Inconsistent sample sizes: X has {X.size(0)} samples, but y has {y.size(0)} samples."
        )

    if X.size(0) < 1:
        raise ValueError(f"Found an empty array with shape {tuple(X.shape)}.")

    if not torch.all(torch.isfinite(X)):
        raise ValueError(
            f"Input X in {estimator_name} contains non-finite values (NaN or Inf)."
        )
    if not torch.all(torch.isfinite(y)):
        raise ValueError(
            f"Input y in {estimator_name} contains non-finite values (NaN or Inf)."
        )

    return X, y
