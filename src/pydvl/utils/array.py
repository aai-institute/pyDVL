"""
This module contains utility functions for working with arrays in a type-agnostic way.
It provides a consistent interface for operations on both NumPy arrays and PyTorch tensors.

The functions in this module are designed to:
1. Detect array types automatically (numpy.ndarray or torch.Tensor)
2. Perform operations using the appropriate library
3. Preserve the input type in the output
4. Minimize unnecessary type conversions

Usage examples:

```python
import numpy as np
import torch
from pydvl.utils.array import array_zeros, array_concatenate, is_tensor

# Works with NumPy arrays
x_np = np.array([1, 2, 3])
zeros_np = array_zeros((3,), like=x_np)  # Returns numpy.ndarray

# Works with PyTorch tensors
x_torch = torch.tensor([1, 2, 3])
zeros_torch = array_zeros((3,), like=x_torch)  # Returns torch.Tensor

# Type checking
is_tensor(x_torch)  # Returns True
is_tensor(x_np)     # Returns False

# Operations preserve types
result = array_concatenate([x_np, zeros_np])  # Returns numpy.ndarray
result = array_concatenate([x_torch, zeros_torch])  # Returns torch.Tensor
```

The module uses a TypeVar `ArrayT` to ensure type preservation across functions,
allowing for proper static type checking with both array types.
"""

from __future__ import annotations

import warnings
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    List,
    Literal,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
    runtime_checkable,
)

import numpy as np
import sklearn.utils
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Self

__all__ = [
    "is_tensor",
    "is_numpy",
    "to_tensor",
    "to_numpy",
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
    "array_exp",
    "array_count_nonzero",
    "array_nonzero",
    "is_categorical",
    "stratified_split_indices",
    "atleast1d",
    "check_X_y",
    "check_X_y_torch",
    "require_torch",
    "try_torch_import",
]

DT = TypeVar("DT")


def try_torch_import(require: bool = False) -> ModuleType | None:
    """Import torch if available, otherwise return None.
    Args:
        require: If True, raise an ImportError if torch is not available.
    """
    try:
        import torch

        return cast(ModuleType, torch)
    except ImportError as e:
        if require:
            raise ImportError("torch is required but not installed.") from e
        return None


torch = try_torch_import()

# Define Tensor type properly for type checking
if TYPE_CHECKING:
    from torch import Tensor
else:
    # At runtime this is just a reference to the actual class
    Tensor = Any if torch is None else torch.Tensor


def is_tensor(array: Any) -> bool:
    """Check if an array is a PyTorch tensor."""
    return torch is not None and isinstance(array, torch.Tensor)


def is_numpy(array: Any) -> bool:
    """Check if an array is a NumPy ndarray."""
    return isinstance(array, np.ndarray)


@runtime_checkable
class Array(Protocol[DT]):
    """Protocol defining a common interface for NumPy arrays and PyTorch tensors.

    This protocol defines the essential methods and properties required for array-like
    operations in PyDVL. It serves as a structural type for both numpy.ndarray
    and torch.Tensor, enabling type-safe generic functions that work with either type.

    The generic parameter DT represents the data type of the array elements.

    !!! note "Type Preservation"
        Functions that accept Array types will generally preserve the input type
        in their outputs. For example, if you pass a torch.Tensor, you'll get a
        torch.Tensor back; if you pass a numpy.ndarray, you'll get a numpy.ndarray back.

    !!! warning
        This is a "best-effort" implementation that covers the methods and properties
        needed by PyDVL, but it is not a complete representation of all functionality
        in NumPy and PyTorch arrays.
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def ndim(self) -> int: ...

    @property
    def dtype(self) -> Any: ...

    def __len__(self) -> int: ...

    def __getitem__(self, key: Any) -> Self: ...

    def __iter__(self) -> Iterator: ...

    def __add__(self, other: Array) -> Array: ...

    def __sub__(self, other) -> Array: ...

    def __mul__(self, other) -> Array: ...

    def __matmul__(self, other) -> Array: ...

    def __array__(self, dtype: DT | None = None) -> NDArray: ...

    def flatten(self, *args, **kwargs) -> Self: ...

    def reshape(self, *args: Any, **kwargs: Any) -> Self: ...

    def tolist(self) -> list: ...

    def item(self) -> DT: ...

    def sum(self, *args: Any, **kwargs: Any) -> Self: ...


def to_tensor(array: Array | ArrayLike) -> Tensor:
    """
    Convert array to torch.Tensor if it's not already.

    Args:
        array: Input array.

    Returns:
        A torch.Tensor representation of the input.

    Raises:
        ImportError: If PyTorch is not available.
    """
    torch = require_torch()
    if isinstance(array, torch.Tensor):
        return array
    return cast(Tensor, torch.as_tensor(array))


def to_numpy(array: Array | ArrayLike) -> NDArray:
    """
    Convert array to a numpy.ndarray if it's not already.

    Args:
        array: Input array.

    Returns:
        A numpy.ndarray representation of the input.
    """
    if isinstance(array, np.ndarray):
        return array
    if (torch := try_torch_import()) is not None and isinstance(array, torch.Tensor):
        return cast(NDArray, array.cpu().detach().numpy())
    return cast(NDArray, np.asarray(array))


ShapeType = Union[int, Tuple[int, ...], List[int]]


def array_zeros(
    shape: ShapeType,
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
        return cast(Array, np.zeros(shape, dtype=dtype))

    if is_numpy(like):
        return cast(Array, np.zeros(shape, dtype=dtype))
    elif is_tensor(like):
        assert torch is not None
        like_tensor = cast(Tensor, like)
        return cast(Array, torch.zeros(shape, dtype=dtype, device=like_tensor.device))
    else:
        # In case 'like' is an unsupported type, fallback with numpy.
        warnings.warn("Reference object type is not recognized. Falling back to numpy.")
        return cast(Array, np.zeros(shape, dtype=dtype))


def array_ones(
    shape: ShapeType,
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
        return cast(Array, np.ones(shape, dtype=dtype))

    if is_numpy(like):
        return cast(Array, np.ones(shape, dtype=dtype))
    elif is_tensor(like):
        assert torch is not None
        like_tensor = cast(Tensor, like)
        return cast(Array, torch.ones(shape, dtype=dtype, device=like_tensor.device))
    else:
        warnings.warn("Reference object type is not recognized. Falling back to numpy.")
        return cast(Array, np.ones(shape, dtype=dtype))


@overload
def array_zeros_like(array: NDArray, dtype: Any | None = None) -> NDArray: ...


@overload
def array_zeros_like(array: Tensor, dtype: Any | None = None) -> Tensor: ...


def array_zeros_like(array: Array, dtype: Any | None = None) -> Array:
    """
    Create a zero-filled array with the same shape and type as `array`.

    Args:
        array: Reference array (numpy array or torch tensor).
        dtype: Data type (optional).

    Returns:
        An array of zeros matching the input type.
    """
    if is_tensor(array):
        assert torch is not None  # Keep mypy happy
        tensor_array = cast(Tensor, array)
        return cast(Array, torch.zeros_like(tensor_array, dtype=dtype))
    return cast(Array, np.zeros_like(array, dtype=dtype))


@overload
def array_ones_like(array: NDArray, dtype: Any | None = None) -> NDArray: ...


@overload
def array_ones_like(array: Tensor, dtype: Any | None = None) -> Tensor: ...


def array_ones_like(array: Array, dtype: Any | None = None) -> Array:
    """
    Create a one-filled array with the same shape and type as `array`.

    Args:
        array: Reference array (numpy array or torch tensor).
        dtype: Data type (optional).

    Returns:
        An array of ones matching the input type.
    """
    if is_numpy(array):
        return cast(Array, np.ones_like(array, dtype=dtype))
    elif is_tensor(array):
        assert torch is not None
        tensor_array = cast(Tensor, array)
        return cast(Array, torch.ones_like(tensor_array, dtype=dtype))
    raise TypeError(f"Unsupported array type: {type(array).__name__}")


@overload
def array_where(condition: NDArray, x: NDArray, y: NDArray) -> NDArray: ...


@overload
def array_where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor: ...


def array_where(condition: Array, x: Array, y: Array) -> Array:
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
        assert torch is not None
        device = None
        for a in (condition, x, y):
            if is_tensor(a):
                device = cast(Tensor, a).device
                break

        condition_tensor = (
            condition
            if is_tensor(condition)
            else torch.as_tensor(to_numpy(condition), device=device)
        )
        x_tensor = x if is_tensor(x) else torch.as_tensor(to_numpy(x), device=device)
        y_tensor = y if is_tensor(y) else torch.as_tensor(to_numpy(y), device=device)
        return cast(Array, torch.where(condition_tensor, x_tensor, y_tensor))
    else:
        return cast(Array, np.where(condition, x, y))


@overload
def array_unique(
    array: NDArray, return_index: Literal[False] = False, **kwargs: Any
) -> NDArray: ...


@overload
def array_unique(
    array: NDArray, return_index: Literal[True], **kwargs: Any
) -> Tuple[NDArray, NDArray]: ...


@overload
def array_unique(
    array: Tensor, return_index: Literal[False] = False, **kwargs: Any
) -> Tensor: ...


@overload
def array_unique(
    array: Tensor, return_index: Literal[True], **kwargs: Any
) -> Tuple[Tensor, Tensor]: ...


def array_unique(
    array: ArrayT, return_index: bool = False, **kwargs: Any
) -> Union[ArrayT, Tuple[ArrayT, NDArray]]:
    """
    Return the unique elements in an array, optionally with indices of their first
    occurrences.

    Args:
        array: Input array.
        return_index: If True, also return the indices of the unique elements.
        **kwargs: Extra keyword arguments for the underlying unique function.

    Returns:
        A unique set of elements, and optionally the indices (only for numpy arrays;
        for torch tensors indices are computed manually).
    """
    if is_tensor(array):
        assert torch is not None
        tensor_array = cast(Tensor, array)
        result = torch.unique(tensor_array, **kwargs)
        if return_index:
            indices = []
            array_cpu = tensor_array.cpu()
            result_cpu = result.cpu()
            # For each unique value, find its first occurrence.
            for val in result_cpu:
                nz = (array_cpu == val).nonzero()
                indices.append(nz[0].item())
            indices_tensor = torch.tensor(
                indices, dtype=torch.long, device=tensor_array.device
            )
            return cast(Tuple[ArrayT, NDArray], (result, indices_tensor.cpu().numpy()))
        return cast(ArrayT, result)
    else:  # Fallback to numpy approach.
        numpy_array = to_numpy(array)
        if return_index:
            # np.unique returns a tuple when return_index=True
            unique_vals, indices = np.unique(
                numpy_array,
                return_index=True,
                **{k: v for k, v in kwargs.items() if k != "return_index"},
            )
            return cast(Tuple[ArrayT, NDArray], (unique_vals, indices))
        else:
            # Simple case - just unique values
            result = np.unique(
                numpy_array,
                return_index=False,
                **{k: v for k, v in kwargs.items() if k != "return_index"},
            )
            return cast(ArrayT, result)


@overload
def array_concatenate(arrays: Sequence[NDArray], axis: int = 0) -> NDArray: ...


@overload
def array_concatenate(arrays: Sequence[Tensor], axis: int = 0) -> Tensor: ...


def array_concatenate(arrays: Sequence[Array], axis: int = 0) -> Array:
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
        assert torch is not None
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
                tensor_arrays.append(torch.as_tensor(to_numpy(a), device=device))
        return cast(Array, torch.cat(tensor_arrays, dim=axis))
    # Otherwise, convert all arrays to numpy arrays.
    numpy_arrays = [to_numpy(a) for a in arrays]
    return cast(Array, np.concatenate(numpy_arrays, axis=axis))


def array_equal(array1: Array[Any], array2: Array[Any]) -> bool:
    """
    Check if two arrays are element-wise equal.

    Args:
        array1: First array.
        array2: Second array.

    Returns:
        True if arrays are equal, otherwise False.
    """
    if is_numpy(array1) and is_numpy(array2):
        return bool(np.array_equal(array1, array2))
    elif is_tensor(array1) and is_tensor(array2) and torch is not None:
        return bool(torch.equal(cast(Tensor, array1), cast(Tensor, array2)))
    # Fall back to comparing numpy representations.
    return bool(np.array_equal(to_numpy(array1), to_numpy(array2)))


def array_arange(
    start: int,
    stop: int | None = None,
    step: int = 1,
    *,
    dtype: DT | None = None,
    like: Array | None = None,
) -> Array[DT]:
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
            return cast(Array, np.arange(start, stop, step, dtype=like.dtype))
        elif is_tensor(like):
            assert torch is not None
            like_tensor = cast(Tensor, like)
            return cast(
                Array,
                torch.arange(
                    start,
                    stop,
                    step,
                    dtype=like_tensor.dtype,
                    device=like_tensor.device,
                ),
            )
        else:
            warnings.warn(
                "Reference object type not recognized. Falling back to numpy."
            )
    return cast(Array, np.arange(start, stop, step, dtype=dtype))  # type: ignore


@overload
def array_slice(array: NDArray, indices: Any) -> NDArray: ...


@overload
def array_slice(array: Tensor, indices: Any) -> Tensor: ...


def array_slice(array: Array, indices: Any) -> Array:
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
            f"Dimension {dim} is out of bounds for array with shape "
            f"{getattr(array, 'shape', None)}."
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
            else torch.as_tensor(to_numpy(key), device=arr_tensor.device)
        )
        key_long = cast(Tensor, key_tensor).to(torch.long)
        return cast(Array, torch.index_select(arr_tensor, dim, key_long))
    else:
        raise TypeError("Unsupported array type for indexing.")


ArrayT = TypeVar("ArrayT", bound=Array, contravariant=True)
ArrayRetT = TypeVar("ArrayRetT", bound=Array, covariant=True)


def stratified_split_indices(
    y: ArrayT, train_size: float | int = 0.8, random_state: int | None = None
) -> Tuple[ArrayT, ArrayT]:
    """
    Compute stratified train/test split indices based on labels.

    Args:
        y: Labels array (numpy array or torch tensor).
        train_size: Fraction or absolute number of training samples.
        random_state: Random seed for reproducibility.

    Returns:
        A tuple (train_indices, test_indices) matching the type of y.
    """
    if torch is not None and isinstance(y, torch.Tensor):
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
        return cast(
            Tuple[ArrayT, ArrayT], (torch.cat(train_indices), torch.cat(test_indices))
        )
    else:
        from sklearn.model_selection import train_test_split

        y_np = to_numpy(y)
        indices = np.arange(len(y_np))
        train_indices, test_indices = train_test_split(
            indices, train_size=train_size, stratify=y_np, random_state=random_state
        )

        return cast(Tuple[ArrayT, ArrayT], (train_indices, test_indices))


@overload
def atleast1d(a: NDArray) -> NDArray: ...


@overload
def atleast1d(a: Tensor) -> Tensor: ...


def atleast1d(a: Array) -> Array:
    """
    Ensures that the array is at least 1D.

    Args:
        a: Input array-like object.

    Returns:
        The array, as a 1D structure.
    """
    if is_numpy(a):
        return cast(Array, np.atleast_1d(a))
    if is_tensor(a):
        assert torch is not None
        tensor_array = cast(Tensor, a)
        return cast(
            Array, tensor_array.unsqueeze(0) if tensor_array.ndim == 0 else tensor_array
        )
    return cast(Array, np.atleast_1d(np.array(a)))


@overload
def check_X_y(
    X: NDArray,
    y: NDArray,
    *,
    multi_output: bool = False,
    estimator: str | object | None = None,
    copy: bool = False,
) -> Tuple[NDArray, NDArray]: ...


@overload
def check_X_y(
    X: Tensor,
    y: Tensor,
    *,
    multi_output: bool = False,
    estimator: str | object | None = None,
    copy: bool = False,
) -> Tuple[Tensor, Tensor]: ...


def check_X_y(
    X: Array,
    y: Array,
    *,
    multi_output: bool = False,
    estimator: str | object | None = None,
    copy: bool = False,
) -> Tuple[Array, Array]:
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
        assert torch is not None
        return cast(
            Tuple[Array, Array],
            check_X_y_torch(
                cast(Tensor, X),
                cast(Tensor, y),
                multi_output=multi_output,
                estimator=estimator,
                copy=copy,
            ),
        )
    return cast(
        Tuple[Array, Array],
        sklearn.utils.check_X_y(
            X, y, multi_output=multi_output, estimator=estimator, copy=copy
        ),
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
            f"Expected at least 2D input for X, got {X.dim()}D with shape "
            f"{tuple(X.shape)}."
        )

    if multi_output:
        if y.dim() not in (1, 2):
            raise ValueError(
                f"Expected y to be 1D or 2D for multi_output, got {y.dim()}D with "
                f"shape {tuple(y.shape)}."
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
            f"Inconsistent sample sizes: X has {X.size(0)} samples, but y has "
            f"{y.size(0)} samples."
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


def require_torch() -> ModuleType:
    torch = try_torch_import(require=True)
    assert torch is not None
    return torch


def array_exp(
    x: Array,
) -> Array:
    """
    Calculate the exponential of array elements.

    Args:
        x: Input array.

    Returns:
        Exponential of each element in the input array.
    """
    if is_tensor(x):
        assert torch is not None
        return cast(Array, torch.exp(cast(Tensor, x)))
    else:  # Fallback to numpy approach
        return cast(Array, np.exp(to_numpy(x)))


def array_count_nonzero(
    x: Array,
) -> int:
    """
    Count the number of non-zero elements in the array.

    Args:
        x: Input array.

    Returns:
        Number of non-zero elements.
    """
    if is_tensor(x):
        assert torch is not None
        tensor_array = cast(Tensor, x)
        return int(torch.count_nonzero(tensor_array).item())
    else:  # Fallback to numpy approach
        numpy_array = to_numpy(x)
        return int(np.count_nonzero(numpy_array))


def array_nonzero(
    x: ArrayT,
) -> tuple[NDArray[np.int_], ...]:
    """
    Find the indices of non-zero elements.

    Args:
        x: Input array.

    Returns:
        Tuple of arrays, one for each dimension of x,
        containing the indices of the non-zero elements in that dimension.
    """
    if is_tensor(x):
        assert torch is not None
        tensor_array = cast(Tensor, x)
        # torch.nonzero returns a tensor of indices
        indices = torch.nonzero(tensor_array, as_tuple=True)
        return cast(tuple[NDArray, ...], tuple(t.cpu().numpy() for t in indices))
    else:  # Fallback to numpy approach
        numpy_array = to_numpy(x)
        return cast(tuple[NDArray, ...], np.nonzero(numpy_array))


def is_categorical(x: Array[Any]) -> bool:
    """
    Check if an array contains categorical data (suitable for unique labels).

    For numpy arrays, checks if the dtype.kind is in "OSUiub"
    (Object, String, Unicode, Unsigned integer, Signed integer, Boolean).

    For torch tensors, checks if the dtype is an integer or boolean type.

    Args:
        x: Input array to check.

    Returns:
        True if the array contains categorical data, False otherwise.
    """
    if is_tensor(x):
        assert torch is not None
        tensor_array = cast(Tensor, x)
        # Check for integer and boolean dtypes in torch
        return tensor_array.dtype in [
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ]
    elif is_numpy(x):
        numpy_array = cast(NDArray, x)
        # Object, String, Unicode, Unsigned integer, Signed integer, boolean
        return numpy_array.dtype.kind in "OSUiub"
    else:
        # For other array-like objects, assume it's categorical
        # (this will be verified when operations are performed)
        return True
