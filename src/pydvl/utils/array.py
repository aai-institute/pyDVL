"""
This module contains utility functions for working with arrays in a type-agnostic way.
It provides a consistent interface for operations on both NumPy arrays and PyTorch tensors.

The functions in this module are designed to:

1. Detect array types automatically (numpy.ndarray or torch.Tensor)
2. Perform operations using the appropriate library
3. Preserve the input type in the output, except for functions intended to operate on
   indices, which always return NDArrays for convenience.
4. Minimize unnecessary type conversions

??? example "Some examples"

    ```python
    import numpy as np
    import torch
    from pydvl.utils.array import array_concatenate, is_tensor

    # Type checking
    is_tensor(x_torch)  # Returns True
    is_tensor(x_np)  # Returns False

    # Operations preserve types
    result = array_concatenate([x_np, zeros_np])  # Returns numpy.ndarray
    result = array_concatenate([x_torch, zeros_torch])  # Returns torch.Tensor
    ```

The module uses a TypeVar `ArrayT` to ensure type preservation across functions,
allowing for proper static type checking with both array types.
"""

from __future__ import annotations

from numbers import Number
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
    "array_unique",
    "array_concatenate",
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

DT = TypeVar("DT", bound=np.generic)


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


def require_torch() -> ModuleType:
    torch = try_torch_import(require=True)
    assert torch is not None
    return torch


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
    def nbytes(self) -> int: ...

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

    def __array__(self, dtype: DT | None = None) -> NDArray[DT]: ...

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
) -> Tuple[Tensor, NDArray]: ...


def array_unique(
    array: NDArray | Tensor, return_index: bool = False, **kwargs: Any
) -> Union[NDArray | Tensor, Tuple[NDArray | Tensor, NDArray]]:
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
            return cast(Tuple[Tensor, NDArray], (result, indices_tensor.cpu().numpy()))
        return cast(Tensor, result)
    else:  # Fallback to numpy approach.
        numpy_array = to_numpy(array)
        if return_index:
            # np.unique returns a tuple when return_index=True
            unique_vals, indices = np.unique(
                numpy_array,
                return_index=True,
                **{k: v for k, v in kwargs.items() if k != "return_index"},
            )
            return cast(Tuple[NDArray, NDArray], (unique_vals, indices))
        else:
            # Simple case - just unique values
            result = np.unique(
                numpy_array,
                return_index=False,
                **{k: v for k, v in kwargs.items() if k != "return_index"},
            )
            return cast(NDArray, result)


@overload
def array_concatenate(arrays: Sequence[NDArray], axis: int = 0) -> NDArray: ...


@overload
def array_concatenate(arrays: Sequence[Tensor], axis: int = 0) -> Tensor: ...


def array_concatenate(
    arrays: Sequence[NDArray | Tensor], axis: int = 0
) -> NDArray | Tensor:
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
        return cast(Tensor, torch.cat(tensor_arrays, dim=axis))
    # Otherwise, convert all arrays to numpy arrays.
    numpy_arrays = [to_numpy(a) for a in arrays]
    return cast(NDArray, np.concatenate(numpy_arrays, axis=axis))


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
        # Final shuffling to avoid sorting the datasets by class
        train_cat = torch.cat(train_indices)
        test_cat = torch.cat(test_indices)
        return cast(
            Tuple[ArrayT, ArrayT],
            (
                train_cat[torch.randperm(len(train_cat))],
                test_cat[torch.randperm(len(test_cat))],
            ),
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
def atleast1d(a: Number) -> NDArray: ...


@overload
def atleast1d(a: NDArray) -> NDArray: ...


@overload
def atleast1d(a: Tensor) -> Tensor: ...


def atleast1d(a: NDArray | Tensor | Number) -> NDArray | Tensor:
    """Ensures that the input is at least 1D.

    For scalar builtin types, the output is an NDArray. Scalar tensors are converted to
    1D tensors

    Args:
        a: Input array-like object or a scalar.

    Returns:
        The input, as a 1D structure.
    """
    if is_numpy(a) or np.isscalar(a):
        return cast(NDArray, np.atleast_1d(a))  # type: ignore
    if is_tensor(a):
        assert torch is not None
        t = cast(Tensor, a)
        return cast(Tensor, t.unsqueeze(0) if t.ndim == 0 else t)
    raise TypeError(f"Unsupported array or scalar type: {type(a).__name__}")


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
    X: NDArray | Tensor,
    y: NDArray | Tensor,
    *,
    multi_output: bool = False,
    estimator: str | object | None = None,
    copy: bool = False,
) -> Tuple[NDArray | Tensor, NDArray | Tensor]:
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
        nz = torch.nonzero(cast(Tensor, x), as_tuple=True)
        return cast(tuple[NDArray, ...], tuple(t.cpu().numpy() for t in nz))
    return cast(tuple[NDArray, ...], np.nonzero(to_numpy(x)))


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
