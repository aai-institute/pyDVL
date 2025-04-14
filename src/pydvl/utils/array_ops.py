"""
This module contains utility functions for working with arrays in a type-agnostic way.
It supports both NumPy arrays and PyTorch tensors with consistent interfaces.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union, cast, overload

import numpy as np
from numpy.typing import NDArray

from pydvl.utils.types import Array, try_torch_import

# Import torch if available
torch = try_torch_import()
Tensor = None if torch is None else torch.Tensor

# Type variables for return type matching
T = TypeVar("T", bound=Array)
ShapeType = Union[int, Tuple[int, ...], List[int]]
DType = Any  # Could be np.dtype or torch.dtype
Device = Any  # Could be str or torch.device

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
    "for_sklearn",
    "for_pytorch",
]


def is_tensor(array: Any) -> bool:
    """Check if an array is a PyTorch tensor.
    
    Args:
        array: Array to check
        
    Returns:
        bool: True if array is a torch.Tensor
    """
    return torch is not None and isinstance(array, torch.Tensor)


def is_numpy(array: Any) -> bool:
    """Check if an array is a NumPy ndarray.
    
    Args:
        array: Array to check
        
    Returns:
        bool: True if array is a np.ndarray
    """
    return isinstance(array, np.ndarray)


@overload
def as_tensor(array: Any, device: Optional[Device] = None) -> Tensor:
    ...


def as_tensor(array: Any, device: Optional[Device] = None) -> Tensor:
    """Convert array to torch.Tensor if it's not already.
    
    Args:
        array: Array to convert
        device: Device to place tensor on
        
    Returns:
        torch.Tensor
        
    Raises:
        ImportError: If PyTorch is not available
    """
    if torch is None:
        raise ImportError("PyTorch is not available")
    
    if isinstance(array, torch.Tensor):
        if device is not None and array.device != device:
            return array.to(device)
        return array
    
    return torch.tensor(array, device=device)


def as_numpy(array: Any) -> NDArray:
    """Convert array to np.ndarray if it's not already.
    
    Args:
        array: Array to convert
        
    Returns:
        np.ndarray
    """
    if isinstance(array, np.ndarray):
        return array
    
    if torch is not None and isinstance(array, torch.Tensor):
        return array.cpu().detach().numpy()
    
    return np.array(array)


@overload
def array_zeros(shape: ShapeType, *, dtype: Optional[DType] = None) -> NDArray:
    ...


@overload
def array_zeros(shape: ShapeType, *, dtype: Optional[DType] = None, like: T) -> T:
    ...


def array_zeros(
    shape: ShapeType, *, dtype: Optional[DType] = None, like: Optional[Array] = None
) -> Array:
    """Create a zero-filled array with the same type as `like`.
    
    Args:
        shape: The shape of the array
        dtype: The data type (optional)
        like: Reference array to match type and device
        
    Returns:
        np.ndarray or torch.Tensor depending on the type of `like`
    """
    if like is None:
        # Default to numpy if no reference is provided
        return np.zeros(shape, dtype=dtype)
    
    if is_numpy(like):
        return np.zeros(shape, dtype=dtype)
    
    elif is_tensor(like):
        # Using any() to ensure mypy understands torch is not None when is_tensor is True
        assert torch is not None
        like_tensor = cast(Tensor, like)
        return torch.zeros(shape, dtype=dtype, device=like_tensor.device)
    
    else:
        return np.zeros(shape, dtype=dtype)


@overload
def array_ones(shape: ShapeType, *, dtype: Optional[DType] = None) -> NDArray:
    ...


@overload
def array_ones(shape: ShapeType, *, dtype: Optional[DType] = None, like: T) -> T:
    ...


def array_ones(
    shape: ShapeType, *, dtype: Optional[DType] = None, like: Optional[Array] = None
) -> Array:
    """Create a one-filled array with the same type as `like`.
    
    Args:
        shape: The shape of the array
        dtype: The data type (optional)
        like: Reference array to match type and device
        
    Returns:
        np.ndarray or torch.Tensor depending on the type of `like`
    """
    if like is None:
        # Default to numpy if no reference is provided
        return np.ones(shape, dtype=dtype)
    
    if is_numpy(like):
        return np.ones(shape, dtype=dtype)
    
    elif is_tensor(like):
        assert torch is not None
        like_tensor = cast(Tensor, like)
        return torch.ones(shape, dtype=dtype, device=like_tensor.device)
    
    else:
        return np.ones(shape, dtype=dtype)


def array_zeros_like(array: T, dtype: Optional[DType] = None) -> T:
    """Create a zero-filled array with the same shape and type as `array`.
    
    Args:
        array: Reference array
        dtype: The data type (optional)
        
    Returns:
        np.ndarray or torch.Tensor depending on the type of `array`
    """
    if is_numpy(array):
        return cast(T, np.zeros_like(array, dtype=dtype))
    
    elif is_tensor(array):
        assert torch is not None
        return cast(T, torch.zeros_like(array, dtype=dtype))
    
    else:
        return cast(T, np.zeros_like(array, dtype=dtype))


def array_ones_like(array: T, dtype: Optional[DType] = None) -> T:
    """Create a one-filled array with the same shape and type as `array`.
    
    Args:
        array: Reference array
        dtype: The data type (optional)
        
    Returns:
        np.ndarray or torch.Tensor depending on the type of `array`
    """
    if is_numpy(array):
        return cast(T, np.ones_like(array, dtype=dtype))
    
    elif is_tensor(array):
        assert torch is not None
        return cast(T, torch.ones_like(array, dtype=dtype))
    
    else:
        return cast(T, np.ones_like(array, dtype=dtype))


def array_where(
    condition: Array, x: Array, y: Array
) -> Array:
    """Return elements chosen from x or y depending on condition.
    
    Args:
        condition: Boolean mask
        x: Values to use where condition is True
        y: Values to use where condition is False
        
    Returns:
        np.ndarray or torch.Tensor depending on the type of inputs
    """
    if torch is not None and any(is_tensor(a) for a in (condition, x, y)):
        # Convert all inputs to tensors if any are tensors
        device = None
        for a in (condition, x, y):
            if is_tensor(a):
                device = cast(Tensor, a).device
                break
        
        condition_tensor = (
            condition if is_tensor(condition) 
            else torch.tensor(as_numpy(condition), device=device)
        )
        x_tensor = (
            x if is_tensor(x) 
            else torch.tensor(as_numpy(x), device=device)
        )
        y_tensor = (
            y if is_tensor(y) 
            else torch.tensor(as_numpy(y), device=device)
        )
        
        return torch.where(condition_tensor, x_tensor, y_tensor)
    else:
        return np.where(condition, x, y)


@overload
def array_unique(array: T, return_index: bool = False, **kwargs: Any) -> T:
    ...


@overload
def array_unique(
    array: T, return_index: bool = True, **kwargs: Any
) -> Tuple[T, Array]:
    ...


def array_unique(
    array: T, return_index: bool = False, **kwargs: Any
) -> Union[T, Tuple[T, Array]]:
    """Return unique elements of an array.
    
    Args:
        array: Input array
        return_index: If True, also return indices of original elements
        **kwargs: Additional arguments specific to the array type
        
    Returns:
        np.ndarray or torch.Tensor depending on the type of `array`,
        optionally with indices
    """
    if is_numpy(array):
        if return_index:
            return cast(Tuple[T, Array], np.unique(array, return_index=return_index, **kwargs))
        else:
            return cast(T, np.unique(array, **kwargs))
    
    elif is_tensor(array) and torch is not None:
        if "return_index" in kwargs:
            del kwargs["return_index"]  # torch.unique doesn't accept return_index as kwarg
            
        # Get unique elements
        result = torch.unique(array, **kwargs)
        
        # PyTorch doesn't have return_index in the same way, need to implement differently
        if return_index:
            indices = []
            array_cpu = array.cpu()
            result_cpu = result.cpu()
            
            # For each unique value, find first occurrence in the original array
            for val in result_cpu:
                indices.append((array_cpu == val).nonzero()[0].item())
            
            indices_tensor = torch.tensor(
                indices, dtype=torch.long, device=array.device
            )
            return cast(T, result), indices_tensor
        
        return cast(T, result)
    
    else:
        if return_index:
            return cast(Tuple[T, Array], np.unique(array, return_index=return_index, **kwargs))
        else:
            return cast(T, np.unique(array, **kwargs))


def array_concatenate(arrays: Sequence[Array], axis: int = 0) -> Array:
    """Join arrays along an existing axis.
    
    Args:
        arrays: Sequence of arrays
        axis: Axis along which to join
        
    Returns:
        np.ndarray or torch.Tensor depending on the type of input arrays
    """
    if not arrays:
        raise ValueError("Cannot concatenate empty array list")
    
    if torch is not None and any(is_tensor(a) for a in arrays):
        # Convert all arrays to tensors if any are tensors
        tensor_arrays = []
        device = None
        
        # Find the first tensor to get its device
        for a in arrays:
            if is_tensor(a):
                device = cast(Tensor, a).device
                break
        
        # Convert all arrays to tensors with the same device
        for a in arrays:
            if is_tensor(a):
                tensor_arrays.append(a)
            else:
                tensor_arrays.append(torch.tensor(as_numpy(a), device=device))
        
        return torch.cat(tensor_arrays, dim=axis)
    else:
        # Convert all arrays to numpy if needed
        numpy_arrays = [as_numpy(a) for a in arrays]
        return np.concatenate(numpy_arrays, axis=axis)


def array_equal(array1: Array, array2: Array) -> bool:
    """Check if two arrays have the same shape and elements.
    
    Args:
        array1: First array
        array2: Second array
        
    Returns:
        bool: True if arrays are equal
    """
    if is_numpy(array1) and is_numpy(array2):
        return np.array_equal(array1, array2)
    
    elif torch is not None and is_tensor(array1) and is_tensor(array2):
        return torch.equal(cast(Tensor, array1), cast(Tensor, array2))
    
    # If types don't match, convert to numpy for comparison
    return np.array_equal(as_numpy(array1), as_numpy(array2))


def for_sklearn(array: Array, function_name: Optional[str] = None) -> NDArray:
    """Ensure array is in the right format for scikit-learn.
    
    Args:
        array: Input array
        function_name: Name of the function requiring numpy (for warning message)
        
    Returns:
        np.ndarray
    """
    if is_tensor(array):
        if function_name:
            warnings.warn(
                f"{function_name} requires numpy arrays. Converting tensor to numpy, "
                f"which may impact performance."
            )
        return as_numpy(array)
    return cast(NDArray, array)


def for_pytorch(
    array: Array, device: Optional[Device] = None, function_name: Optional[str] = None
) -> Tensor:
    """Ensure array is in the right format for PyTorch.
    
    Args:
        array: Input array
        device: Device to place tensor on
        function_name: Name of the function requiring tensor (for warning message)
        
    Returns:
        torch.Tensor
        
    Raises:
        ImportError: If PyTorch is not available
    """
    if torch is None:
        raise ImportError("PyTorch is not available")
    
    if is_numpy(array):
        if function_name:
            warnings.warn(
                f"{function_name} requires PyTorch tensors. Converting numpy array to tensor, "
                f"which may impact performance."
            )
        return torch.tensor(array, device=device)
    
    tensor = cast(Tensor, array)
    if device is not None and tensor.device != device:
        return tensor.to(device)
    return tensor