"""This module contains types, protocols, decorators and generic function
transformations. Some of it probably belongs elsewhere.
"""

from __future__ import annotations

from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

import numpy as np
import sklearn.utils
from numpy.random import Generator, SeedSequence
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Self

__all__ = [
    "BaggingModel",
    "BaseModel",
    "IndexT",
    "PointwiseScore",
    "MapFunction",
    "NameT",
    "ReduceFunction",
    "Seed",
    "SupervisedModel",
    "atleast1d",
    "ensure_seed_sequence",
    "require_torch",
    "try_torch_import",
    "validate_number",
]

if TYPE_CHECKING:
    # This can be confusing: these symbols cannot be used at runtime!
    import torch as torch_mod  # for use in casts and types
    from torch import Tensor, nn  # for convenience


def try_torch_import(require: bool = False) -> ModuleType | None:
    """Import torch if available, otherwise return None.
    Args:
        require: If True, raise an ImportError if torch is not available.
    """
    try:
        import torch

        return torch
    except ImportError as e:
        if require:
            raise ImportError("torch is required but not installed.") from e
        return None


def require_torch() -> ModuleType:
    torch = try_torch_import(require=True)
    assert torch is not None
    return torch


IndexT = TypeVar("IndexT", bound=np.int_)
NameT = TypeVar("NameT", np.object_, np.int_)
Seed = Union[int, Generator]
R = TypeVar("R", covariant=True)


class MapFunction(Protocol[R]):
    def __call__(self, *args: Any, **kwargs: Any) -> R: ...


class ReduceFunction(Protocol[R]):
    def __call__(self, *args: Any, **kwargs: Any) -> R: ...


DT = TypeVar("DT")


@runtime_checkable
class Array(Protocol[DT]):
    """Poor man's intersection of NDArray and torch.Tensor.

    !!! warning
        This is a "best-effort" attempt at having something usable for our purposes, but
        by no means complete. If it causes more trouble than it solves, it should be
        removed.
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


ArrayT = TypeVar("ArrayT", bound=Array, contravariant=True)
ArrayRetT = TypeVar("ArrayRetT", bound=Array, covariant=True)


class PointwiseScore(Protocol[ArrayT, ArrayRetT]):
    def __call__(self, y_true: ArrayT, y_pred: ArrayT) -> ArrayRetT: ...


@runtime_checkable
class BaseModel(Protocol[ArrayT]):
    """This is the minimal model protocol with the method `fit()`"""

    def fit(self, x: ArrayT, y: ArrayT | None):
        """Fit the model to the data

        Args:
            x: Independent variables
            y: Dependent variable
        """
        pass


@runtime_checkable
class SupervisedModel(Protocol[ArrayT, ArrayRetT]):
    """This is the standard sklearn Protocol with the methods `fit()`, `predict()` and
    `score()`.
    """

    def fit(self, x: ArrayT, y: ArrayT | None):
        """Fit the model to the data

        Args:
            x: Independent variables
            y: Dependent variable
        """
        pass

    def predict(self, x: ArrayT) -> ArrayRetT:
        """Compute predictions for the input

        Args:
            x: Independent variables for which to compute predictions

        Returns:
            Predictions for the input
        """
        pass

    def score(self, x: ArrayT, y: ArrayT | None) -> float:
        """Compute the score of the model given test data

        Args:
            x: Independent variables
            y: Dependent variable

        Returns:
            The score of the model on `(x, y)`
        """
        pass


@runtime_checkable
class BaggingModel(Protocol):
    """Any model with the attributes `n_estimators` and `max_samples` is considered a
    bagging model."""

    n_estimators: int
    max_samples: float

    def fit(self, x: NDArray, y: NDArray | None):
        """Fit the model to the data

        Args:
            x: Independent variables
            y: Dependent variable
        """
        pass

    def predict(self, x: NDArray) -> NDArray:
        """Compute predictions for the input

        Args:
            x: Independent variables for which to compute predictions

        Returns:
            Predictions for the input
        """
        pass


def ensure_seed_sequence(
    seed: Optional[Union[Seed, SeedSequence]] = None,
) -> SeedSequence:
    """
    If the passed seed is a SeedSequence object then it is returned as is. If it is
    a Generator the internal protected seed sequence from the generator gets extracted.
    Otherwise, a new SeedSequence object is created from the passed (optional) seed.

    Args:
        seed: Either an int, a Generator object a SeedSequence object or None.

    Returns:
        A SeedSequence object.

    !!! tip "New in version 0.7.0"
    """
    if isinstance(seed, SeedSequence):
        return seed
    elif isinstance(seed, Generator):
        return cast(SeedSequence, seed.bit_generator.seed_seq)  # type: ignore
    else:
        return SeedSequence(seed)


T = TypeVar("T", bound=Union[int, float, np.number])


def validate_number(
    name: str,
    value: Any,
    dtype: Type[T],
    lower: T | None = None,
    upper: T | None = None,
) -> T:
    """Ensure that the value is of the given type and within the given bounds.

    For int and float types, this function is lenient with numpy numeric types and
    will convert them to the appropriate Python type as long as no precision is lost.

    Args:
        name: The name of the variable to validate.
        value: The value to validate.
        dtype: The type to convert the value to.
        lower: The lower bound for the value (inclusive).
        upper: The upper bound for the value (inclusive).

    Raises:
        TypeError: If the value is not of the given type.
        ValueError: If the value is not within the given bounds, if there is precision
            loss, e.g. when forcing a float to an int, or if `dtype` is not a valid
            scalar type.
    """
    if not isinstance(value, (int, float, np.number)):
        raise TypeError(f"'{name}' is not a number, it is {type(value).__name__}")
    if not issubclass(dtype, (np.number, int, float)):
        raise ValueError(f"type '{dtype}' is not a valid scalar type")

    converted = dtype(value)
    if not np.isnan(converted) and not np.isclose(converted, value, rtol=0, atol=0):
        raise ValueError(
            f"'{name}' cannot be converted to {dtype.__name__} without precision loss"
        )
    value = cast(T, converted)

    if lower is not None and value < lower:  # type: ignore
        raise ValueError(f"'{name}' is {value}, but it should be >= {lower}")
    if upper is not None and value > upper:  # type: ignore
        raise ValueError(f"'{name}' is {value}, but it should be <= {upper}")
    return value


@runtime_checkable
class TorchSupervisedModel(Protocol):
    """This is the standard sklearn Protocol with the methods `fit()`, `predict()`
    and `score()`, but accepting Tensors and with some additional methods
    used by TorchUtility
    """

    device: str | torch_mod.device

    def fit(self, x: Tensor, y: Tensor | None):
        """Fit the model to the data

        Args:
            x: Independent variables
            y: Dependent variable
        """
        ...

    def predict(self, x: Tensor) -> Tensor:
        """Compute predictions for the input

        Args:
            x: Independent variables for which to compute predictions

        Returns:
            Predictions for the input
        """
        ...

    def score(self, x: Tensor, y: Tensor | None) -> float:
        """Compute the score of the model given test data

        Args:
            x: Independent variables
            y: Dependent variable

        Returns:
            The score of the model on `(x, y)`
        """
        ...

    def get_params(self, deep: bool = False) -> dict[str, Any]: ...

    @staticmethod
    def reshape_inputs(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Reshape arbitrary tensors into the shapes required by the
        model."""
        ...

    def make_model(self) -> nn.Module: ...


def check_X_y_torch(  # noqa N802
    X: Tensor,  # noqa N802
    y: Tensor,
    *,
    multi_output: bool = False,
    estimator: str | object | None = None,
    copy: bool = False,
):
    """Input validation for torch tensors mimicking some of the functionality of
    [sklearn.utils.check_X_y][].

    Args:
        X: Input data. Expected to be at least 2D with shape (n_samples, ...).
        y: Target values. If `multi_output` is `False`, expected to be a 1D tensor
           of shape (n_samples,).  If `multi_output` is `True`, `y` can be 1D or 2D
           with shape (n_samples,) or (n_samples, n_outputs).
        multi_output: Whether to allow multi-output targets (i.e. 2D y).
        estimator: The estimator instance or its name. If provided, its name is
            included in error messages.
        copy: If True, X and y are cloned to avoid modifying the originals.

    Returns:
        X_converted: The validated X tensor.
        y_converted: The validated y tensor.

    Raises
        ValueError or TypeError if any of the validation checks fail.
    """
    torch = require_torch()

    estimator_name = (
        estimator.__class__.__name__
        if estimator is not None and not isinstance(estimator, str)
        else estimator or "check_X_y"
    )

    if y is None:
        raise ValueError(
            f"{estimator_name} requires y to be passed, but the target y is None"
        )

    if not isinstance(X, torch.Tensor):
        raise TypeError(
            f"Expected X to be a torch.Tensor, got {type(X).__name__} instead."
        )
    if not isinstance(y, torch.Tensor):
        raise TypeError(
            f"Expected y to be a torch.Tensor, got {type(y).__name__} instead."
        )

    if copy:
        X = X.clone()
        y = y.clone()

    # If X is 1D, add an extra dimension.
    if X.dim() == 1:
        X = X.unsqueeze(1)

    # Check that X is at least 2D.
    if X.dim() < 2:
        raise ValueError(
            f"Expected at least 2D input for X, got {X.dim()}D tensor with shape {tuple(X.shape)}."
        )

    # Validate y based on multi_output.
    if multi_output:
        # Allow 1D or 2D.
        if y.dim() not in (1, 2):
            raise ValueError(
                f"Expected y to be 1D or 2D for multi_output, got {y.dim()}D tensor with shape {tuple(y.shape)}."
            )
    else:
        if y.dim() == 2 and y.size(1) == 1:
            # If y is a column vector, flatten it.
            y = y.view(-1)
        elif y.dim() != 1:
            raise ValueError(
                f"Expected 1D input for y, got {y.dim()}D tensor with shape {tuple(y.shape)}."
            )

    if X.size(0) != y.size(0):
        raise ValueError(
            f"Inconsistent number of samples: X has {X.size(0)} samples, y has {y.size(0)} samples."
        )

    # Ensure X has at least one sample.
    if X.size(0) < 1:
        raise ValueError(f"Found array with 0 samples (shape={tuple(X.shape)}).")

    if not torch.all(torch.isfinite(X)):
        raise ValueError(
            f"Input X in {estimator_name} contains non-finite values (NaN or Inf)."
        )

    if not torch.all(torch.isfinite(y)):
        raise ValueError(
            f"Input y in {estimator_name} contains non-finite values (NaN or Inf)."
        )

    return X, y


def check_X_y(  # noqa N802
    X: ArrayT,  # noqa N802
    y: ArrayT,
    *,
    multi_output: bool = False,
    estimator: str | object | None = None,
    copy: bool = False,
):
    """Input validation for numpy arrays mimicking some of the functionality of
    [sklearn.utils.check_X_y][].

    Args:
        X: Input data. Expected to be at least 2D with shape (n_samples, ...).
        y: Target values. If `multi_output` is `False`, expected to be a 1D array of
           shape (n_samples,).  If `multi_output` is `True`, `y` can be 1D or 2D with
           shape (n_samples,) or (n_samples, n_outputs).
        multi_output: Whether to allow multi-output targets (i.e. 2D y).
        estimator: The estimator instance or its name. If provided, its name is included
            in error messages.
        copy: If True, X and y are cloned to avoid modifying the originals.

    Returns:
        X_converted: The validated X array.
        y_converted: The validated y array.

    Raises
        ValueError or TypeError if any of the validation checks fail.
    """
    if torch := try_torch_import():
        if all(isinstance(a, torch.Tensor) for a in (X, y)):
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


def atleast1d(a: Array | ArrayLike) -> Array:
    """Ensures that the array is at least 1D.

    This is a best-effort attempt to convert non array-like objects to 1D arrays. If the
    input is a list, tuple or any other object that can be converted to a numpy array,
    it will be converted to a 1D numpy array. If the input is a scalar, it will be
    converted to a 1D array with one element. If the input is a numpy array, it will be
    reshaped to be 1D. If the input is a torch tensor, it will be reshaped to be 1D.

    Args:
        a: The array to check. This can be a numpy array, a torch tensor, a list, a
            tuple, a bytearray or a bytes object.
    Returns:
        The array, reshaped to be a 1D numpy array, or a torch tensor if the input was
            already a torch tensor.
    """
    if isinstance(a, np.ndarray):
        return cast(NDArray, np.atleast_1d(a))
    if torch := try_torch_import():
        if isinstance(a, torch.Tensor):
            return cast(torch_mod.Tensor, a.unsqueeze(0) if a.ndim == 0 else a)
    return cast(NDArray, np.atleast_1d(np.array(a)))
