"""This module contains types, protocols, decorators and generic function
transformations. Some of it probably belongs elsewhere.
"""

from __future__ import annotations

from typing import (
    Any,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

import numpy as np
from numpy.random import Generator, SeedSequence
from numpy.typing import NDArray

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
    "ensure_seed_sequence",
    "validate_number",
]


IndexT = TypeVar("IndexT", bound=np.int_)
NameT = TypeVar("NameT", np.object_, np.int_)
R = TypeVar("R", covariant=True)
Seed = Union[int, Generator]


class MapFunction(Protocol[R]):
    def __call__(self, *args: Any, **kwargs: Any) -> R: ...


class ReduceFunction(Protocol[R]):
    def __call__(self, *args: Any, **kwargs: Any) -> R: ...


class PointwiseScore(Protocol):
    def __call__(self, y_true: NDArray, y_pred: NDArray) -> NDArray: ...


@runtime_checkable
class BaseModel(Protocol):
    """This is the minimal model protocol with the method `fit()`"""

    def fit(self, x: NDArray, y: NDArray | None):
        """Fit the model to the data

        Args:
            x: Independent variables
            y: Dependent variable
        """
        pass


@runtime_checkable
class SupervisedModel(Protocol):
    """This is the standard sklearn Protocol with the methods `fit()`, `predict()` and
    `score()`.
    """

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

    def score(self, x: NDArray, y: NDArray | None) -> float:
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
