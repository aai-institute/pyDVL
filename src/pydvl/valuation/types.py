"""
This module contains different types used by [pydvl.valuation][]

If you are interested in extending valuation methods, you might need to subclass
[ValueUpdate][pydvl.valuation.types.ValueUpdate], [Sample][pydvl.valuation.types.Sample]
or [ClasswiseSample][pydvl.valuation.types.ClasswiseSample]. These are the data types
used for communication between the samplers on the main process and the workers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Protocol,
    Type,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

import numpy as np
import torch as torch_mod
from numpy.typing import NDArray
from torch import Tensor, nn
from typing_extensions import Self, TypeAlias

from pydvl.utils.array import (
    Array,
    ArrayRetT,
    ArrayT,
    array_concatenate,
    array_equal,
    to_numpy,
    try_torch_import,
)

__all__ = [
    "BaggingModel",
    "BatchGenerator",
    "ClasswiseSample",
    "IndexT",
    "IndexSetT",
    "LossFunction",
    "NameT",
    "NullaryPredicate",
    "PointwiseScore",
    "Sample",
    "SampleBatch",
    "SampleGenerator",
    "SampleT",
    "SemivalueCoefficient",
    "SupervisedModel",
    "TorchSupervisedModel",
    "UtilityEvaluation",
    "ValueUpdate",
    "ValueUpdateT",
]

IndexT: TypeAlias = np.int_
IndexSetT: TypeAlias = NDArray[IndexT]
NameT: TypeAlias = Union[np.object_, np.int_, np.str_]
NullaryPredicate: TypeAlias = Callable[[], bool]

torch = try_torch_import()


@dataclass(frozen=True)
class ValueUpdate:
    """ValueUpdates are emitted by evaluation strategies.

    Typically, a value update is the product of a marginal utility, the sampler weight
    and the valuation's coefficient. Instead of multiplying weights, coefficients and
    utilities directly, the strategy works in log-space for numerical stability using
    the samplers' log-weights and the valuation methods' log-coefficients.

    The updates from all workers are converted back to linear space by
    [LogResultUpdater][pydvl.valuation.samplers.base.LogResultUpdater].

    Attributes:
        idx: Index of the sample the update corresponds to.
        log_update: Logarithm of the absolute value of the update.
        sign: Sign of the update.
    """

    idx: IndexT | None
    log_update: float
    sign: int

    def __init__(self, idx: IndexT | None, log_update: float, sign: int):
        object.__setattr__(self, "idx", idx)
        object.__setattr__(self, "log_update", log_update)
        object.__setattr__(self, "sign", sign)


ValueUpdateT = TypeVar("ValueUpdateT", bound=ValueUpdate, contravariant=True)


@dataclass(frozen=True)
class Sample:
    idx: IndexT | None
    """Index of current sample"""

    subset: NDArray[np.int_]
    """Indices of current sample"""

    def __post_init__(self):
        """Ensure that the subset is a numpy array of integers."""
        try:
            self.__dict__["subset"] = to_numpy(self.subset)
        except Exception:
            raise TypeError(
                f"subset must be a numpy array, got {type(self.subset).__name__}"
            )
        if self.subset.size == 0:
            self.__dict__["subset"] = self.subset.astype(int)
        if not np.issubdtype(self.subset.dtype, np.integer):
            raise TypeError(
                f"subset must be a numpy array of integers, got {self.subset.dtype}"
            )

    # Make the unpacking operator work
    def __iter__(self):  # No way to type the return Iterator properly
        return iter((self.idx, self.subset))

    def __hash__(self):
        """This type must be hashable for the utility caching to work.
        We use hashlib.sha256 which is about 4-5x faster than hash(), and returns the
        same value in all processes, as opposed to hash() which is salted in each
        process
        """
        sha256_hash = hashlib.sha256(self.subset.tobytes()).hexdigest()
        return int(sha256_hash, base=16)

    def with_idx_in_subset(self) -> Self:
        """Return a copy of sample with idx added to the subset.

        Returns the original sample if idx was already part of the subset.

        Returns:
            Sample: A copy of the sample with idx added to the subset.

        Raises:
            ValueError: If idx is None.
        """
        if self.idx in self.subset:
            return self

        if self.idx is None:
            raise ValueError("Cannot add idx to subset if idx is None.")

        new_subset = array_concatenate([self.subset, np.array([self.idx])])
        return replace(self, subset=new_subset)

    def with_idx(self, idx: IndexT) -> Self:
        """Return a copy of sample with idx changed.

        Returns the original sample if idx is the same.

        Args:
            idx: New value for idx.

        Returns:
            Sample: A copy of the sample with idx changed.
        """
        if self.idx == idx:
            return self

        return replace(self, idx=idx)

    def with_subset(self, subset: Array[IndexT]) -> Self:
        """Return a copy of sample with the subset changed.

        Args:
            subset: New value for subset.

        Returns:
            A copy of the sample with subset changed.
        """
        return replace(self, subset=to_numpy(subset))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sample):
            return False

        idx_equal = self.idx == other.idx
        subset_equal = array_equal(self.subset, other.subset)

        return idx_equal and subset_equal


@dataclass(frozen=True)
class ClasswiseSample(Sample):
    """Sample class for classwise shapley valuation"""

    label: int
    """Label of the current sample"""

    ooc_subset: NDArray[np.int_]
    """Indices of out-of-class elements, i.e., those with a label different from
    this sample's label"""

    def __post_init__(self):
        """Ensure that the subset and ooc_subset are numpy arrays of integers."""
        super().__post_init__()
        try:
            self.__dict__["ooc_subset"] = to_numpy(self.ooc_subset)
        except Exception:
            raise TypeError(
                f"ooc_subset must be a numpy array, got {type(self.ooc_subset).__name__}"
            )
        if self.ooc_subset.size == 0:
            self.__dict__["ooc_subset"] = self.ooc_subset.astype(int)
        if not np.issubdtype(self.ooc_subset.dtype, np.integer):
            raise TypeError(
                f"ooc_subset must be a numpy array of integers, got {self.ooc_subset.dtype}"
            )

    # Make the unpacking operator work
    def __iter__(self):  # No way to type the return Iterator properly
        return iter((self.idx, self.subset, self.label, self.ooc_subset))

    def __hash__(self):
        array_bytes = self.subset.tobytes() + self.ooc_subset.tobytes()
        sha256_hash = hashlib.sha256(array_bytes).hexdigest()
        return int(sha256_hash, base=16)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClasswiseSample):
            return False

        idx_equal = self.idx == other.idx
        label_equal = self.label == other.label
        subset_equal = array_equal(self.subset, other.subset)
        ooc_equal = array_equal(self.ooc_subset, other.ooc_subset)

        return idx_equal and subset_equal and label_equal and ooc_equal


SampleT = TypeVar("SampleT", bound=Sample)

SampleBatch = Iterable[SampleT]
SampleGenerator = Generator[SampleT, None, None]
BatchGenerator = Generator[SampleBatch[SampleT], None, None]


@dataclass(frozen=True)
class UtilityEvaluation:
    idx: IndexT
    subset: IndexSetT
    evaluation: float

    def __iter__(self):  # No way to type the return Iterator properly
        return iter((self.idx, self.subset, self.evaluation))


class LossFunction(Protocol):
    def __call__(self, y_true: Array, y_pred: Array) -> Array: ...


class SemivalueCoefficient(Protocol):
    def __call__(self, n: int, k: int) -> float:
        """A semi-value coefficient is a function of the number of elements in the set,
        and the size of the subset for which the coefficient is being computed.
        Because both coefficients and sampler weights can be very large or very small,
        we perform all computations in log-space to avoid numerical issues.

        Args:
            n: Total number of elements in the set.
            k: Size of the subset for which the coefficient is being computed

        Returns:
            The natural logarithm of the semi-value coefficient.
        """
        ...


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
