from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Generator, Iterable, Protocol, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "BatchGenerator",
    "IndexT",
    "IndexSetT",
    "LossFunction",
    "NameT",
    "NullaryPredicate",
    "Sample",
    "SampleBatch",
    "SampleGenerator",
    "SampleT",
    "UtilityEvaluation",
    "ValueUpdate",
]

IndexT = np.int_
IndexSetT = NDArray[IndexT]
NameT = Union[np.object_, np.int_]
NullaryPredicate = Callable[[], bool]


@dataclass(frozen=True)
class ValueUpdate:
    idx: IndexT
    update: float


@dataclass(frozen=True)
class Sample:
    idx: IndexT | None
    subset: NDArray[IndexT]

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


SampleT = TypeVar("SampleT", bound=Sample)

SampleBatch = Iterable[Sample]
SampleGenerator = Generator[Sample, None, None]
BatchGenerator = Generator[SampleBatch, bool, None]


@dataclass(frozen=True)
class UtilityEvaluation:
    idx: IndexT
    subset: IndexSetT
    evaluation: float

    def __iter__(self):  # No way to type the return Iterator properly
        return iter((self.idx, self.subset, self.evaluation))


class LossFunction(Protocol):
    def __call__(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        ...
