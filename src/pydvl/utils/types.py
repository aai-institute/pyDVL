""" This module contains types, protocols, decorators and generic function
transformations. Some of it probably belongs elsewhere.
"""
from __future__ import annotations

from abc import ABCMeta
from typing import Any, Optional, Protocol, TypeVar, Union, cast

from numpy.random import Generator, SeedSequence
from numpy.typing import NDArray

__all__ = ["SupervisedModel", "MapFunction", "ReduceFunction", "NoPublicConstructor"]

R = TypeVar("R", covariant=True)


class MapFunction(Protocol[R]):
    def __call__(self, *args: Any, **kwargs: Any) -> R:
        ...


class ReduceFunction(Protocol[R]):
    def __call__(self, *args: Any, **kwargs: Any) -> R:
        ...


class SupervisedModel(Protocol):
    """This is the minimal Protocol that valuation methods require from
    models in order to work.

    All that is needed are the standard sklearn methods `fit()`, `predict()` and
    `score()`.
    """

    def fit(self, x: NDArray, y: NDArray):
        pass

    def predict(self, x: NDArray) -> NDArray:
        pass

    def score(self, x: NDArray, y: NDArray) -> float:
        pass


class NoPublicConstructor(ABCMeta):
    """Metaclass that ensures a private constructor

    If a class uses this metaclass like this:

        class SomeClass(metaclass=NoPublicConstructor):
            pass

    If you try to instantiate your class (`SomeClass()`),
    a `TypeError` will be thrown.

    Taken almost verbatim from:
    [https://stackoverflow.com/a/64682734](https://stackoverflow.com/a/64682734)
    """

    def __call__(cls, *args, **kwargs):
        raise TypeError(
            f"{cls.__module__}.{cls.__qualname__} cannot be initialized directly. "
            "Use the proper factory instead."
        )

    def create(cls, *args: Any, **kwargs: Any):
        return super().__call__(*args, **kwargs)


Seed = Union[int, Generator]


def ensure_seed_sequence(
    seed: Optional[Union[Seed, SeedSequence]] = None
) -> SeedSequence:
    """
    If the passed seed is a SeedSequence object then it is returned as is. If it is
    a Generator the internal protected seed sequence from the generator gets extracted.
    Otherwise, a new SeedSequence object is created from the passed (optional) seed.

    Args:
        seed: Either an int, a Generator object a SeedSequence object or None.

    Returns:
        A SeedSequence object.
    """
    if isinstance(seed, SeedSequence):
        return seed
    elif isinstance(seed, Generator):
        return cast(SeedSequence, seed.bit_generator.seed_seq)  # type: ignore
    else:
        return SeedSequence(seed)
