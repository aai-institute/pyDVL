""" This module contains types, protocols, decorators and generic function
transformations. Some of it probably belongs elsewhere.
"""
from __future__ import annotations

import inspect
from abc import ABCMeta
from typing import Any, Callable, Protocol, TypeVar

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


def maybe_add_argument(fun: Callable, new_arg: str) -> Callable:
    """Wraps a function to accept the given keyword parameter if it doesn't
    already.

    If `fun` already takes a keyword parameter of name `new_arg`, then it is
    returned as is. Otherwise, a wrapper is returned which merely ignores the
    argument.

    Args:
        fun: The function to wrap
        new_arg: The name of the argument that the new function will accept
            (and ignore).

    Returns:
        A new function accepting one more keyword argument.
    """
    params = inspect.signature(fun).parameters
    if new_arg in params.keys():
        return fun

    def wrapper(*args, **kwargs):
        try:
            del kwargs[new_arg]
        except KeyError:
            pass
        return fun(*args, **kwargs)

    return wrapper


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
