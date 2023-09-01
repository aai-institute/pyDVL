""" This module contains types, protocols, decorators and generic function
transformations. Some of it probably belongs elsewhere.
"""
from __future__ import annotations

import functools
from abc import ABCMeta
from typing import Any, Callable, Optional, Protocol, TypeVar, Union, cast

from numpy.random import Generator, SeedSequence
from numpy.typing import NDArray

from pydvl.utils.functional import get_free_args_fn

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


def call_fun_remove_arg(*args, fn: Callable, arg: str, **kwargs):
    """
    Calls the given function with the given arguments. In the process it removes the
    specified keyword argument from the keyword arguments.

    Args:
        args: Positional arguments to pass to the function.
        fn: The function to call.
        arg: The name of the argument to remove.
        kwargs: Keyword arguments to pass to the function.

    Returns:
        The return value of the function.
    """
    try:
        del kwargs[arg]
    except KeyError:
        pass

    return fn(*args, **kwargs)


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
    if new_arg in get_free_args_fn(fun):
        return fun

    return functools.partial(call_fun_remove_arg, fn=fun, arg=new_arg)


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
