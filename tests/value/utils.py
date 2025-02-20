from __future__ import annotations

import time
from copy import deepcopy
from functools import wraps
from logging import getLogger
from typing import Callable, Protocol, Tuple, TypeVar

from pydvl.utils.types import Seed

logger = getLogger(__name__)

ReturnT = TypeVar("ReturnT")


def call_with_seeds(fun: Callable, *args, seeds: Tuple[Seed, ...], **kwargs) -> Tuple:
    """
    Execute a function multiple times with different seeds. It copies the arguments
    and keyword arguments before passing them to the function.

    Args:
        fun: The function to execute.
        args: The arguments to pass to the function.
        seeds: The seeds to use.
        kwargs: The keyword arguments to pass to the function.

    Returns:
        A tuple of the results of the function.
    """
    return tuple(fun(*deepcopy(args), **deepcopy(kwargs), seed=seed) for seed in seeds)


class TimedCallable(Protocol[ReturnT]):
    """A callable that has an attribute to keep track of execution time."""

    execution_time: float

    def __call__(self, *args, **kwargs) -> ReturnT: ...


def timed(fun: Callable[..., ReturnT]) -> TimedCallable:
    """
    Takes a function `func` and returns a function with the same input arguments and
    the original return value along with the execution time.

    Args:
        fun: The function to be measured, accepting arbitrary arguments and returning
            any type.

    Returns:
        A wrapped function that, when called, returns a tuple containing the original
            function's result and its execution time in seconds. The decorated function
            will have the same input arguments and return type as the original function.
    """

    wrapper: TimedCallable

    @wraps(fun)
    def wrapper(*args, **kwargs) -> ReturnT:
        start_time = time.time()
        result = fun(*args, **kwargs)
        end_time = time.time()
        wrapper.execution_time = end_time - start_time
        logger.info(f"{fun.__name__} took {wrapper.execution_time:.5f} seconds.")
        return result

    wrapper.execution_time = 0.0
    return wrapper
