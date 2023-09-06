from __future__ import annotations

import time
from copy import deepcopy
from functools import wraps
from logging import getLogger
from typing import Callable, Optional, Tuple, TypeVar

from pydvl.utils.types import Seed

logger = getLogger(__name__)

ReturnType = TypeVar("ReturnType")


def call_fn_multiple_seeds(
    fn: Callable, *args, seeds: Tuple[Seed, ...], **kwargs
) -> Tuple:
    """
    Execute a function multiple times with different seeds. It copies the arguments
    and keyword arguments before passing them to the function.

    Args:
        fn: The function to execute.
        args: The arguments to pass to the function.
        seeds: The seeds to use.
        kwargs: The keyword arguments to pass to the function.

    Returns:
        A tuple of the results of the function.
    """
    return tuple(fn(*deepcopy(args), **deepcopy(kwargs), seed=seed) for seed in seeds)


def measure_execution_time(
    func: Callable[..., ReturnType]
) -> Callable[..., Tuple[Optional[ReturnType], float]]:
    """
    Takes a function `func` and returns a function with the same input arguments and
    the original return value along with the execution time.

    Args:
        func: The function to be measured, accepting arbitrary arguments and returning
            any type.

    Returns:
        A wrapped function that, when called, returns a tuple containing the original
            function's result and its execution time in seconds. The decorated function
            will have the same input arguments and return type as the original function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Tuple[Optional[ReturnType], float]:
        result = None
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"{func.__name__} took {execution_time:.5f} seconds.")
            return result, execution_time

    return wrapper
