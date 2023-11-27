"""
!!! Warning
    This module is deprecated and will be removed in a future release.
    It implements a wrapper for the [tqdm](https://tqdm.github.io/) progress bar
    iterator for easy toggling, but this functionality is already provided by
    the `disable` argument of `tqdm`.
"""
import collections.abc
import logging
from functools import wraps
from time import time
from typing import Iterable, Iterator, Union

from tqdm.auto import tqdm

__all__ = ["maybe_progress"]

logger = logging.getLogger(__name__)


class MockProgress(collections.abc.Iterator):
    """A Naive mock class to use with maybe_progress and tqdm.
    Mocked methods don't support return values.
    Mocked properties don't do anything
    """

    class MiniMock:
        def __call__(self, *args, **kwargs):
            pass

        def __add__(self, other):
            pass

        def __sub__(self, other):
            pass

        def __mul__(self, other):
            pass

        def __floordiv__(self, other):
            pass

        def __truediv__(self, other):
            pass

    def __init__(self, iterator: Union[Iterator, Iterable]):
        # Since there is no _it in __dict__ at this point, doing here
        # self._it = iterator
        # results in a call to __getattr__() and the assignment fails, so we
        # use __dict__ instead
        self.__dict__["_it"] = iterator

    def __iter__(self):
        return iter(self._it)

    def __next__(self):
        return next(self._it)

    def __getattr__(self, key):
        return self.MiniMock()

    def __setattr__(self, key, value):
        pass


def maybe_progress(
    it: Union[int, Iterable, Iterator], display: bool = False, **kwargs
) -> Union[tqdm, MockProgress]:
    """Returns either a tqdm progress bar or a mock object which wraps the
    iterator as well, but ignores any accesses to methods or properties.

    Args:
        it: the iterator to wrap
        display: set to True to return a tqdm bar
        kwargs: Keyword arguments that will be forwarded to tqdm
    """
    if isinstance(it, int):
        it = range(it)  # type: ignore
    return tqdm(it, **kwargs) if display else MockProgress(it)


def log_duration(func):
    """
    Decorator to log execution time of a function
    """

    @wraps(func)
    def wrapper_log_duration(*args, **kwargs):
        func_name = func.__qualname__
        logger.info(f"Function '{func_name}' is starting.")
        start_time = time()
        result = func(*args, **kwargs)
        duration = time() - start_time
        logger.info(f"Function '{func_name}' completed. Duration: {duration:.2f} sec")
        return result

    return wrapper_log_duration
