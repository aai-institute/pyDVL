from __future__ import annotations

import logging
from functools import wraps
from itertools import cycle, takewhile
from time import time
from typing import TYPE_CHECKING, Collection, Generic, Iterable, Iterator, TypeVar

from tqdm.auto import tqdm

# This is needed to avoid circular import errors
if TYPE_CHECKING:
    from pydvl.valuation.result import ValuationResult
    from pydvl.valuation.stopping import StoppingCriterion

__all__ = ["log_duration", "Progress", "repeat_indices"]

logger = logging.getLogger(__name__)


def repeat_indices(
    indices: Collection[int],
    result: "ValuationResult",
    done: "StoppingCriterion",
    **kwargs,
) -> Iterator[int]:
    """Helper function to cycle indefinitely over a collection of indices
    until the stopping criterion is satisfied while displaying progress.

    Args:
        indices: Collection of indices that will be cycled until done.
        result: Object containing the current results.
        done: Stopping criterion.
        kwargs: Keyword arguments passed to tqdm.
    """
    with tqdm(total=100, unit="%", **kwargs) as pbar:
        it = takewhile(lambda _: not done(result), cycle(indices))
        for i in it:
            yield i
            pbar.update(100 * done.completion() - pbar.n)
            pbar.refresh()


def log_duration(_func=None, *, log_level=logging.DEBUG):
    """
    Decorator to log execution time of a function with a configurable logging level.
    It can be used with or without specifying a log level.
    """

    def decorator_log_duration(func):
        @wraps(func)
        def wrapper_log_duration(*args, **kwargs):
            func_name = func.__qualname__
            logger.log(log_level, f"Function '{func_name}' is starting.")
            start_time = time()
            result = func(*args, **kwargs)
            duration = time() - start_time
            logger.log(
                log_level,
                f"Function '{func_name}' completed. " f"Duration: {duration:.2f} sec",
            )
            return result

        return wrapper_log_duration

    if _func is None:
        # If log_duration was called without arguments, return decorator
        return decorator_log_duration
    else:
        # If log_duration was called with a function, apply decorator directly
        return decorator_log_duration(_func)


T = TypeVar("T")


class Progress(Generic[T]):
    """Displays an optional progress bar for an iterable, using
    [StoppingCriterion.completion][pydvl.value.stopping.StoppingCriterion.completion]
    for the progress.

    Args:
        iterable: The iterable to wrap.
        is_done: The stopping criterion.
        kwargs: Additional keyword arguments passed to tqdm.
            - `total`: The total number of items in the iterable (Default: 100)
            - `unit`: The unit of the progress bar. (Default: %)
            - `desc`: Description of the progress bar. (Default: str(is_done))
            - `bar_format`: Format of the progress bar. (Default is a percentage bar)
            - plus anything else that tqdm accepts
    """

    def __init__(
        self,
        iterable: Iterable[T],
        is_done: StoppingCriterion,
        **kwargs,
    ) -> None:
        self.iterable = iterable
        self.is_done = is_done
        self.total = kwargs.pop("total", 100)
        self.unit = kwargs.pop("unit", "%")
        self.desc = kwargs.pop("desc", str(is_done))
        self.bar_format = "{desc}: {percentage:0.2f}%|{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        self.kwargs = kwargs
        self.pbar: tqdm | None = None

    def __iter__(self) -> Iterator[T]:
        with tqdm(
            total=self.total,
            desc=self.desc,
            unit=self.unit,
            bar_format=self.bar_format,
            **self.kwargs,
        ) as self.pbar:
            for item in self.iterable:
                self.pbar.n = self.total * self.is_done.completion()
                self.pbar.refresh()
                yield item

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pbar is not None:
            self.pbar.close()
