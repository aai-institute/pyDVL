import logging
from functools import wraps
from itertools import cycle, takewhile
from time import time
from typing import TYPE_CHECKING, Collection, Iterator

from tqdm.auto import tqdm

# This is needed to avoid circular import errors
if TYPE_CHECKING:
    from pydvl.value.result import ValuationResult
    from pydvl.value.stopping import StoppingCriterion

__all__ = ["repeat_indices", "log_duration"]

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
