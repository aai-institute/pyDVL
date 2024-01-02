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
