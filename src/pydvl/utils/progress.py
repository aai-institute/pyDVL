from collections.abc import Iterator
from itertools import cycle, takewhile
from typing import Collection

from tqdm.auto import tqdm

from pydvl.value.result import ValuationResult
from pydvl.value.stopping import StoppingCriterion

__all__ = ["repeat_indices"]


def repeat_indices(
    indices: Collection[int], result: ValuationResult, done: StoppingCriterion, **kwargs
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
