from valuation.utils.dataset import Dataset
from valuation.utils.numeric import vanishing_derivatives, utility, powerset
from valuation.utils.types import SupervisedModel
from valuation.utils.parallel import run_and_gather, parallel_wrap

__all__ = ['SupervisedModel', 'Dataset',
           'run_and_gather', 'parallel_wrap', 'vanishing_derivatives',
           'utility', 'powerset', 'maybe_progress']

from typing import Iterator
from tqdm.auto import tqdm


def maybe_progress(it: Iterator, display: bool, **tqdm_kwargs):
    return tqdm(it, **tqdm_kwargs) if display else it
