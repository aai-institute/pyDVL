from __future__ import annotations

from itertools import islice
from typing import Any, Generator, Iterable, List, TypeVar

import numpy as np

from pydvl.utils.types import Seed

T = TypeVar("T")


def take_n(it: Iterable[T], size: int = 2) -> Generator[List[T], Any, None]:
    """Takes tuples of `size` items from an iterator at a time.

    Args:
        it: The generator to take items from.
        size: The number of items to take at a time.
    Returns:
        A generator yielding tuples of n items.
    """

    while batch := list(islice(it, size)):
        yield batch


class StochasticSamplerMixin:
    """Mixin class for samplers which use a random number generator."""

    def __init__(self, *args, seed: Seed | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._rng = np.random.default_rng(seed)
