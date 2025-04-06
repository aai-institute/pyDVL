from __future__ import annotations

from copy import deepcopy
from typing import Callable, Tuple

from pydvl.utils.types import Seed


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
