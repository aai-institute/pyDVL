from __future__ import annotations

import inspect
from functools import partial
from typing import Callable, Set, Tuple, Union

__all__ = ["get_free_args_fn"]


def get_free_args_fn(fun: Union[Callable, partial]) -> Set[str]:
    """
    Accept a function or partial definition and return the set of arguments that are
    free. An argument is free if it is not set by the partial and is a parameter of the
    function.

    Args:
        fun: A partial or a function to unroll.

    Returns:
        A set of arguments that were set by the partial.
    """
    args_set_by_partial: Set[str] = set()

    def _rec_unroll_partial_function(g: Union[Callable, partial]) -> Callable:
        """
        Store arguments and recursively call itself if the function is a partial. In the
        end, return the initial wrapped function.

        Args:
            g: A partial or a function to unroll.

        Returns:
            Initial wrapped function.
        """
        nonlocal args_set_by_partial

        if isinstance(g, partial):
            args_set_by_partial.update(g.keywords.keys())
            args_set_by_partial.update(g.args)
            inner_fn = g.keywords["fn"] if "fn" in g.keywords else g.func
            return _rec_unroll_partial_function(inner_fn)
        else:
            return g

    wrapped_fn = _rec_unroll_partial_function(fun)
    sig = inspect.signature(wrapped_fn)
    return args_set_by_partial | set(sig.parameters.keys())
