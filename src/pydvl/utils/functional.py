from __future__ import annotations

import inspect
from functools import partial
from typing import Callable, Set, Tuple

__all__ = ["unroll_partial_fn_args"]


def unroll_partial_fn_args(fun: Callable) -> Set[str]:
    """
    Unroll a function that was set by functools.partial.

    Args:
        fun: Either or a function to unroll.

    Returns:
        A tuple of the unrolled function and a set of the parameters that were set by
        functools.partial.
    """
    args_set_by_partial: Set[str] = set()

    def _rec_unroll_partial_function(g: Callable):
        """
        Store arguments and recursively call itself if the function is a partial. In the
        end, return the original function.
        """
        nonlocal args_set_by_partial

        if isinstance(g, partial):
            args_set_by_partial.update(g.keywords.keys())
            args_set_by_partial.update(g.args)
            return _rec_unroll_partial_function(g.keywords["fun"])
        else:
            return g

    wrapped_fn = _rec_unroll_partial_function(fun)
    sig = inspect.signature(wrapped_fn)
    return args_set_by_partial | set(sig.parameters.keys())
