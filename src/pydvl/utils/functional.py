from __future__ import annotations

import inspect
from functools import partial
from typing import Callable, Set, Tuple

__all__ = ["fn_accepts_param_name"]


def fn_accepts_param_name(fn: Callable, param_name: str) -> bool:
    """
    Checks if a function accepts a given parameter, even if it is set by partial.

    :param fn: The function to check.
    :param param_name: The name of the parameter to check.
    :return: True if the function accepts the parameter, False otherwise.
    """

    wrapped_fn, args_set_by_partial = _unroll_partial_fn(fn)

    sig = inspect.signature(wrapped_fn)
    params = sig.parameters

    # Check if the parameter was set by functools.partial
    if param_name in args_set_by_partial:
        return False

    # Check if the function accepts the specific parameter
    if param_name in params:
        return True

    # Check if the function accepts **kwargs
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return True

    return False


def _unroll_partial_fn(fn: Callable) -> Tuple[Callable, Set[str]]:
    """
    Unroll a function that was set by functools.partial.

    :param fn: Either or a function to unroll.
    :return: A tuple of the unrolled function and a set of the parameters that were set
        by functools.partial.
    """
    args_set_by_partial = set()

    def _rec_unroll_partial_function(g: Callable):
        nonlocal args_set_by_partial

        if isinstance(g, partial):
            args_set_by_partial.update(g.keywords.keys())
            args_set_by_partial.update(g.args)
            return _rec_unroll_partial_function(g.func)
        else:
            return g

    return _rec_unroll_partial_function(fn), args_set_by_partial
