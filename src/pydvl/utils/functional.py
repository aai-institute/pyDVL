from __future__ import annotations

import inspect
from functools import partial
from typing import Callable, Set, Tuple, Union

__all__ = ["get_free_args_fn", "fn_accept_additional_argument"]


def fn_accept_additional_argument(*args, fn: Callable, arg: str, **kwargs):
    """
    Calls the given function with the given arguments. In the process of calling the
    wrapped function, it removes the specified keyword argument from the passed keyword
    arguments. This function can be pickled by `pickle` as it is on the .

    Args:
        args: Positional arguments to pass to the function.
        fn: The function to call.
        arg: The name of the argument to remove.
        kwargs: Keyword arguments to pass to the function.

    Returns:
        The return value of the function.
    """
    try:
        del kwargs[arg]
    except KeyError:
        pass

    return fn(*args, **kwargs)


def get_free_args_fn(fun: Union[Callable, partial]) -> Set[str]:
    """
    Accept a function or a partial definition and return the set of arguments that are
    free. An argument is considered free if it is not set by the partial and is a
    parameter of the function. Formally, this can be described as follows:

    Let phi be the function that returns the set of arguments that are set by a given
    function. For functions f and g, one can extend the function phi recursively to psi
    as follows:

        a) If `f = partial(g, **kwargs)`, two cases arise:
            a) If `g = fn_accept_additional_argument`, then
                `psi(f) = psi(kwargs.fn) + {kwargs.arg}`
            b) Else, `psi(f) = psi(g) - kwargs.keys()`
        b) Else,`psi(g) = phi(g)` (Note that this is the base case.)

    Transforming `fn_accept_additional_argument` into a partial function is done by:
    `f(fn, arg) = partial(fn_accept_additional_argument, **{fn: fn, arg: arg})`
    This function is the inverse (with respect to phi) to the function
    `g(fn, arg, val) = partial(g, {arg: val})` in the sense that
    `phi(fn) = phi(f(g(fn, arg, val), arg))`

    Together, these components form an algebraic system for reasoning about function
    arguments. Each operation (phi, psi, f, g) transforms a function into a new
    function in a way that changes the set of arguments that have been applied, and we
    have precise rules for understanding those transformations.

    Args:
        fun: A function composed of raw functions `f` or partial functions as
            constructed in the description.

    Returns:
        A set of arguments that were set by the partial.
    """
    args_set_by_partial: Set[str] = set()

    def _rec_unroll_partial_function_args(g: Union[Callable, partial]) -> Callable:
        """
        Store arguments and recursively call itself if the function is a partial. In the
        end, return the initial wrapped function. Besides partial functions it also
        supports `partial(fn_accept_additional_argument, *args, **kwargs)` constructs.

        Args:
            g: A partial or a function to unroll.

        Returns:
            Initial wrapped function.
        """
        nonlocal args_set_by_partial

        if isinstance(g, partial) and g.func == fn_accept_additional_argument:
            arg = g.keywords["arg"]
            if arg in args_set_by_partial:
                args_set_by_partial.remove(arg)
            return _rec_unroll_partial_function_args(g.keywords["fn"])
        elif isinstance(g, partial):
            args_set_by_partial.update(g.keywords.keys())
            args_set_by_partial.update(g.args)
            return _rec_unroll_partial_function_args(g.func)
        else:
            return g

    wrapped_fn = _rec_unroll_partial_function_args(fun)
    sig = inspect.signature(wrapped_fn)
    return args_set_by_partial | set(sig.parameters.keys())
