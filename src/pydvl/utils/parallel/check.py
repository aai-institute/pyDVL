import inspect
from functools import partial
from typing import Callable


def check_fn_accepts_parameter(fn: Callable, param_name: str) -> bool:
    """
    Checks whether the given function accepts the given parameter or not.

    :param fn: The function to check.
    :param param_name: The name of the parameter to check.

    :return: True if the function accepts the parameter, False otherwise.
    """
    args_set_by_partial = set()

    # Recursive function to go through nested functools.partial objects
    def check_partial(g: Callable):
        nonlocal args_set_by_partial

        # If the function is a functools.partial, get the original function
        if isinstance(g, partial):
            args_set_by_partial.update(g.keywords.keys())
            args_set_by_partial.update(g.args)
            return check_partial(g.func)
        else:
            return g

    # Get the original function from functools.partial if needed
    original_function = check_partial(fn)

    sig = inspect.signature(original_function)
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
