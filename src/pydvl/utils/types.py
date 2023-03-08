""" This module contains types, protocols, decorators and generic function
transformations. Some of it probably belongs elsewhere.
"""
import inspect
from typing import Callable, Protocol, Type

from numpy.typing import NDArray

__all__ = ["SupervisedModel"]


class SupervisedModel(Protocol):
    """This is the minimal Protocol that valuation methods require from
    models in order to work.

    All that is needed are the standard sklearn methods `fit()`, `predict()` and
    `score()`.
    """

    def fit(self, x: NDArray, y: NDArray):
        pass

    def predict(self, x: NDArray) -> NDArray:
        pass

    def score(self, x: NDArray, y: NDArray) -> float:
        pass


def maybe_add_argument(fun: Callable, new_arg: str):
    """Wraps a function to accept the given keyword parameter if it doesn't
    already.

    If `fun` already takes a keyword parameter of name `new_arg`, then it is
    returned as is. Otherwise, a wrapper is returned which merely ignores the
    argument.

    :param fun: The function to wrap
    :param new_arg: The name of the argument that the new function will accept
        (and ignore).
    :return: A new function accepting one more keyword argument.
    """
    params = inspect.signature(fun).parameters
    if new_arg in params.keys():
        return fun

    def wrapper(*args, **kwargs):
        try:
            del kwargs[new_arg]
        except KeyError:
            pass
        return fun(*args, **kwargs)

    return wrapper
