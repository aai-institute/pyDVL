""" This module contains types, protocols, decorators and generic function
transformations. Some of it probably belongs elsewhere.
"""
import inspect
from enum import Enum
from typing import Callable, Optional, Protocol, Type, Union

from numpy import ndarray
from sklearn.metrics import get_scorer

__all__ = ["SupervisedModel", "Scorer", "compose_score"]


class SupervisedModel(Protocol):
    """This is the minimal Protocol that valuation methods require from
    models in order to work.

    All that is needed are the standard sklearn methods `fit()`, `predict()` and
    `score()`.
    """

    def fit(self, x: ndarray, y: ndarray):
        pass

    def predict(self, x: ndarray) -> ndarray:
        pass

    def score(self, x: ndarray, y: ndarray) -> float:
        pass


Scorer = Callable[[SupervisedModel, ndarray, ndarray], float]


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


# FIXME: This probably should be somewhere else
def compose_score(
    score: Union[str, Scorer],
    transformation: Callable[[float], float],
    name: str = None,
):
    """Composes a scoring function with an arbitrary scalar transformation.

    Useful to squash unbounded scores into ranges manageable by data valuation
    methods.

    .. code-block:: python
       :caption: Example usage

       sigmoid = lambda x: 1/(1+np.exp(-x))
       compose_score("r2", sigmoid, "squashed r2")

    :param score: Either a callable or a string naming any of sklearn's scorers
    :param transformation: A scalar transformation
    :param name: A string representation for the composition, for `str()`.

    :return: The function composition.
    """
    scoring_function: Scorer = get_scorer(score) if isinstance(score, str) else score

    class NewScorer(object):
        def __init__(self, scorer: Scorer, name: Optional[str] = None):
            self._scorer = scorer
            self._name = name or "Composite " + getattr(
                self._scorer, "__name__", "scorer"
            )

        def __call__(self, *args, **kwargs):
            score = self._scorer(*args, **kwargs)
            return transformation(score)

        def __str__(self):
            return self._name

        def __repr__(self):
            capitalized_name = "".join(s.capitalize() for s in self._name.split(" "))
            return f"{capitalized_name} (scorer={self._scorer})"

    return NewScorer(scoring_function, name=name)
