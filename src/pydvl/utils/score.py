"""
This module provides a :class:`Scorer` class that wraps scoring functions with
additional information.

Scorers can be constructed in the same way as in scikit-learn: either from 
known strings or from a callable. Greater values must be better. If they are not,
a negated version can be used (see scikit-learn's `make_scorer() 
<https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html>_`)

:class:`Scorer` provides additional information about the scoring function, like
its range and default values.
"""
from typing import Callable, Optional, Protocol, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit
from sklearn.metrics import get_scorer

from pydvl.utils.types import SupervisedModel


class ScorerCallable(Protocol):
    """Signature for a scorer"""

    def __call__(self, model: SupervisedModel, X: NDArray, y: NDArray) -> float:
        ...


class Scorer:
    """A scoring callable that takes a model, data, and labels and returns a
    scalar.

    :param scoring: Either a string or callable that can be passed to
        `get_scorer <>`_.
    :param default: score to be used when a model cannot be fit, e.g. when too
        little data is passed, or errors arise.
    :param range: numerical range of the score function. Some Monte Carlo
        methods can use this to estimate the number of samples required for a
        certain quality of approximation. If not provided, it can be read from
        the ``scoring`` object if it provides it, for instance if it was
        constructed with :func:`~pydvl.utils.types.compose_score`.
    :param name: The name of the scorer. If not provided, the name of the
        function passed will be used.
    """

    _name: str
    range: NDArray[np.float_]

    def __init__(
        self,
        scoring: Union[str, ScorerCallable],
        default: float = np.nan,
        range: Tuple = (-np.inf, np.inf),
        name: Optional[str] = None,
    ):
        self._scorer = get_scorer(scoring)
        self.default = default
        # TODO: auto-fill from known scorers ?
        self.range = np.array(range)
        self._name = getattr(self._scorer, "__name__", name or "scorer")

    def __call__(self, model: SupervisedModel, X: NDArray, y: NDArray) -> float:
        return self._scorer(model, X, y)  # type: ignore

    def __str__(self):
        return self._name

    def __repr__(self):
        capitalized_name = "".join(s.capitalize() for s in self._name.split(" "))
        return f"{capitalized_name} (scorer={self._scorer})"


def compose_score(
    scorer: Scorer,
    transformation: Callable[[float], float],
    range: Tuple[float, float],
    name: str,
) -> Scorer:
    """Composes a scoring function with an arbitrary scalar transformation.

    Useful to squash unbounded scores into ranges manageable by data valuation
    methods.

    .. code-block:: python
       :caption: Example usage

       sigmoid = lambda x: 1/(1+np.exp(-x))
       compose_score(Scorer("r2"), sigmoid, range=(0,1), name="squashed r2")

    :param scorer: The object to be composed.
    :param transformation: A scalar transformation
    :param range: The range of the transformation. This will be used e.g. by
        :class:`~pydvl.utils.utility.Utility` for the range of the composed.
    :param name: A string representation for the composition, for `str()`.
    :return: The composite :class:`Scorer`.
    """

    class NewScorer(Scorer):
        def __call__(self, model: SupervisedModel, X: NDArray, y: NDArray) -> float:
            score = self._scorer(model=model, X=X, y=y)
            return transformation(score)

    return NewScorer(scorer, range=range, name=name)


def sigmoid(x: float) -> float:
    result: float = expit(x).item()
    return result


squashed_r2 = compose_score(Scorer("r2"), sigmoid, (0, 1), "squashed r2")
squashed_variance = compose_score(
    Scorer("explained_variance"), sigmoid, (0, 1), "squashed explained variance"
)
