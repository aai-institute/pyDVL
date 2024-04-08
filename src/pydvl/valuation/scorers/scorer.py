"""
This module provides a [Scorer][pydvl.utils.score.Scorer] class that wraps
scoring functions with additional information.

Scorers are the fundamental building block of many data valuation methods. They
are typically used by the [Utility][pydvl.valuation.utility.Utility] class to
evaluate the quality of a model when trained on subsets of the training data.

Scorers can be constructed in the same way as in scikit-learn: either from 
known strings or from a callable. Greater values must be better. If they are not,
a negated version can be used, see scikit-learn's
[make_scorer()](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).

[Scorer][pydvl.utils.score.Scorer] provides additional information about the
scoring function, like its range and default values, which can be used by some
data valuation methods (like
[group_testing_shapley()][pydvl.valuation.methods.gt_shapley.group_testing_shapley]) to
estimate the number of samples required for a certain quality of approximation.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import get_scorer

from pydvl.utils.types import SupervisedModel

__all__ = ["Scorer", "ScorerCallable"]


class ScorerCallable(Protocol):
    """Signature for a scorer"""

    def __call__(
        self, model: SupervisedModel, X: NDArray[Any], y: NDArray[Any]
    ) -> float:
        ...


class Scorer:
    """A scoring callable that takes a model, data, and labels and returns a
    scalar.

    Args:
        scoring: Either a string or callable that can be passed to
            [get_scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.get_scorer.html).
        default: score to be used when a model cannot be fit, e.g. when too
            little data is passed, or errors arise.
        range: numerical range of the score function. Some Monte Carlo
            methods can use this to estimate the number of samples required for a
            certain quality of approximation. If not provided, it can be read from
            the `scoring` object if it provides it, for instance if it was
            constructed with [compose_score()][pydvl.utils.score.compose_score].
        name: The name of the scorer. If not provided, the name of the
            function passed will be used.

    !!! tip "New in version 0.5.0"

    """

    _name: str
    range: NDArray[np.float_]

    def __init__(
        self,
        scoring: str | ScorerCallable,
        default: float = np.nan,
        range: tuple[float, float] = (-np.inf, np.inf),
        name: str | None = None,
    ):
        if name is None and isinstance(scoring, str):
            name = scoring
        self._scorer = get_scorer(scoring)
        self.default = default
        # TODO: auto-fill from known scorers ?
        self.range = np.array(range)
        self._name = getattr(self._scorer, "__name__", name or "scorer")

    def __call__(
        self, model: SupervisedModel, X: NDArray[Any], y: NDArray[Any]
    ) -> float:
        return self._scorer(model, X, y)  # type: ignore

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        capitalized_name = "".join(s.capitalize() for s in self._name.split(" "))
        return f"{capitalized_name} (scorer={self._scorer})"
