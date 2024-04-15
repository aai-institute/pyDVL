"""
This module provides a
[SupervisedScorer][pydvl.valuation.scorer.supervised.SupervisedScorer] class that wraps
scoring functions for supervised problems with additional information.

Supervised scorers can be constructed in the same way as in scikit-learn: either from
known strings or from a callable. Greater values must be better. If they are not,
a negated version can be used, see scikit-learn's
[make_scorer()](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).

[SupervisedScorer][pydvl.valuation.scorers.SupervisedScorer] holds the test data used to
evaluate the model.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import get_scorer

from pydvl.utils.types import SupervisedModel

__all__ = ["SupervisedScorer", "SupervisedScorerCallable"]

from pydvl.valuation.dataset import Dataset
from pydvl.valuation.scorers.base import Scorer


class SupervisedScorerCallable(Protocol):
    """Signature for a scorer"""

    def __call__(
        self, model: SupervisedModel, X: NDArray[Any], y: NDArray[Any]
    ) -> float:
        ...


class SupervisedScorer(Scorer):
    """A scoring callable that takes a model, data, and labels and returns a
    scalar.

    Args:
        scoring: Either a string or callable that can be passed to
            [get_scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.get_scorer.html).
        test_data: Dataset where the score will be evaluated.
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

    !!! tip "Changed in version 0.10.0"
        This is now `SupervisedScorer` and holds the test data used to evaluate the
        model.

    """

    _scorer: SupervisedScorerCallable

    def __init__(
        self,
        scoring: str | SupervisedScorerCallable | SupervisedModel,
        test_data: Dataset,
        default: float,
        range: tuple[float, float] = (-np.inf, np.inf),
        name: str | None = None,
    ):
        super().__init__()
        if isinstance(scoring, SupervisedModel):
            from sklearn.metrics import check_scoring

            self._scorer = check_scoring(scoring)
            if name is None:
                name = f"Default scorer for {scoring.__class__.__name__}"
        elif isinstance(scoring, str):
            self._scorer = get_scorer(scoring)
            if name is None:
                name = scoring
        else:
            self._scorer = scoring
            if name is None:
                name = getattr(scoring, "__name__", "scorer")
        self.test_data = test_data
        self.default = default
        # TODO: auto-fill from known scorers ?
        self.range = np.array(range, dtype=np.float_)
        self.name = name

    def __call__(self, model: SupervisedModel) -> float:
        return self._scorer(model, self.test_data.x, self.test_data.y)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        capitalized_name = "".join(s.capitalize() for s in self.name.split(" "))
        return f"{capitalized_name} (scorer={self._scorer})"
