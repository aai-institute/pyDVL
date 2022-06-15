from typing import Iterable, Optional, Tuple

import numpy as np
from sklearn.metrics import check_scoring

from valuation.utils import (
    Dataset,
    MemcachedConfig,
    Scorer,
    SupervisedModel,
    maybe_progress,
    memcached,
)
from valuation.utils.logging import logger

__all__ = ["Utility", "bootstrap_test_score"]


class Utility:
    """A convenience wrapper with configurable memoization"""

    model: SupervisedModel
    data: Dataset
    scoring: Scorer

    def __init__(
        self,
        model: SupervisedModel,
        data: Dataset,
        scoring: Optional[Scorer],
        catch_errors: bool = True,
        default_score: float = 0,
        enable_cache: bool = True,
        cache_options: MemcachedConfig = None,
    ):
        """
        :param model: Any supervised model
        :param data: a split Dataset
        :param scoring: Same as in sklearn's `cross_validate()`: a string,
            a scorer callable or None for the default `model.score()`. Greater
            values must be better. If they are not, a negated version can be
            used (see `make_scorer`)
        :param catch_errors: set to True to return np.nan if fit() fails. This
            hack helps when a step in a pipeline fails if there are too few data
            points
        :param default_score: score in the case of models that have not been fit,
            e.g. when too little data is passed, or errors arise.
        :param enable_cache: whether to use memcached for memoization.
        """
        self.model = model
        self.data = data
        self.scoring = scoring
        self.catch_errors = catch_errors
        self.default_score = default_score

        if enable_cache:
            if cache_options is None:
                cache_options = dict()

            self._utility_wrapper = memcached(**cache_options)(self._utility)
        else:
            self._utility_wrapper = self._utility

        # FIXME: can't modify docstring of methods. Instead, I could use a
        #  factory which creates the class on the fly with the right doc.
        # self.__call__.__doc__ = self._utility_wrapper.__doc__

    def __call__(self, indices: Iterable[int]) -> float:
        return self._utility_wrapper(frozenset(indices))

    def _utility(self, indices: frozenset) -> float:
        """Fits the model on a subset of the training data and scores it on the
        test data. If the object is constructed with cache_size > 0, results are
        memoized to avoid duplicate computation. This is useful in particular
        when computing utilities of permutations of indices.

        :param indices: a subset of indices from data.x_train.index. The type
         must be hashable for the caching to work, e.g. wrap the argument with
         `frozenset` (rather than `tuple` since order should not matter)

        :return: 0 if no indices are passed, otherwise the value the scorer
        on the test data.
        """
        if not indices:
            return 0
        scorer = check_scoring(self.model, self.scoring)
        x = self.data.x_train[list(indices)]
        y = self.data.y_train[list(indices)]
        try:
            self.model.fit(x, y)
            return float(scorer(self.model, self.data.x_test, self.data.y_test))
        except Exception as e:
            if self.catch_errors:
                logger.warning(str(e))
                return self.default_score
            else:
                raise e


def bootstrap_test_score(
    u: Utility, bootstrap_iterations: int, progress: bool = False
) -> Tuple[float, float]:
    """That. Here for lack of a better place."""
    scorer = check_scoring(u.model, u.scoring)
    _scores = []
    u.model.fit(u.data.x_train, u.data.y_train)
    n_test = len(u.data.x_test)
    for _ in maybe_progress(
        range(bootstrap_iterations), progress, desc="Bootstrapping"
    ):
        sample = np.random.randint(low=0, high=n_test, size=n_test)
        score = scorer(u.model, u.data.x_test[sample], u.data.y_test[sample])
        _scores.append(score)

    return float(np.mean(_scores)), float(np.std(_scores))
