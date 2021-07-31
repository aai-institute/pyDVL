import numpy as np

from typing import Iterable, Tuple
from sklearn.metrics import check_scoring
from valuation.utils.logging import _logger
from valuation.utils import Dataset, SupervisedModel, Scorer, maybe_progress,\
    memcached

__all__ = ['Utility', 'bootstrap_test_score']


class Utility:
    """ A convenience wrapper with configurable memoization """
    model: SupervisedModel
    data: Dataset
    scoring: Scorer

    def __init__(self, model: SupervisedModel, data: Dataset, scoring: Scorer,
                 catch_errors: bool = True, cache_size: int = 4096):
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
        :param cache_size: Number of invocations to memoize. Set to None or 0
            to disable.
        """
        self.model = model
        self.data = data
        self.scoring = scoring
        self.catch_errors = catch_errors

        if cache_size is not None and cache_size > 0:
            self._utility_wrapper = memcached()(self._utility)
        else:
            self._utility_wrapper = self._utility

    def __call__(self, indices: Iterable[int]) -> float:
        return self._utility_wrapper(frozenset(indices))

    def _utility(self, indices: frozenset) -> float:
        """ Fits the model on a subset of the training data and scores it on the
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
            return 0.0
        scorer = check_scoring(self.model, self.scoring)
        x = self.data.x_train[list(indices)]
        y = self.data.y_train[list(indices)]
        try:
            self.model.fit(x, y)
            return scorer(self.model, self.data.x_test, self.data.y_test)
        except Exception as e:
            if self.catch_errors:
                _logger.warning(str(e))
                return np.nan
            else:
                raise e


def bootstrap_test_score(u: Utility,
                         bootstrap_iterations: int,
                         progress: bool = False) \
        -> Tuple[float, float]:
    """ That. Here for lack of a better place. """
    scorer = check_scoring(u.model, u.scoring)
    _scores = []
    u.model.fit(u.data.x_train, u.data.y_train)
    n_test = len(u.data.x_test)
    for _ in maybe_progress(range(bootstrap_iterations), progress,
                            desc="Bootstrapping"):
        sample = np.random.randint(low=0, high=n_test, size=n_test)
        score = scorer(u.model, u.data.x_test[sample], u.data.y_test[sample])
        _scores.append(score)

    return float(np.mean(_scores)), float(np.std(_scores))
