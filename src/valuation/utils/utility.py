from typing import Iterable, Optional

import numpy as np
from sklearn.metrics import check_scoring

from valuation.utils import Dataset, MemcachedConfig, Scorer, SupervisedModel, memcached
from valuation.utils.caching import serialize
from valuation.utils.logging import logger

__all__ = ["Utility"]


class Utility:
    """A convenience wrapper with configurable memoization"""

    model: SupervisedModel
    data: Dataset
    scoring: Optional[Scorer]

    def __init__(
        self,
        model: SupervisedModel,
        data: Dataset,
        scoring: Optional[Scorer] = None,
        catch_errors: bool = True,
        default_score: float = 0,
        enable_cache: bool = True,
        cache_options: Optional[MemcachedConfig] = None,
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
        self._signature = None

        if enable_cache:
            if cache_options is None:
                cache_options = dict()  # type: ignore
            self._signature = serialize((hash(model), hash(data), hash(scoring)))
            self._utility_wrapper = memcached(**cache_options)(  # type: ignore
                self._utility, signature=self._signature
            )
        else:
            self._utility_wrapper = self._utility

        # FIXME: can't modify docstring of methods. Instead, I could use a
        #  factory which creates the class on the fly with the right doc.
        # self.__call__.__doc__ = self._utility_wrapper.__doc__

    def __call__(self, indices: Iterable[int]) -> float:
        utility: float = self._utility_wrapper(frozenset(indices))
        return utility

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
        x, y = self.data.get_train_data(list(indices))
        try:
            self.model.fit(x, y)
            return float(scorer(self.model, self.data.x_test, self.data.y_test))
        except Exception as e:
            if self.catch_errors:
                logger.warning(str(e))  # type: ignore
                return self.default_score
            else:
                raise e

    @property
    def signature(self):
        return self._signature
