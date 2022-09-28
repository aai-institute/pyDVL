import logging
import warnings
from typing import Iterable, Optional

from sklearn.metrics import check_scoring

from .caching import memcached, serialize
from .config import MemcachedConfig
from .dataset import Dataset
from .types import Scorer, SupervisedModel

__all__ = ["Utility"]

logger = logging.getLogger(__name__)


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
        show_warnings: bool = True,
        default_score: float = 0,
        enable_cache: bool = True,
        cache_options: Optional[MemcachedConfig] = None,
    ):
        """
        It holds all the most important elements of the Shapley values calculation,
        namely the model, the data and the scoring.
        It can also cache the training results, which speeds up
        the overall calculation for big models that take a long time to train.

        :param model: Any supervised model. Typical choices can be found at
            https://scikit-learn.org/stable/supervised_learning.html
        :param data: dataset or grouped dataset.
        :param scoring: Same as in sklearn's `cross_validate()`: a string,
            a scorer callable or None for the default `model.score()`. Greater
            values must be better. If they are not, a negated version can be
            used (see `make_scorer`)
        :param catch_errors: set to True to catch the errors when fit() fails. This
            could happen in several steps of the pipeline, e.g. when too little
            training data is passed, which happens often during the Shapley value calculations.
            When this happens, the default_score is returned as a score and Shapley value
            calculation continues.
        :param show_warnings: True for printing warnings fit fails.
            Used only when catch_errors is True
        :param default_score: score in the case of models that have not been fit,
            e.g. when too little data is passed, or errors arise.
        :param enable_cache: whether to use memcached for memoization.
        """
        self.model = model
        self.data = data
        self.scoring = scoring
        self.catch_errors = catch_errors
        self.show_warnings = show_warnings
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
                if self.show_warnings:
                    warnings.warn(str(e), RuntimeWarning)
                return self.default_score
            else:
                raise e

    @property
    def signature(self):
        """Signature used for caching model results"""
        return self._signature
