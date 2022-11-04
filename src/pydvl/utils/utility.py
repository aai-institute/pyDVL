"""
This module contains classes to manage and learn utility functions for the
computation of values. Please see the documentation on :ref:`data valuation` for
more information.

:class:`Utility` holds information about model, data and scoring function (which
is the utility* function itself for Shapley value). It is automatically cached
across machines.

:class:`DataUtilityLearning` adds support for learning the scoring function
to avoid repeated re-training of the model.

"""
import logging
import warnings
from typing import TYPE_CHECKING, Dict, FrozenSet, Iterable, Optional, Tuple

import numpy as np
from sklearn.metrics import check_scoring

from .caching import memcached, serialize
from .config import MemcachedConfig
from .dataset import Dataset
from .types import Scorer, SupervisedModel

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["Utility", "DataUtilityLearning"]

logger = logging.getLogger(__name__)


class Utility:
    """Convenience wrapper with configurable memoization of the scoring function.

    An instance of `Utility` holds the triple of model, dataset and scoring
    function which determines the value of data points. This is mostly used for
    the computation of :ref:`Shapley values<data valuation>`.

    Since evaluating the scoring function requires retraining the model, this
    class wraps it and caches the results of each execution. Caching is
    available both locally and across nodes, but must always be enabled for your
    project first, see :ref:`how to set up the cache<caching setup>`.

    :param model: Any supervised model. Typical choices can be found at
            https://scikit-learn.org/stable/supervised_learning.html
    :param data: :class:`Dataset` or :class:`GroupedDataset`.
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
    :param cache_options:
    """

    model: SupervisedModel
    data: Dataset
    scoring: Optional[Scorer]

    def __init__(
        self,
        model: SupervisedModel,
        data: Dataset,
        scoring: Optional[Scorer] = None,
        *,
        catch_errors: bool = True,
        show_warnings: bool = False,
        default_score: float = 0.0,
        enable_cache: bool = True,
        cache_options: Optional[MemcachedConfig] = None,
    ):
        self.model = model
        self.data = data
        self.scoring = scoring
        self.catch_errors = catch_errors
        self.show_warnings = show_warnings
        self.default_score = default_score
        self.enable_cache = enable_cache
        self.cache_options = cache_options
        self._signature = serialize((hash(model), hash(data), hash(scoring)))

        self._initialize_utility_wrapper()

        # FIXME: can't modify docstring of methods. Instead, I could use a
        #  factory which creates the class on the fly with the right doc.
        # self.__call__.__doc__ = self._utility_wrapper.__doc__

    def _initialize_utility_wrapper(self):
        if self.enable_cache:
            if self.cache_options is None:
                cache_options = dict()  # type: ignore
            else:
                cache_options = self.cache_options
            self._utility_wrapper = memcached(**cache_options)(  # type: ignore
                self._utility, signature=self._signature
            )
        else:
            self._utility_wrapper = self._utility

    def __call__(self, indices: Iterable[int]) -> float:
        utility: float = self._utility_wrapper(frozenset(indices))
        return utility

    def _utility(self, indices: FrozenSet) -> float:
        """Fits the model on a subset of the training data and scores it on the
        test data. If the object is constructed with `enable_cache = True`,
        results are memoized to avoid duplicate computation. This is useful in
        particular when computing utilities of permutations of indices or when
        randomly sampling from the powerset of indices.

        :param indices: a subset of valid indices for
            :attr:`~pydvl.utils.dataset.Dataset.x_train`. The type must be
            hashable for the caching to work, e.g. wrap the argument with
            `frozenset <https://docs.python.org/3/library/stdtypes.html#frozenset>`_
            (rather than `tuple` since order should not matter)
        :return: 0 if no indices are passed, `default_score` if we fail to fit
            the model or the scorer returns NaN, otherwise the score on the test
            data.
        """
        if not indices:
            return 0.0
        scorer = check_scoring(self.model, self.scoring)
        x, y = self.data.get_training_data(list(indices))
        try:
            self.model.fit(x, y)
            score = float(scorer(self.model, self.data.x_test, self.data.y_test))
            # Some scorers raise exceptions if they return NaNs, some might not
            if np.isnan(score):
                if self.show_warnings:
                    warnings.warn(f"Scorer returned NaN", RuntimeWarning)
                return self.default_score
            return score
        except Exception as e:
            if self.catch_errors:
                if self.show_warnings:
                    warnings.warn(str(e), RuntimeWarning)
                return self.default_score
            raise e

    @property
    def signature(self):
        """Signature used for caching model results."""
        return self._signature

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle _utility_wrapper
        del state["_utility_wrapper"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add _utility_wrapper back since it doesn't exist in the pickle
        self._initialize_utility_wrapper()


class DataUtilityLearning:
    """Implementation of Data Utility Learning algorithm [1]_.

    This object wraps a :class:`~pydvl.utils.utility.Utility` and delegates
    calls to it, up until a given budget (number of iterations). Every tuple
    of input and output (a so-called *utility sample*) is stored. Once the
    budget is exhausted, `DataUtilityLearning` fits the given model to the
    utility samples. Subsequent calls will use the learned model to predict the
    utility instead of delegating.

    :param u: The :class:`~pydvl.utils.utility.Utility` to learn.
    :param training_budget: Number of utility samples to collect before fitting
        the given model
    :param model: A supervised regression model

    :Example:

    >>> from pydvl.utils import Utility, DataUtilityLearning, Dataset
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from sklearn.datasets import load_iris
    >>> dataset = Dataset.from_sklearn(load_iris())
    >>> u = Utility(LogisticRegression(), dataset, enable_cache=False)
    >>> wrapped_u = DataUtilityLearning(u, 3, LinearRegression())
    ... # First 3 calls will be computed normally
    >>> for i in range(3):
    ...     _ = wrapped_u((i,))
    >>> wrapped_u((1, 2, 3)) # Subsequent calls will be computed using the fit model for DUL
    0.0

    .. note::
        .. [1] `Tianhao Wang, Yu Yang, Ruoxi Jia.
           "Improving Cooperative Game Theory-based Data Valuation via Data Utility Learning."
           arXiv, 2021 <https://arxiv.org/abs/2107.06336v2>`_.
    """

    def __init__(
        self, u: Utility, training_budget: int, model: SupervisedModel
    ) -> None:
        self.utility = u
        self.training_budget = training_budget
        self.model = model
        self._current_iteration = 0
        self._is_model_fit = False
        self._utility_samples: Dict[FrozenSet, Tuple["NDArray", float]] = {}

    def _convert_indices_to_boolean_vector(self, x: Iterable[int]) -> "NDArray":
        boolean_vector = np.zeros((1, len(self.utility.data)), dtype=bool)
        if x is not None:
            boolean_vector[:, tuple(x)] = True
        return boolean_vector

    def __call__(self, indices: Iterable[int]) -> float:
        indices_boolean_vector = self._convert_indices_to_boolean_vector(indices)
        frozen_indices = frozenset(indices)
        if self._current_iteration < self.training_budget:
            utility = self.utility(frozen_indices)
            self._utility_samples[frozen_indices] = (indices_boolean_vector, utility)
            self._current_iteration += 1
        else:
            if not self._is_model_fit:
                X, y = zip(*self._utility_samples.values())
                X = np.vstack(X)
                y = np.asarray(y)
                self.model.fit(X, y)
                self._is_model_fit = True
            if frozen_indices in self._utility_samples:
                utility = self._utility_samples[frozen_indices][1]
            else:
                utility = self.model.predict(indices_boolean_vector).item()
        return utility

    @property
    def data(self) -> Dataset:
        """Returns the wrapped utility's :class:`~pydvl.utils.dataset.Dataset`."""
        return self.utility.data
