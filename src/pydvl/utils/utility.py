"""
This module contains classes to manage and learn utility functions for the
computation of values. Please see the documentation on :ref:`data valuation` for
more information.

:class:`Utility` holds information about model, data and scoring function (the
latter being what one usually understands under *utility* in the general
definition of Shapley value). It is automatically cached across machines.

:class:`DataUtilityLearning` adds support for learning the scoring function
to avoid repeated re-training of the model to compute the score.

This module also contains Utility classes for toy games that are used
for testing and for demonstration purposes.

"""
import logging
import warnings
from typing import Dict, FrozenSet, Iterable, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.metrics import check_scoring

from pydvl.utils import Dataset
from pydvl.utils.caching import CacheStats, memcached, serialize
from pydvl.utils.config import MemcachedConfig
from pydvl.utils.score import Scorer
from pydvl.utils.types import SupervisedModel

__all__ = ["Utility", "DataUtilityLearning", "MinerGameUtility", "GlovesGameUtility"]

logger = logging.getLogger(__name__)


class Utility:
    """Convenience wrapper with configurable memoization of the scoring
    function.

    An instance of ``Utility`` holds the triple of model, dataset and scoring
    function which determines the value of data points. This is mosly used for
    the computation of :ref:`Shapley values<Shapley>` and
    :ref:`Least Core values<Least Core>`.

    The Utility expect the model to fulfill
    the :class:`pydvl.utils.types.SupervisedModel` interface i.e. to have
    ``fit()``, ``predict()``, and ``score()`` methods.

    When calling the utility, the model will be
    `cloned <https://scikit-learn.org/stable/modules/generated/sklearn.base
    .clone.html>`_
    if it is a Sci-Kit Learn model, otherwise a copy is created using
    ``deepcopy()``
    from the builtin `copy <https://docs.python.org/3/library/copy.html>`_
    module.

    Since evaluating the scoring function requires retraining the model
    and that can be time-consuming, this class wraps it and caches
    the results of each execution. Caching is available both locally
    and across nodes, but must always be enabled for your
    project first, see :ref:`how to set up the cache<caching setup>`.

    :param model: Any supervised model. Typical choices can be found at
            https://scikit-learn.org/stable/supervised_learning.html
    :param data: :class:`Dataset` or :class:`GroupedDataset`.
    :param scorer: A scoring object. If None, the ``score()`` method of the model
        will be used. See :mod:`~pydvl.utils.scorer` for ways to create
        and compose scorers, in particular how to set default values and ranges.
        For convenience, a string can be passed, which will be used to construct
        a :class:`~pydvl.utils.scorer.Scorer`.
    :param default_score: As a convenience when no ``scorer`` object is passed
        (where a default value can be provided), this argument also allows to set
        the default score for models that have not been fit, e.g. when too little
        data is passed, or errors arise.
    :param score_range: As with ``default_score``, this is a convenience argument
        for when no ``scorer`` argument is provided, to set the numerical range
        of the score function. Some Monte Carlo methods can use this to estimate
        the number of samples required for a certain quality of approximation.
    :param catch_errors: set to ``True`` to catch the errors when fit() fails.
        This could happen in several steps of the pipeline, e.g. when too little
        training data is passed, which happens often during Shapley value
        calculations. When this happens, the :attr:`default_score` is returned
        as a score and computation continues.
    :param show_warnings: Set to ``False`` to suppress warnings thrown by
        ``fit()``.
    :param enable_cache: If ``True``, use memcached for memoization.
    :param cache_options: Optional configuration object for memcached.
    :param clone_before_fit: If True, the model will be cloned before calling
        ``fit()``.

    :Example:

    >>> from pydvl.utils import Utility, DataUtilityLearning, Dataset
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from sklearn.datasets import load_iris
    >>> dataset = Dataset.from_sklearn(load_iris(), random_state=16)
    >>> u = Utility(LogisticRegression(random_state=16), dataset)
    >>> u(dataset.indices)
    0.9

    """

    model: SupervisedModel
    data: Dataset
    scorer: Scorer

    def __init__(
        self,
        model: SupervisedModel,
        data: Dataset,
        scorer: Optional[Union[str, Scorer]] = None,
        *,
        default_score: float = 0.0,
        score_range: Tuple[float, float] = (-np.inf, np.inf),
        catch_errors: bool = True,
        show_warnings: bool = False,
        enable_cache: bool = False,
        cache_options: Optional[MemcachedConfig] = None,
        clone_before_fit: bool = True,
    ):
        self.model = self._clone_model(model)
        self.data = data
        if isinstance(scorer, str):
            scorer = Scorer(scorer, default=default_score, range=score_range)
        self.scorer = check_scoring(self.model, scorer)
        self.default_score = scorer.default if scorer is not None else default_score
        # TODO: auto-fill from known scorers ?
        self.score_range = scorer.range if scorer is not None else np.array(score_range)
        self.catch_errors = catch_errors
        self.show_warnings = show_warnings
        self.enable_cache = enable_cache
        self.cache_options: MemcachedConfig = cache_options or MemcachedConfig()
        self.clone_before_fit = clone_before_fit
        self._signature = serialize((hash(self.model), hash(data), hash(scorer)))
        self._initialize_utility_wrapper()

        # FIXME: can't modify docstring of methods. Instead, I could use a
        #  factory which creates the class on the fly with the right doc.
        # self.__call__.__doc__ = self._utility_wrapper.__doc__

    def _initialize_utility_wrapper(self):
        if self.enable_cache:
            self._utility_wrapper = memcached(**self.cache_options)(  # type: ignore
                self._utility, signature=self._signature
            )
        else:
            self._utility_wrapper = self._utility

    def __call__(self, indices: Iterable[int]) -> float:
        utility: float = self._utility_wrapper(frozenset(indices))
        return utility

    def _utility(self, indices: FrozenSet) -> float:
        """Clones the model, fits it on a subset of the training data
        and scores it on the test data.

        If the object is constructed with ``enable_cache = True``, results are
        memoized to avoid duplicate computation. This is useful in particular
        when computing utilities of permutations of indices or when randomly
        sampling from the powerset of indices.

        :param indices: a subset of valid indices for
            :attr:`~pydvl.utils.dataset.Dataset.x_train`. The type must be
            hashable for the caching to work, e.g. wrap the argument with
            `frozenset <https://docs.python.org/3/library/stdtypes.html#frozenset>`_
            (rather than `tuple` since order should not matter)
        :return: 0 if no indices are passed, :attr:`default_score`` if we fail
            to fit the model or the scorer returns `NaN`. Otherwise, the score
            of the model on the test data.
        """
        if not indices:
            return 0.0

        x_train, y_train = self.data.get_training_data(list(indices))
        x_test, y_test = self.data.get_test_data(list(indices))

        with warnings.catch_warnings():
            if not self.show_warnings:
                warnings.simplefilter("ignore")
            try:
                if self.clone_before_fit:
                    model = self._clone_model(self.model)
                else:
                    model = self.model
                model.fit(x_train, y_train)
                score = float(self.scorer(model, x_test, y_test))
                # Some scorers raise exceptions if they return NaNs, some might not
                if np.isnan(score):
                    warnings.warn("Scorer returned NaN", RuntimeWarning)
                    return self.default_score
                return score
            except Exception as e:
                if self.catch_errors:
                    warnings.warn(str(e), RuntimeWarning)
                    return self.default_score
                raise

    @staticmethod
    def _clone_model(model: SupervisedModel) -> SupervisedModel:
        """Clones the passed model to avoid the possibility
        of reusing a fitted estimator

        :param model: Any supervised model. Typical choices can be found at
            https://scikit-learn.org/stable/supervised_learning.html
        """
        try:
            model = clone(model)
        except TypeError:
            # This happens if the passed model is not an sklearn model
            # In this case, we just make a deepcopy of the model.
            model = clone(model, safe=False)
        model = cast(SupervisedModel, model)
        return model

    @property
    def signature(self):
        """Signature used for caching model results."""
        return self._signature

    @property
    def cache_stats(self) -> Optional[CacheStats]:
        """Cache statistics are gathered when cache is enabled.
        See :class:`~pydvl.utils.caching.CacheInfo` for all fields returned.
        """
        if self.enable_cache:
            return self._utility_wrapper.stats  # type: ignore
        return None

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle _utility_wrapper
        state.pop("_utility_wrapper", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add _utility_wrapper back since it doesn't exist in the pickle
        self._initialize_utility_wrapper()


class DataUtilityLearning:
    """Implementation of Data Utility Learning algorithm
    :footcite:t:`wang_improving_2022`.

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
    >>> u = Utility(LogisticRegression(), dataset)
    >>> wrapped_u = DataUtilityLearning(u, 3, LinearRegression())
    ... # First 3 calls will be computed normally
    >>> for i in range(3):
    ...     _ = wrapped_u((i,))
    >>> wrapped_u((1, 2, 3)) # Subsequent calls will be computed using the fit model for DUL
    0.0

    """

    def __init__(
        self, u: Utility, training_budget: int, model: SupervisedModel
    ) -> None:
        self.utility = u
        self.training_budget = training_budget
        self.model = model
        self._current_iteration = 0
        self._is_model_fit = False
        self._utility_samples: Dict[FrozenSet, Tuple[NDArray[np.bool_], float]] = {}

    def _convert_indices_to_boolean_vector(self, x: Iterable[int]) -> NDArray[np.bool_]:
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


class MinerGameUtility(Utility):
    r"""Toy game utility that is used for testing and demonstration purposes.

    Consider a group of n miners, who have discovered large bars of gold.

    If two miners can carry one piece of gold,
    then the payoff of a coalition $S$ is:

    $${
    v(S) = \left\{\begin{array}{lll}
    \mid S \mid / 2 & \text{, if} & \mid S \mid \text{ is even} \\
    ( \mid S \mid - 1)/2 & \text{, if} & \mid S \mid \text{ is odd}
    \end{array}\right.
    }$$

    If there are more than two miners and there is an even number of miners,
    then the core consists of the single payoff where each miner gets 1/2.

    If there is an odd number of miners, then the core is empty.

    Taken from: https://en.wikipedia.org/wiki/Core_(game_theory)

    :param n_miners: Number of miners that participate in the game.
    """

    def __init__(self, n_miners: int, **kwargs):
        if n_miners <= 2:
            raise ValueError(f"n_miners, {n_miners} should be > 2")
        self.n_miners = n_miners

        x = np.arange(n_miners)[..., np.newaxis]
        # The y values don't matter here
        y = np.zeros_like(x)

        self.data = Dataset(x_train=x, y_train=y, x_test=x, y_test=y)

    def __call__(self, indices: Iterable[int]) -> float:
        n = len(tuple(indices))
        if n % 2 == 0:
            return n / 2
        else:
            return (n - 1) / 2

    def _initialize_utility_wrapper(self):
        pass

    def exact_least_core_values(self) -> Tuple[NDArray[np.float_], float]:
        if self.n_miners % 2 == 0:
            values = np.array([0.5] * self.n_miners)
            subsidy = 0.0
        else:
            values = np.array(
                [(self.n_miners - 1) / (2 * self.n_miners)] * self.n_miners
            )
            subsidy = (self.n_miners - 1) / (2 * self.n_miners)
        return values, subsidy

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n={self.n_miners})"


class GlovesGameUtility(Utility):
    r"""Toy game utility that is used for testing and demonstration purposes.

    In this game, some players have a left glove and others a right glove.
    Single gloves have a worth of zero while pairs have a worth of 1.

    The payoff of a coalition $S$ is:

    $${
    v(S) = \min( \mid S \cap L \mid, \mid S \cap R \mid )
    }$$

    Where $L$, respectively $R$, is the set of players with left gloves,
    respectively right gloves.

    :param left: Number of players with a left glove.
    :param right: Number of player with a right glove.

    """

    def __init__(self, left: int, right: int, **kwargs):
        self.left = left
        self.right = right

        x = np.empty(left + right)[..., np.newaxis]
        # The y values don't matter here
        y = np.zeros_like(x)

        self.data = Dataset(x_train=x, y_train=y, x_test=x, y_test=y)

    def __call__(self, indices: Iterable[int]) -> float:
        left_sum = float(np.sum(np.asarray(indices) < self.left))
        right_sum = float(np.sum(np.asarray(indices) >= self.left))
        return min(left_sum, right_sum)

    def _initialize_utility_wrapper(self):
        pass

    def exact_least_core_values(self) -> Tuple[NDArray[np.float_], float]:
        if self.left == self.right:
            subsidy = -0.5
            values = np.array([0.5] * (self.left + self.right))
        elif self.left < self.right:
            subsidy = 0.0
            values = np.array([1.0] * self.left + [0.0] * self.right)
        else:
            subsidy = 0.0
            values = np.array([0.0] * self.left + [1.0] * self.right)
        return values, subsidy

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(L={self.left}, R={self.right})"
