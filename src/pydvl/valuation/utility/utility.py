from __future__ import annotations

import logging
import warnings
from typing import cast

import numpy as np
from sklearn.base import clone

from pydvl.utils.caching import CacheBackend, CachedFuncConfig, CacheStats
from pydvl.utils.types import SupervisedModel
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.scorers.supervised import SupervisedScorer
from pydvl.valuation.types import Sample, SampleT

__all__ = ["Utility"]

from pydvl.valuation.utility.base import UtilityBase

logger = logging.getLogger(__name__)


# Need a generic because subclasses might use subtypes of Sample
class Utility(UtilityBase[SampleT]):
    """Convenience wrapper with configurable memoization of the scoring
    function.

    An instance of `Utility` holds the triple of model, dataset and scoring
    function which determines the value of data points. This is used for the
    computation of [all game-theoretic values][game-theoretical-methods] like
    [Shapley values][pydvl.valuation.shapley] and [the Least
    Core][pydvl.valuation.least_core].

    The Utility expect the model to fulfill the [SupervisedModel][pydvl.utils.types.SupervisedModel]
    interface i.e. to have a `fit()` method.

    When calling the utility, the model will be
    [cloned](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html)
    if it is a Sci-Kit Learn model, otherwise a copy is created using
    [copy.deepcopy][]

    Since evaluating the scoring function requires retraining the model and that
    can be time-consuming, this class wraps it and caches the results of each
    execution. Caching is available both locally and across nodes, but must
    always be enabled for your project first, see [the documentation][getting-started-cache]
    and the [module documentation][pydvl.utils.caching].

    Attributes:
        model: The supervised model.
        scorer: A scoring function. If None, the `score()` method of the model
            will be used. See [score][pydvl.utils.score] for ways to create
            and compose scorers, in particular how to set default values and
            ranges.

    Args:
        model: Any supervised model. Typical choices can be found in the
            [sci-kit learn documentation][https://scikit-learn.org/stable/supervised_learning.html].
        scorer: A scoring object. If None, the `score()` method of the model
            will be used. See [scorers][pydvl.valuation.scorers] for ways to create
            and compose scorers, in particular how to set default values and
            ranges. For convenience, a string can be passed, which will be used
            to construct a [SupervisedScorer][pydvl.valuation.scorers.SupervisedScorer].
        catch_errors: set to `True` to catch the errors when `fit()` fails. This
            could happen in several steps of the pipeline, e.g. when too little
            training data is passed, which happens often during Shapley value
            calculations. When this happens, the [scorer's default
            value][pydvl.valuation.scorers.SupervisedScorer] is returned as a score and
            computation continues.
        show_warnings: Set to `False` to suppress warnings thrown by `fit()`.
        cache_backend: Optional instance of [CacheBackend][pydvl.utils.caching.base.CacheBackend]
            used to wrap the _utility method of the Utility instance.
            By default, this is set to None and that means that the utility evaluations
            will not be cached.
        cached_func_options: Optional configuration object for cached utility evaluation.
        clone_before_fit: If `True`, the model will be cloned before calling
            `fit()`.

    ??? Example
        ``` pycon
        >>> from pydvl.utils import Utility, DataUtilityLearning, Dataset
        >>> from sklearn.linear_model import LinearRegression, LogisticRegression
        >>> from sklearn.datasets import load_iris
        >>> dataset = Dataset.from_sklearn(load_iris(), random_state=16)
        >>> u = Utility(LogisticRegression(random_state=16), dataset)
        >>> u(Sample(subset=dataset.indices))
        0.9
        ```

        With caching enabled:

        ```pycon
        >>> from pydvl.utils import Utility, DataUtilityLearning, Dataset
        >>> from pydvl.utils.caching.memory import InMemoryCacheBackend
        >>> from sklearn.linear_model import LinearRegression, LogisticRegression
        >>> from sklearn.datasets import load_iris
        >>> dataset = Dataset.from_sklearn(load_iris(), random_state=16)
        >>> cache_backend = InMemoryCacheBackend()
        >>> u = Utility(LogisticRegression(random_state=16), dataset, cache_backend=cache_backend)
        >>> u(Sample(subset=dataset.indices))
        0.9
        ```

    """

    model: SupervisedModel
    scorer: SupervisedScorer

    def __init__(
        self,
        model: SupervisedModel,
        scorer: SupervisedScorer,
        *,
        catch_errors: bool = True,
        show_warnings: bool = False,
        cache_backend: CacheBackend | None = None,
        cached_func_options: CachedFuncConfig | None = None,
        clone_before_fit: bool = True,
    ):
        self.model = self._clone_model(model)
        self.scorer = scorer
        self.clone_before_fit = clone_before_fit
        self.catch_errors = catch_errors
        self.show_warnings = show_warnings
        self.cache = cache_backend
        if cached_func_options is None:
            cached_func_options = CachedFuncConfig()
        # TODO: Find a better way to do this.
        if cached_func_options.hash_prefix is None:
            # FIX: This does not handle reusing the same across runs.
            cached_func_options.hash_prefix = str(hash((model, scorer)))
        self.cached_func_options = cached_func_options
        self._initialize_utility_wrapper()

    def with_dataset(self, dataset: Dataset):
        copy = type(self)(
            model=self.model,
            scorer=self.scorer,
            catch_errors=self.catch_errors,
            show_warnings=self.show_warnings,
            cache_backend=self.cache,
            cached_func_options=self.cached_func_options,
            clone_before_fit=self.clone_before_fit,
        )
        copy.training_data = dataset
        return copy

    def _initialize_utility_wrapper(self):
        if self.cache is not None:
            self._utility_wrapper = self.cache.wrap(
                self._utility, config=self.cached_func_options
            )
        else:
            self._utility_wrapper = self._utility

    def __call__(self, sample: SampleT) -> float:
        """
        Args:
            sample: contains a subset of valid indices for the
                `x_train` attribute of [Dataset][pydvl.utils.dataset.Dataset].
        """
        return cast(float, self._utility_wrapper(sample))

    def _utility(self, sample: SampleT) -> float:
        """Clones the model, fits it on a subset of the training data
        and scores it on the test data.

        If an instance of [CacheBackend][pydvl.utils.caching.base.CacheBackend]
        is passed during construction, results are
        memoized to avoid duplicate computation. This is useful in particular
        when computing utilities of permutations of indices or when randomly
        sampling from the powerset of indices.

        Args:
            sample: contains a subset of valid indices for the
                `x_train` attribute of [Dataset][pydvl.utils.dataset.Dataset].

        Returns:
            0 if no indices are passed, `scorer.default` if we fail to fit the
                model or the scorer returns [numpy.NaN][]. Otherwise, the score
                of the model.
        """
        if len(sample.subset) == 0:
            return self.scorer.default

        if self.training_data is None:
            raise ValueError("No training data provided")

        x_train, y_train = self.training_data.get_data(sample.subset)

        with warnings.catch_warnings():
            if not self.show_warnings:
                warnings.simplefilter("ignore")
            try:
                if self.clone_before_fit:
                    model = self._clone_model(self.model)
                else:
                    model = self.model
                model.fit(x_train, y_train)
                score = float(self.scorer(model))
                # Some scorers raise exceptions if they return NaNs, some might not
                if np.isnan(score):
                    warnings.warn("Scorer returned NaN", RuntimeWarning)
                    return self.scorer.default
                return score
            except Exception as e:
                if self.catch_errors:
                    warnings.warn(str(e), RuntimeWarning)
                    return self.scorer.default
                raise

    @staticmethod
    def _clone_model(model: SupervisedModel) -> SupervisedModel:
        """Clones the passed model to avoid the possibility
        of reusing a fitted estimator

        Args:
            model: Any supervised model. Typical choices can be found
                on [this page](https://scikit-learn.org/stable/supervised_learning.html)
        """
        try:
            model = clone(model)
        except TypeError:
            # This happens if the passed model is not an sklearn model
            # In this case, we just make a deepcopy of the model.
            model = clone(model, safe=False)
        return cast(SupervisedModel, model)

    @property
    def cache_stats(self) -> CacheStats | None:
        """Cache statistics are gathered when cache is enabled.
        See [CacheStats][pydvl.utils.caching.base.CacheStats] for all fields returned.
        """
        cache_stats: CacheStats | None = None
        if self.cache is not None:
            cache_stats = self._utility_wrapper.stats
        return cache_stats

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle _utility_wrapper
        state.pop("_utility_wrapper", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add _utility_wrapper back since it doesn't exist in the pickle
        self._initialize_utility_wrapper()
