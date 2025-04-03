"""
This module implements a utility function for supervised models.

[ModelUtility][pydvl.valuation.utility.modelutility.ModelUtility] holds a model and a
scorer. Each call to the utility will fit the model on a subset of the training data and
evaluate the scorer on the test data. It is used by all the valuation methods in
[pydvl.valuation][pydvl.valuation].

This class is geared towards sci-kit-learn models, but can be used with any object that
implements the [BaseModel][pydvl.utils.types.BaseModel] protocol, i.e. that has a
`fit()` method.

!!! danger "Errors are hidden by default"
    During semi-value computations, the utility can be evaluated on subsets that
    break the fitting process. For instance, a classifier might require at least two
    classes to fit, but the utility is sometimes evaluated on subsets with only one
    class. This will raise an error with most classifiers. To avoid this, we set by
    default `catch_errors=True` upon instantiation, which will catch the error and
    return the scorer's default value instead. While we show a warning to signal that
    something went wrong, this suppression can lead to unexpected results, so it is
    important to be aware of this setting and to set it to `False` when testing, or if
    you are sure that the utility will not be evaluated on problematic subsets.


## Examples

??? Example "Standard usage"
    The utility takes a model and a scorer and is passed to the valuation method. Here's
    the basic usage:

    ```python
    from joblib import parallel_config
    from pydvl.valuation import (
        Dataset, MinUpdates, ModelUtility, SupervisedScorer, TMCShapleyValuation
    )

    train, test = Dataset.from_arrays(X, y, ...)
    model = SomeModel()  # Implementing the basic scikit-learn interface
    scorer =  SupervisedScorer("r2", test, default=0.0, range=(-np.inf, 1.0))
    utility = ModelUtility(model, scorer, catch_errors=True, show_warnings=True)
    valuation = TMCShapleyValuation(utility, is_done=MinUpdates(1000))
    with parallel_config(n_jobs=-1):
        valuation.fit(train)
    ```

??? Example "Directly calling the utility"
    The following code instantiates a utility object and calls it directly. The
    underlying logistic regression model will be trained on the indices passed as
    argument, and evaluated on the test data.

    ```python
    from pydvl.valuation.utility import ModelUtility
    from pydvl.valuation.dataset import Dataset
    from pydvl.valuation.scorers import SupervisedScorer
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.datasets import load_iris

    train, test = Dataset.from_sklearn(load_iris(), random_state=16)
    scorer =  SupervisedScorer("accuracy", test, default=0.0, range=(0.0, 1.0))
    u = ModelUtility(LogisticRegression(random_state=16), scorer, catch_errors=True)
    u(Sample(None, subset=train.indices))
    ```

??? Example "Enabling the cache"
    In this example an in-memory cache is used. Note that caching is only useful
    under certain conditions, and does not really speed typical Monte Carlo
    approximations. See [the introduction][#getting-started-cache] and the [module
    documentation][pydvl.utils.caching] for more.

    ```python
    (...)  # Imports as above
    cache_backend = InMemoryCacheBackend()  # See other backends in the caching module
    u = ModelUtility(
            model=LogisticRegression(random_state=16),
            scorer=SupervisedScorer("accuracy", test, default=0.0, range=(0.0, 1.0)),
            cache_backend=cache_backend,
            catch_errors=True
        )
    u(Sample(None, subset=train.indices))
    u(Sample(None, subset=train.indices))  # The second call does not retrain the model
    ```

## Data type of the underlying data arrays

In principle, very few to no assumptions are made about the data type. As long as it is
contained in a [Dataset][pydvl.valuation.dataset.Dataset] object, it should work. If
your data needs special handling before being fed to the model from the `Dataset`, you
can override the
[sample_to_data()][pydvl.valuation.utility.modelutility.ModelUtility.sample_to_data]
method. Be sure not to rely on the data being static for this. If you need to transform
it before fitting, then override
[with_dataset()][pydvl.valuation.utility.base.UtilityBase.with_dataset].

!!! warning "Caveats with parallel computation"
    When running in parallel, the utility is copied to each worker, which implies
    copying the dataset as well, which can obviously be very expensive. In order to
    alleviate the problem, one can memmap the data to disk. Alas, automatic memmapping
    by joblib does not work for nested structures like
    [Dataset][pydvl.valuation.dataset.Dataset] objects, nor for pytorch tensors. For
    now, it should be possible to [use memmap
    manually](https://joblib.readthedocs.io/en/latest/auto_examples/parallel_memmap.html)
    but it hasn't been tested.

    If you are working on a cluster, the data will be copied to each worker. In this
    case, subclassing of `Dataset` and `Utility` will be necessary to minimize copying,
    and the solution will depend on your storage solution. Feel free to open an issue if
    you need help with this.
"""

from __future__ import annotations

import logging
import warnings
from typing import Generic, TypeVar, cast

import numpy as np
from sklearn.base import clone

from pydvl.utils.caching import CacheBackend, CachedFuncConfig, CacheStats
from pydvl.utils.functional import suppress_warnings
from pydvl.utils.types import BaseModel
from pydvl.valuation.scorers import Scorer
from pydvl.valuation.types import SampleT

__all__ = ["ModelUtility"]

from pydvl.valuation.utility.base import UtilityBase

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


# Need a generic because subclasses might use subtypes of Sample
class ModelUtility(UtilityBase[SampleT], Generic[SampleT, ModelT]):
    """Convenience wrapper with configurable memoization of the utility.

    An instance of `ModelUtility` holds the tuple of model, and scoring function which
    determines the value of data points. This is used for the computation of [all
    game-theoretic values][game-theoretical-methods] like [Shapley
    values][pydvl.valuation.methods.shapley] and [the Least
    Core][pydvl.valuation.methods.least_core].

    `ModelUtility` expects the model to fulfill at least the
    [BaseModel][pydvl.utils.types.BaseModel] interface, i.e. to have a `fit()` method

    When calling the utility, the model will be
    [cloned](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html)
    if it is a Scikit-Learn model, otherwise a copy is created using [copy.deepcopy][]

    Since evaluating the scoring function requires retraining the model and that can be
    time-consuming, this class wraps it and caches the results of each execution.
    Caching is available both locally and across nodes, but must always be enabled for
    your project first, because **most stochastic methods do not** benefit much from it.
    See [the documentation][getting-started-cache] and the [module
    documentation][pydvl.utils.caching].

    Attributes:
        model: The supervised model.
        scorer: A scoring function. If None, the `score()` method of the model
            will be used. See [score][pydvl.utils.score] for ways to create
            and compose scorers, in particular how to set default values and
            ranges.

    Args:
        model: Any supervised model. Typical choices can be found in the
            [sci-kit learn documentation](https://scikit-learn.org/stable/supervised_learning.html).
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
            used to memoize results to avoid duplicate computation. Note however, that
            for most stochastic methods, cache hits are rare, making the memory expense
            of caching not worth it (YMMV).
        cached_func_options: Optional configuration object for cached utility evaluation.
        clone_before_fit: If `True`, the model will be cloned before calling
            `fit()`.
    """

    model: ModelT
    scorer: Scorer

    def __init__(
        self,
        model: ModelT,
        scorer: Scorer,
        *,
        catch_errors: bool = True,
        show_warnings: bool = True,
        cache_backend: CacheBackend | None = None,
        cached_func_options: CachedFuncConfig | None = None,
        clone_before_fit: bool = True,
    ):
        self.clone_before_fit = clone_before_fit
        self.model = self._maybe_clone_model(model, clone_before_fit)
        self.scorer = scorer
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

    def _initialize_utility_wrapper(self):
        if self.cache is not None:
            self._utility_wrapper = self.cache.wrap(
                self._utility, config=self.cached_func_options
            )
        else:
            self._utility_wrapper = self._utility

    def __call__(self, sample: SampleT | None) -> float:
        """
        Args:
            sample: contains a subset of valid indices for the
                `x_train` attribute of [Dataset][pydvl.utils.dataset.Dataset].
        """
        if sample is None or len(sample.subset) == 0:
            return self.scorer.default

        return cast(float, self._utility_wrapper(sample))

    def _compute_score(self, model: ModelT) -> float:
        """Computes the score of a fitted model.

        Args:
            model: fitted model
        Returns:
            Computed score or the scorer's default value in case of an error
            or a NaN value.
        """
        try:
            score = float(self.scorer(model))
            # Some scorers raise exceptions if they return NaNs, some might not
            if np.isnan(score):
                warnings.warn("Scorer returned NaN", RuntimeWarning)
                return self.scorer.default
        except Exception as e:
            if self.catch_errors:
                warnings.warn(str(e), RuntimeWarning)
                return self.scorer.default
            raise
        return score

    def sample_to_data(self, sample: SampleT) -> tuple:
        """Returns the raw data corresponding to a sample.

        Subclasses can override this e.g. to do reshaping of tensors. Be careful not to
        rely on `self.training_data` not changing between calls to this method. For
        manipulations to it, use the `with_dataset()` method.

        Args:
            sample: contains a subset of valid indices for the
                `x_train` attribute of [Dataset][pydvl.utils.dataset.Dataset].
        Returns:
            Tuple of the training data and labels corresponding to the sample indices.
        """
        if self.training_data is None:
            raise ValueError("No training data provided")

        x_train, y_train = self.training_data.data(sample.subset)
        return x_train, y_train

    @suppress_warnings(flag="show_warnings")
    def _utility(self, sample: SampleT) -> float:
        """Clones the model, fits it on a subset of the training data
        and scores it on the test data.

        Args:
            sample: contains a subset of valid indices for the
                `x` attribute of [Dataset][pydvl.valuation.dataset.Dataset].

        Returns:
            0 if no indices are passed, `scorer.default` if we fail to fit the
                model or the scorer returns [numpy.nan][]. Otherwise, the score
                of the model.
        """
        x_train, y_train = self.sample_to_data(sample)

        try:
            model = self._maybe_clone_model(self.model, self.clone_before_fit)
            model.fit(x_train, y_train)
            score = self._compute_score(model)
            return score
        except Exception as e:
            if self.catch_errors:
                warnings.warn(str(e), RuntimeWarning)
                return self.scorer.default
            raise

    @staticmethod
    def _maybe_clone_model(model: ModelT, do_clone: bool) -> ModelT:
        """Clones the passed model to avoid the possibility of reusing a fitted
        estimator.

        Args:
            model: Any supervised model. Typical choices can be found
                on [this page](https://scikit-learn.org/stable/supervised_learning.html)
            do_clone: Whether to clone the model or not.
        """
        if not do_clone:
            return model
        try:
            model = clone(model)
        except TypeError:
            # This happens if the passed model is not an sklearn model
            # In this case, we just make a deepcopy of the model.
            model = clone(model, safe=False)
        return cast(ModelT, model)

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
