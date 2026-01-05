"""
This module implements a utility function for supervised models.

[ModelUtility][pydvl.valuation.utility.modelutility.ModelUtility] holds a model and a
scorer. Each call to the utility will fit the model on a subset of the training data and
evaluate the scorer on the test data. It is used by all the valuation methods in
[pydvl.valuation][pydvl.valuation].

This class is geared towards sci-kit-learn models, but can be used with any object that
implements the [BaseModel][pydvl.utils.types.BaseModel] protocol, i.e. that has a
`fit()` method.

## Incremental training with partial_fit

[PartialFitModelUtility][pydvl.valuation.utility.modelutility.PartialFitModelUtility]
extends ModelUtility to support models with `partial_fit` capability. This is particularly
beneficial for TMC Shapley and other permutation-based methods, where training data grows
incrementally. Instead of retraining from scratch for each subset, the utility uses
`partial_fit` to update the model incrementally, significantly reducing computation time
for compatible models (e.g., SGDClassifier, MLPClassifier, PassiveAggressiveClassifier).

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
    approximations. See [the introduction][getting-started-cache] and the [module
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

!!! warning "Data copying when running in parallel"
    When running in parallel, the utility and the dataset are copied to each worker. To
    avoid this, you can use `mmap=True` when constructing
    [Dataset][pydvl.valuation.dataset.Dataset]. Read [Working with large
    datasets][large-datasets-parallelization] for more information on the subject.
"""

from __future__ import annotations

import logging
import warnings
from typing import Generic, TypeVar, cast

from typing_extensions import Self

import numpy as np
from sklearn.base import clone

from pydvl.utils.caching import CacheBackend, CachedFuncConfig, CacheStats
from pydvl.utils.functional import suppress_warnings
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.scorers import Scorer
from pydvl.valuation.types import BaseModel, SampleT

__all__ = ["ModelUtility", "PartialFitModelUtility"]

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
            [sci-kit learn documentation](
            https://scikit-learn.org/stable/supervised_learning.html).
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
        cache_backend: Optional instance of [CacheBackend][
        pydvl.utils.caching.base.CacheBackend]
            used to memoize results to avoid duplicate computation. Note however, that
            for most stochastic methods, cache hits are rare, making the memory expense
            of caching not worth it (YMMV).
        cached_func_options: Optional configuration object for cached utility
        evaluation.
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
        self.full_fit_count = 0

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
        print("Utility")
        self.full_fit_count += 1
        print(f"Full fit count: {self.full_fit_count}")
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


class PartialFitModelUtility(ModelUtility[SampleT, ModelT]):
    """Model utility that supports incremental training with partial_fit.

    This utility extends ModelUtility to support models with partial_fit capability
    (e.g., SGDClassifier, MLPClassifier, MiniBatchKMeans). When used with permutation
    samplers in TMC Shapley, this allows for incremental training as data points are
    added sequentially, avoiding complete retraining for each subset.

    The utility automatically detects if the model supports partial_fit and uses it
    when appropriate. If partial_fit is not available, it falls back to regular fit().

    !!! info "How it works"
        When processing a permutation like [σ₁] → [σ₁, σ₂] → [σ₁, σ₂, σ₃], instead of:
        - Training from scratch on [σ₁]
        - Training from scratch on [σ₁, σ₂]
        - Training from scratch on [σ₁, σ₂, σ₃]

        We do:
        - Train on [σ₁]
        - partial_fit with [σ₂] to get model trained on [σ₁, σ₂]
        - partial_fit with [σ₃] to get model trained on [σ₁, σ₂, σ₃]

    !!! warning "Parallelization considerations"
        The partial_fit optimization works within a single permutation processing.
        When running in parallel, each worker processes its permutations independently.
        Since the state is maintained per-worker, this is safe.

    Args:
        model: Any supervised model. Models supporting partial_fit (like SGDClassifier,
            MLPClassifier) will benefit from incremental training.
        scorer: A scoring object.
        catch_errors: Set to `True` to catch errors when `fit()` or `partial_fit()` fails.
        show_warnings: Set to `False` to suppress warnings thrown by `fit()` or `partial_fit()`.
        cache_backend: Optional cache backend for memoization.
        cached_func_options: Optional configuration for cached utility evaluation.
        clone_before_fit: If `True`, the model will be cloned before calling `fit()`.
            Note: For partial_fit, we always clone to maintain state correctly.

    ??? Example "Usage with TMC Shapley"
        ```python
        from sklearn.linear_model import SGDClassifier
        from pydvl.valuation import (
            PartialFitModelUtility, SupervisedScorer, TMCShapleyValuation, Dataset
        )

        train, test = Dataset.from_arrays(X_train, y_train, X_test, y_test)
        model = SGDClassifier(random_state=42)
        scorer = SupervisedScorer("accuracy", test, default=0.0, range=(0.0, 1.0))
        utility = PartialFitModelUtility(model, scorer)
        valuation = TMCShapleyValuation(utility, is_done=MinUpdates(1000))
        valuation.fit(train)
        ```
    """

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
        super().__init__(
            model,
            scorer,
            catch_errors=catch_errors,
            show_warnings=show_warnings,
            cache_backend=cache_backend,
            cached_func_options=cached_func_options,
            clone_before_fit=clone_before_fit,
        )
        # State for partial_fit optimization
        self._current_model: ModelT | None = None
        self._current_indices: set[int] = set()
        self._supports_partial_fit = hasattr(model, "partial_fit")
        # Cache for unique classes, computed once per dataset
        self._classes: np.ndarray | None = None
        self.full_fit_count = 0
        self.partial_fit_count = 0

    def with_dataset(self, data: Dataset, copy: bool = True) -> Self:
        utility = super().with_dataset(data, copy)
        # Invalidate classes cache when dataset changes
        utility._classes = None
        return utility

    def reset_partial_fit_state(self):
        """Reset the partial_fit state for a new permutation.

        This should be called at the start of each permutation to ensure
        we start with a fresh model.
        """
        self._current_model = None
        self._current_indices = set()

    @suppress_warnings(flag="show_warnings")
    def _utility(self, sample: SampleT) -> float:
        """Fits or partially fits the model on a subset and scores it.

        This method tries to use partial_fit when possible for efficiency.
        If partial_fit is not applicable, it falls back to full fit().

        Args:
            sample: Contains indices for training.

        Returns:
            The score of the model, or scorer.default on error.
        """
        if sample is None or len(sample.subset) == 0:
            return self.scorer.default

        print("Partial fit utility")
        print(f"Partial fit count: {self.partial_fit_count}")
        print(f"Full fit count: {self.full_fit_count}")
        try:
            # Check if we can use partial_fit
            # Optimization: logic integrated to avoid double set creation
            use_partial = False
            x_new = None
            y_new = None
            new_indices = set()

            if self._supports_partial_fit and self._current_model is not None:
                # Fast check on lengths first to avoid set creation if obviously not adding points
                # Note: We assume only adding points (superset)
                if len(sample.subset) >= len(self._current_indices):
                    sample_set = set(sample.subset)
                    if self._current_indices.issubset(sample_set):
                        use_partial = True
                        new_indices = sample_set - self._current_indices
                        if len(new_indices) > 0:
                            if self.training_data is None:
                                raise ValueError("No training data provided")
                            new_indices_array = np.array(sorted(new_indices))
                            x_new, y_new = self.training_data.data(new_indices_array)

            if use_partial:
                # Incremental training with partial_fit
                print("Partial fit")
                self.partial_fit_count += 1
                # Only proceed with partial_fit if there are new points
                if x_new is not None and len(x_new) > 0:
                    # For classifiers, partial_fit may need classes parameter on first call
                    if hasattr(self._current_model, "classes_") or not hasattr(
                        self._current_model, "partial_fit"
                    ):
                        # Model already has classes or doesn't need them (e.g. regressor or already fitted)
                        self._current_model.partial_fit(x_new, y_new)
                    else:
                        # First partial_fit call for a classifier - need to provide classes
                        if self._classes is None:
                            if self.training_data is None:
                                raise ValueError("No training data provided")
                            _, y_all = self.training_data.data()
                            self._classes = np.unique(y_all)

                        self._current_model.partial_fit(x_new, y_new, classes=self._classes)

                    self._current_indices.update(new_indices)

                score = self._compute_score(self._current_model)
                return score
            else:
                print("Full fit")
                self.full_fit_count += 1
                # Full training from scratch
                x_train, y_train = self.sample_to_data(sample)
                model = self._maybe_clone_model(self.model, self.clone_before_fit)
                model.fit(x_train, y_train)

                # Update state for potential future partial_fit
                if self._supports_partial_fit:
                    self._current_model = model
                    self._current_indices = set(sample.subset)

                score = self._compute_score(model)
                return score

        except Exception as e:
            if self.catch_errors:
                warnings.warn(str(e), RuntimeWarning)
                # Reset state on error to avoid corrupted model
                self.reset_partial_fit_state()
                return self.scorer.default
            raise

    def __getstate__(self):
        state = super().__getstate__()
        # Don't pickle the current model state (it's worker-specific)
        state.pop("_current_model", None)
        state.pop("_current_indices", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        # Restore the partial_fit state attributes
        self._current_model = None
        self._current_indices = set()