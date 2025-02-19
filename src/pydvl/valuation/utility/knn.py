from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_is_fitted

from pydvl.utils.caching import CacheBackend, CachedFuncConfig
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.types import Sample, SampleT
from pydvl.valuation.utility import ModelUtility

__all__ = ["KNNClassifierUtility"]

from pydvl.valuation import Scorer
from pydvl.valuation.utility.modelutility import ModelT


class _DummyScorer(Scorer):
    default = 0.0

    def __call__(self, model) -> float:
        return np.nan


class KNNClassifierUtility(ModelUtility[Sample, KNeighborsClassifier]):
    """Utility object for KNN Classifiers.

    The utility function is the model's predicted probability for the true class.

    This works both as a Utility object for general game theoretic valuation methods and
    for specialized valuation methods for KNN classifiers.

    Args:
        model: A KNN classifier model.
        test_data: The test data to evaluate the model on.
        catch_errors: set to `True` to catch the errors when `fit()` fails. This
            could happen in several steps of the pipeline, e.g. when too little
            training data is passed, which happens often during Shapley value
            calculations. When this happens, the [scorer's default
            value][pydvl.valuation.scorers.SupervisedScorer] is returned as a score and
            computation continues.
        show_warnings: Set to `False` to suppress warnings thrown by `fit()`.
        cache_backend: Optional instance of [CacheBackend][
            pydvl.utils.caching.base.CacheBackend] used to wrap the _utility method of
            the Utility instance. By default, this is set to None and that means that
            the utility evaluations will not be cached.
        cached_func_options: Optional configuration object for cached utility
            evaluation.
        clone_before_fit: If `True`, the model will be cloned before calling
            `fit()`.

    """

    def __init__(
        self,
        model: KNeighborsClassifier,
        test_data: Dataset,
        *,
        catch_errors: bool = True,
        show_warnings: bool = False,
        cache_backend: CacheBackend | None = None,
        cached_func_options: CachedFuncConfig | None = None,
    ):
        self.test_data = test_data
        self.sorted_neighbors: NDArray[np.int_] | None = None
        dummy_scorer = _DummyScorer()

        super().__init__(
            model=model,
            scorer=dummy_scorer,  # not applicable
            catch_errors=catch_errors,
            show_warnings=show_warnings,
            cache_backend=cache_backend,
            cached_func_options=cached_func_options,
            clone_before_fit=False,  # ensure we don't clone the fitted model
        )

    def _utility(self, sample: SampleT) -> float:
        """

        Args:
            sample: contains a subset of valid indices for the
                `x` attribute of [Dataset][pydvl.valuation.dataset.Dataset].

        Returns:
            0 if no indices are passed, otherwise the KNN utility for the sample.
        """
        if self.training_data is None:
            raise ValueError("No training data provided")

        check_is_fitted(
            self.model,
            msg="The KNN model has to be fitted before calling the utility.",
        )

        _, y_train = self.training_data.data()
        x_test, y_test = self.test_data.data()
        n_neighbors = self.model.get_params()["n_neighbors"]

        if self.sorted_neighbors is None:
            self.sorted_neighbors = self.model.kneighbors(x_test, return_distance=False)

        # Labels of the (restricted) nearest neighbors to each test point
        nns_labels = np.full((len(x_test), n_neighbors), None)
        for i, neighbors in enumerate(self.sorted_neighbors):
            restricted_ns = neighbors[np.isin(neighbors, sample.subset)]
            nns_labels[i][: len(restricted_ns)] = y_train[restricted_ns]
        # Likelihood of the correct labels
        probs = np.asarray(nns_labels == y_test[:, None]).sum(axis=1) / n_neighbors
        return float(probs.mean())

    def _compute_score(self, model: ModelT) -> float:
        raise NotImplementedError("This method should not be called")
