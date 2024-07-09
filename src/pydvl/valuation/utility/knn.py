from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier

from pydvl.utils.caching import CacheBackend, CachedFuncConfig
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.scorers import KNNClassifierScorer
from pydvl.valuation.types import Sample
from pydvl.valuation.utility import ModelUtility

__all__ = ["KNNClassifierUtility"]


class KNNClassifierUtility(ModelUtility[Sample, KNeighborsClassifier]):
    """Utility object for KNN Classifiers.

    The utility function is the likelihood of the true class given the model's
    prediction.

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
        clone_before_fit: bool = True,
    ):
        scorer = KNNClassifierScorer(test_data)

        self.test_data = test_data

        super().__init__(
            model=model,
            scorer=scorer,
            catch_errors=catch_errors,
            show_warnings=show_warnings,
            cache_backend=cache_backend,
            cached_func_options=cached_func_options,
            clone_before_fit=clone_before_fit,
        )
