from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier

from pydvl.utils.caching import CacheBackend, CachedFuncConfig
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.scorers import KNNScorer
from pydvl.valuation.types import Sample
from pydvl.valuation.utility import ModelUtility

__all__ = ["KNNUtility"]


class KNNUtility(ModelUtility[Sample, KNeighborsClassifier]):
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
        scorer = KNNScorer(test_data)

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
