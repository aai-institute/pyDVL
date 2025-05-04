"""
This module defines the utility used by class-wise Shapley valuation methods.

See [the documentation][classwise-shapley-intro] for more information.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from pydvl.utils.array import Array
from pydvl.utils.caching import CacheBackend, CachedFuncConfig
from pydvl.valuation.scorers.classwise import ClasswiseSupervisedScorer
from pydvl.valuation.types import ClasswiseSample, IndexT, SupervisedModel
from pydvl.valuation.utility import ModelUtility

__all__ = ["ClasswiseModelUtility"]


class ClasswiseModelUtility(ModelUtility[ClasswiseSample, SupervisedModel]):
    """ModelUtility class that is specific to class-wise shapley valuation.

    It expects a class-wise scorer and a classification task.

    Args:
        model: Any supervised model. Typical choices can be found in the [sci-kit learn
            documentation](https://scikit-learn.org/stable/supervised_learning.html).
        scorer: A class-wise scoring object.
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
    """

    def __init__(
        self,
        model: SupervisedModel,
        scorer: ClasswiseSupervisedScorer,
        *,
        catch_errors: bool = False,
        show_warnings: bool = False,
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
        if not isinstance(self.scorer, ClasswiseSupervisedScorer):
            raise ValueError("Scorer must be an instance of ClasswiseSupervisedScorer")
        self.scorer: ClasswiseSupervisedScorer

    def _utility(self, sample: ClasswiseSample) -> float:
        # We override this method here because we have to:
        #   - set the label on the scorer
        #   - combine the in-class and out-of-class subsets
        self.scorer.label = sample.label
        new_sample = sample.with_subset(
            cast(
                Array[IndexT],
                np.union1d(sample.subset, sample.ooc_subset).astype(np.int_),
            )
        )
        return super()._utility(new_sample)
