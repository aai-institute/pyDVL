from __future__ import annotations

import numpy as np

from pydvl.utils.caching import CacheBackend, CachedFuncConfig
from pydvl.utils.types import SupervisedModel
from pydvl.valuation.scorers.classwise import ClasswiseSupervisedScorer
from pydvl.valuation.types import CSSample
from pydvl.valuation.utility import ModelUtility

__all__ = ["ClasswiseModelUtility"]


class ClasswiseModelUtility(ModelUtility[CSSample, SupervisedModel]):
    def __init__(
        self,
        model: SupervisedModel,
        scorer: ClasswiseSupervisedScorer,
        *,
        catch_errors: bool = True,
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

    def _compute_score(self, model: SupervisedModel, sample: CSSample) -> float:
        """Computes the score of a fitted model.

        Args:
            model: fitted model
            sample: contains a subset of valid indices for the
                `x` attribute of [Dataset][pydvl.valuation.dataset.Dataset].

        Returns:
            Computed score or the scorer's default value in case of an error
            or a NaN value.
        """
        self.scorer.with_label(sample.label)
        return super()._compute_score(model, sample)

    def _utility(self, sample: CSSample) -> float:
        new_sample = sample.with_subset(np.union1d(sample.subset, sample.ooc_subset))
        return super()._utility(new_sample)
