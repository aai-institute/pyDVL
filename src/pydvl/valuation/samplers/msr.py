"""
This module implements Maximum Sample Re-use (MSR) sampling for valuation, as described
in (Wang et al.)<sup><a href="#wang_data_2023">1</a></sup>.

The idea behind MSR is to update all indices in the dataset with every evaluation of the
utility function on a sample. Updates are divided into positive, if the index is in the
sample, and negative, if it is not. The two running means are later combined into a
final result.

Note that this requires defining a special evaluation strategy and result updater, as
returned by the [make_strategy][pydvl.valuation.samplers.MSRSampler.make_strategy] and
[result_updater][pydvl.valuation.samplers.MSRSampler.result_updater] methods,
respectively.

## References

[^1]: <a name="wang_data_2023"></a>Wang, J.T. and Jia, R., 2023.
    [Data Banzhaf: A Robust Data Valuation Framework for Machine Learning](
    https://proceedings.mlr.press/v206/wang23e.html).
    In: Proceedings of The 26th International Conference on Artificial Intelligence and
    Statistics, pp. 6388-6421.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from pydvl.utils.numeric import random_subset
from pydvl.utils.types import Seed
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.samplers.base import (
    EvaluationStrategy,
    IndexSampler,
    ResultUpdater,
    SamplerT,
)
from pydvl.valuation.samplers.utils import StochasticSamplerMixin
from pydvl.valuation.types import (
    IndexSetT,
    NullaryPredicate,
    Sample,
    SampleBatch,
    SampleGenerator,
    ValueUpdate,
)
from pydvl.valuation.utility.base import UtilityBase

__all__ = ["MSRSampler"]


@dataclass(frozen=True)
class MSRValueUpdate(ValueUpdate):
    is_positive: bool


class MSRResultUpdater(ResultUpdater[MSRValueUpdate]):
    """Update running means for MSR valuation.

    This class is used to update two running means for positive and negative updates
    separately. The two running means are later combined into a final result.

    Since MSR-Banzhaf values are not a mean over marginals, both the variances of the
    marginals and the update counts are ill-defined. We use the following conventions:

    1. The counts are defined as the minimum of the two counts. This definition enables
    us to ensure a minimal number of updates for both running means via stopping
    criteria and correctly detects that no actual update has taken place if one of the
    counts is zero.

    2. We reverse engineer the variances so that they yield correct standard errors
    given our convention for the counts and the normal calculation of standard errors in
    the valuation result.

    Note that we cannot use the normal addition or subtraction defined by the
    [ValuationResult][pydvl.valuation.result.ValuationResult] because it is weighted
    with counts. If we were to simply subtract the negative result from the positive we
    would get wrong variance estimates, misleading update counts and even wrong values
    if no further precaution is taken.
    """

    def __init__(self, result: ValuationResult):
        self.result = result
        self.positive = ValuationResult.zeros(
            algorithm=result.algorithm, indices=result.indices, data_names=result.names
        )
        self.negative = ValuationResult.zeros(
            algorithm=result.algorithm, indices=result.indices, data_names=result.names
        )

    def __call__(self, update: MSRValueUpdate) -> ValuationResult:
        assert update.idx is not None
        if update.is_positive:
            self.positive.update(update.idx, update.update)
        else:
            self.negative.update(update.idx, update.update)
        return self.combine_results()

    def combine_results(self) -> ValuationResult:
        """Combine the positive and negative running means into a final result.
        Returns:
            The combined valuation result.

        TODO: Verify that the two running means are statistically independent (which is
            assumed in the aggregation of variances).

        """
        # define counts as minimum of the two counts (see docstring)
        counts = np.minimum(self.positive.counts, self.negative.counts)

        values = self.positive.values + self.negative.values
        values[counts == 0] = np.nan

        # define variances that yield correct standard errors (see docstring)
        pos_var = self.positive.variances / np.clip(self.positive.counts, 1, np.inf)
        neg_var = self.negative.variances / np.clip(self.negative.counts, 1, np.inf)
        variances = np.where(counts != 0, (pos_var + neg_var) * counts, np.inf)

        self.result = ValuationResult(
            values=values,
            variances=variances,
            counts=counts,
            indices=self.result.indices,
            data_names=self.result.names,
            algorithm=self.result.algorithm,
        )

        return self.result


class MSRSampler(StochasticSamplerMixin, IndexSampler[MSRValueUpdate]):
    """Sampler for unweighted Maximum Sample Re-use (MSR) valuation.

    The sampling is similar to a
    [UniformSampler][pydvl.valuation.samplers.UniformSampler] but without an outer
    index. However,the MSR sampler uses a special evaluation strategy and result updater,
    as returned by the [make_strategy][pydvl.valuation.samplers.MSRSampler.make_strategy]
    and [result_updater][pydvl.valuation.samplers.MSRSampler.result_updater] methods,
    respectively.

    Two running means are updated separately for positive and negative updates. The two
    running means are later combined into a final result.

    Args:
        batch_size: Number of samples to generate in each batch.
        seed: Seed for the random number generator.

    """

    def __init__(self, batch_size: int = 1, seed: Seed | None = None):
        super().__init__(batch_size=batch_size, seed=seed)
        self._count = 0

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        while True:
            self._count += 1
            subset = random_subset(indices, seed=self._rng)
            yield Sample(None, subset)

    def weight(self, n: int, subset_len: int) -> float:
        return 2 ** (n - 1) if n > 0 else 1.0  # type: ignore

    def make_strategy(
        self,
        utility: UtilityBase,
        coefficient: Callable[[int, int, float], float] | None = None,
    ) -> MSREvaluationStrategy:
        return MSREvaluationStrategy(self, utility, coefficient)

    def result_updater(self, result: ValuationResult) -> MSRResultUpdater:
        return MSRResultUpdater(result)


class MSREvaluationStrategy(EvaluationStrategy[SamplerT, MSRValueUpdate]):
    """Evaluation strategy for Maximum Sample Re-use (MSR) valuation.

    The MSR evaluation strategy makes one utility evaluation per sample but generates
    `n_indices` many updates from it. The updates will be used to update two running
    means that will later be combined into a final value. We use the field
    `ValueUpdate.is_positive` field to inform [MSRResultUpdater][pydvl.valuation.samplers.MSRResultUpdater]
     of which of the two running
    means must be updated.
    """

    def process(
        self, batch: SampleBatch, is_interrupted: NullaryPredicate
    ) -> List[MSRValueUpdate]:
        updates = []
        for sample in batch:
            updates.extend(self._process_sample(sample))
            if is_interrupted():
                break
        return updates

    def _process_sample(self, sample: Sample) -> List[MSRValueUpdate]:
        u_value = self.utility(sample)
        mask = np.zeros(self.n_indices, dtype=bool)
        mask[sample.subset] = True

        updates = []
        for i, m in enumerate(mask):
            k = len(sample.subset) - int(m)
            coeff = self.coefficient(self.n_indices, k)
            if not m:
                coeff = -coeff
            updates.append(
                MSRValueUpdate(idx=i, update=u_value * coeff, is_positive=bool(m))
            )
        return updates
