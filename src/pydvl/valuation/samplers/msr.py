from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
from numpy.typing import NDArray

from pydvl.utils.numeric import random_subset
from pydvl.utils.types import Seed
from pydvl.valuation.samplers.base import EvaluationStrategy, IndexSampler, SamplerT
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


class MSRSampler(StochasticSamplerMixin, IndexSampler):
    """Sampler for unweighted Maximum Sample Re-use (MSR) valuation.

    This is similar to a UniformSampler without an outer index.

    Args:
        batch_size: Number of samples to generate in each batch.
        seed: Seed for the random number generator.

    """

    def __init__(self, batch_size: int = 1, seed: Seed | None = None):
        super().__init__(batch_size=batch_size, seed=seed)

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        while True:
            subset = random_subset(indices, seed=self._rng)
            yield Sample(None, subset)

    @staticmethod
    def weight(n: int, subset_len: int) -> float:
        return 1.0

    def make_strategy(
        self,
        utility: UtilityBase,
        coefficient: Callable[[int, int, float], float] | None = None,
    ) -> MSREvaluationStrategy:
        return MSREvaluationStrategy(self, utility, coefficient)


class MSREvaluationStrategy(EvaluationStrategy[SamplerT, MSRValueUpdate]):
    """Evaluation strategy for Maximum Sample Re-use (MSR) valuation.

    The MSR evaluation strategy makes one utility evaluation per sample but generates
    `n_indices` many updates from it. The updates will be used to update two running
    means that will later be combined into a final value. We send the
    `ValueUpdate.kind` field to `ValueUpdateKind.POSITIVE` or `ValueUpdateKind.NEGATIVE`
    to decide which of the two running means is going to be updated.
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
        mask: NDArray[np.bool_] = np.zeros(self.n_indices, dtype=bool)
        mask[sample.subset] = True

        updates = []
        for i, m in enumerate(mask):
            updates.append(MSRValueUpdate(idx=i, update=u_value, is_positive=m))
        return updates
