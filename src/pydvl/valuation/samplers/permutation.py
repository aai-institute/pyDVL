r"""
Permutation-based samplers.

TODO: explain the formulation and the different samplers.


## References

[^1]: <a name="mitchell_sampling_2022"></a>Mitchell, Rory, Joshua Cooper, Eibe
      Frank, and Geoffrey Holmes. [Sampling Permutations for Shapley Value
      Estimation](https://jmlr.org/papers/v23/21-0439.html). Journal of Machine
      Learning Research 23, no. 43 (2022): 1–46.
[^2]: <a name="watson_accelerated_2023"></a>Watson, Lauren, Zeno Kujawa, Rayna Andreeva,
      Hao-Tsung Yang, Tariq Elahi, and Rik Sarkar. [Accelerated Shapley Value
      Approximation for Data Evaluation](https://doi.org/10.48550/arXiv.2311.05346).
      arXiv, 9 November 2023.
"""

from __future__ import annotations

import logging
import math
from copy import copy
from itertools import permutations
from typing import Callable, cast

import numpy as np

from pydvl.utils.types import Seed
from pydvl.valuation.samplers.base import EvaluationStrategy, IndexSampler
from pydvl.valuation.samplers.truncation import NoTruncation, TruncationPolicy
from pydvl.valuation.samplers.utils import StochasticSamplerMixin
from pydvl.valuation.types import (
    IndexSetT,
    IndexT,
    NullaryPredicate,
    Sample,
    SampleBatch,
    SampleGenerator,
    ValueUpdate,
)
from pydvl.valuation.utility.base import UtilityBase

__all__ = [
    "PermutationSampler",
    "AntitheticPermutationSampler",
    "DeterministicPermutationSampler",
    "PermutationEvaluationStrategy",
]


logger = logging.getLogger(__name__)


class PermutationSampler(StochasticSamplerMixin, IndexSampler):
    """Sample permutations of indices and iterate through each returning
    increasing subsets, as required for the permutation definition of
    semi-values.

    For a permutation `(3,1,4,2)`, this sampler returns in sequence the following
    [Samples][pydvl.valuation.samplers.Sample] (tuples of index and subset):

    `(3, {3})`, `(1, {3,1})`, `(4, {3,1,4})` and `(2, {3,1,4,2})`.

    !!! info "Batching"
        PermutationSamplers always batch their outputs to include a whole permutation
        of the index set, i.e. the batch size is always the number of indices.

    Args:
        truncation: A policy to stop the permutation early.
        seed: Seed for the random number generator.
    """

    def __init__(
        self, truncation: TruncationPolicy | None = None, seed: Seed | None = None
    ):
        super().__init__(seed=seed)
        self.truncation = truncation or NoTruncation()

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        """Generates the permutation samples.

        Samples are yielded one by one, not as whole permutations. These are batched
        together by calling iter() on the sampler.

        Args:
            indices:
        """
        if len(indices) == 0:
            return
        while True:
            yield Sample(-1, self._rng.permutation(indices))

    @staticmethod
    def weight(n: int, subset_len: int) -> float:
        return n * math.comb(n - 1, subset_len) if n > 0 else 1.0

    def make_strategy(
        self,
        utility: UtilityBase,
        coefficient: Callable[[int, int], float] | None = None,
    ) -> PermutationEvaluationStrategy:
        return PermutationEvaluationStrategy(self, utility, coefficient)


class AntitheticPermutationSampler(PermutationSampler):
    """Samples permutations like
    [PermutationSampler][pydvl.valuation.samplers.PermutationSampler], but after
    each permutation, it returns the same permutation in reverse order.

    This sampler was suggested in (Mitchell et al. 2022)<sup><a
    href="#mitchell_sampling_2022">1</a></sup>

    !!! tip "New in version 0.7.1"
    """

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        while True:
            permutation = self._rng.permutation(indices)
            yield Sample(-1, permutation)
            yield Sample(-1, permutation[::-1])


class DeterministicPermutationSampler(PermutationSampler):
    """Samples all n! permutations of the indices deterministically, and
    iterates through them, returning sets as required for the permutation-based
    definition of semi-values.
    """

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        for permutation in permutations(indices):
            yield Sample(-1, np.array(permutation, copy=False))

    def length(self, indices: IndexSetT) -> int:
        return math.factorial(len(indices))


class PermutationEvaluationStrategy(EvaluationStrategy[PermutationSampler]):
    """Computes marginal values for permutation sampling schemes.

    This strategy iterates over permutations from left to right, computing the marginal
    utility wrt. the previous one at each step to save computation.
    """

    def __init__(
        self,
        sampler: PermutationSampler,
        utility: UtilityBase,
        coefficient: Callable[[int, int], float] | None = None,
    ):
        super().__init__(sampler, utility, coefficient)
        self.truncation = copy(sampler.truncation)
        self.truncation.reset(utility)  # Perform initial setup (e.g. total_utility)

    def process(
        self, batch: SampleBatch, is_interrupted: NullaryPredicate
    ) -> list[ValueUpdate]:
        self.truncation.reset(self.utility)  # Reset before every batch (must be cached)
        r = []
        for sample in batch:
            truncated = False
            curr = prev = self.utility(None)
            permutation = sample.subset
            for i, idx in enumerate(permutation):
                # FIXME: type checker claims this could be Any (?)
                idx = cast(IndexT, idx)
                if not truncated:
                    curr = self.utility(Sample(idx, permutation[: i + 1]))
                marginal = curr - prev
                marginal *= self.coefficient(self.n_indices, i + 1)
                r.append(ValueUpdate(idx, marginal))
                prev = curr
                if not truncated and self.truncation(idx, curr, self.n_indices):
                    truncated = True
                if is_interrupted():
                    break
        return r
