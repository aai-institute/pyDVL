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
from abc import ABC
from copy import copy
from itertools import permutations

import numpy as np

from pydvl.utils.functional import suppress_warnings
from pydvl.utils.numeric import logcomb
from pydvl.utils.types import Seed
from pydvl.valuation.samplers.base import (
    EvaluationStrategy,
    IndexSampler,
    )
from pydvl.valuation.samplers.truncation import NoTruncation, TruncationPolicy
from pydvl.valuation.samplers.utils import StochasticSamplerMixin
from pydvl.valuation.types import (
    IndexSetT,
    NullaryPredicate,
    Sample,
    SampleBatch,
    SampleGenerator,
    SemivalueCoefficient,
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


class PermutationSamplerBase(IndexSampler, ABC):
    """Base class for permutation samplers."""

    def __init__(
        self,
        *args,
        truncation: TruncationPolicy | None = None,
        batch_size: int = 1,
        **kwargs,
    ):
        super().__init__(batch_size=batch_size)
        self.truncation = truncation or NoTruncation()

    def log_weight(self, n: int, subset_len: int) -> float:
        r"""Log probability of sampling a set S from a set of size n.

        Returns $p(S) = p(S|k) p(k)$, where $p(S|k)$ is the probability of sampling a
        set of size k. See [the module's
        documentation][pydvl.valuation.samplers.permutation] for details.
        """
        if n > 0:
            return float(-np.log(n) - logcomb(n, subset_len))
        return 0.0

    def make_strategy(
        self,
        utility: UtilityBase,
        coefficient: SemivalueCoefficient | None = None,
    ) -> PermutationEvaluationStrategy:
        return PermutationEvaluationStrategy(self, utility, coefficient)


class PermutationSampler(StochasticSamplerMixin, PermutationSamplerBase):
    """Samples permutations of indices.

    !!! info "Batching"
        Even though this sampler supports batching, it is not recommended to use it
        since the
        [PermutationEvaluationStrategy][pydvl.valuation.samplers.permutation.PermutationEvaluationStrategy]
        processes whole permutations in one go, effectively batching the computation of
        up to n-1 marginal utilities in one process.

    Args:
        truncation: A policy to stop the permutation early.
        seed: Seed for the random number generator.
    """

    def __init__(
        self,
        truncation: TruncationPolicy | None = None,
        seed: Seed | None = None,
        batch_size: int = 1,
    ):
        super().__init__(seed=seed, truncation=truncation, batch_size=batch_size)

    @property
    def skip_indices(self) -> IndexSetT:
        return self._skip_indices

    @skip_indices.setter
    def skip_indices(self, indices: IndexSetT):
        """Sets the indices to skip when generating permutations. This can be used
        to avoid updating indices that have already converged, but can lead to biased
        estimates if not done carefully."""
        self._skip_indices = indices

    def generate(self, indices: IndexSetT) -> SampleGenerator:
        """Generates the permutation samples.

        Args:
            indices: The indices to sample from. If empty, no samples are generated. If
                [skip_indices][pydvl.valuation.samplers.base.IndexSampler.skip_indices]
                is set, these indices are removed from the set before generating the
                permutation.
        """
        if len(indices) == 0:
            return
        while True:
            _indices = np.setdiff1d(indices, self.skip_indices)
            yield Sample(None, self._rng.permutation(_indices))


class AntitheticPermutationSampler(PermutationSampler):
    """Samples permutations like
    [PermutationSampler][pydvl.valuation.samplers.PermutationSampler], but after
    each permutation, it returns the same permutation in reverse order.

    This sampler was suggested in (Mitchell et al. 2022)<sup><a
    href="#mitchell_sampling_2022">1</a></sup>

    !!! tip "New in version 0.7.1"
    """

    def generate(self, indices: IndexSetT) -> SampleGenerator:
        for sample in super().generate(indices):
            permutation = sample.subset
            yield Sample(None, permutation)
            yield Sample(None, permutation[::-1])


class DeterministicPermutationSampler(PermutationSamplerBase):
    """Samples all n! permutations of the indices deterministically, and
    iterates through them, returning sets as required for the permutation-based
    definition of semi-values.
    """

    def generate(self, indices: IndexSetT) -> SampleGenerator:
        for permutation in permutations(indices):
            yield Sample(None, np.asarray(permutation))

    def sample_limit(self, indices: IndexSetT) -> int:
        if len(indices) == 0:
            return 0
        return math.factorial(len(indices))


class PermutationEvaluationStrategy(
    EvaluationStrategy[PermutationSamplerBase, ValueUpdate]
):
    """Computes marginal values for permutation sampling schemes in log-space.

    This strategy iterates over permutations from left to right, computing the marginal
    utility wrt. the previous one at each step to save computation.
    """

    def __init__(
        self,
        sampler: PermutationSamplerBase,
        utility: UtilityBase,
        coefficient: SemivalueCoefficient | None = None,
    ):
        super().__init__(sampler, utility, coefficient)
        self.truncation = copy(sampler.truncation)
        self.truncation.reset(utility)  # Perform initial setup (e.g. total_utility)

    @suppress_warnings(categories=(RuntimeWarning,), flag="show_warnings")
    def process(
        self, batch: SampleBatch, is_interrupted: NullaryPredicate
    ) -> list[ValueUpdate]:
        r = []
        for sample in batch:
            self.truncation.reset(self.utility)
            truncated = False
            curr = prev = self.utility(None)
            permutation = sample.subset
            for i, idx in enumerate(permutation):  # type: int, np.int_
                if not truncated:
                    new_sample = sample.with_idx(idx).with_subset(permutation[: i + 1])
                    curr = self.utility(new_sample)
                marginal = curr - prev
                sign = np.sign(marginal)
                log_marginal = -np.inf if marginal == 0 else np.log(marginal * sign)
                log_marginal += self.valuation_coefficient(self.n_indices, i)
                # Note the -1, see the discussion in PermutationSamplerBase.log_weight
                log_marginal -= self.sampler_weight(self.n_indices - 1, i)
                r.append(ValueUpdate(idx, log_marginal, sign))
                prev = curr
                if not truncated and self.truncation(idx, curr, self.n_indices):
                    truncated = True
                if is_interrupted():
                    return r
        return r
