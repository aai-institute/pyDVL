"""
AMESampler implements the general AME sampling scheme.
For each index i, it first draws a probability p from a user-specified distribution
(with density f), then includes every other index independently with probability p.
The Monte Carlo correction is computed by numerically evaluating
    I(k) = ∫₀¹ f(p) p^k (1-p)^(n-1-k) dp,
so that the log_weight is given by
    log_weight(n, k) = -log(n) - log(I(k)).
For the uniform case (f(p)=1), one recovers the Shapley weighting:
    -log(n) - betaln(k+1, n-k).
"""

from __future__ import annotations

from typing import Callable, Type

import numpy as np
from scipy.integrate import quad

from pydvl.utils import Seed, complement, random_subset
from pydvl.valuation.samplers.base import EvaluationStrategy
from pydvl.valuation.samplers.powerset import (
    IndexIteration,
    PowersetSampler,
    SequentialIndexIteration,
)
from pydvl.valuation.samplers.utils import StochasticSamplerMixin
from pydvl.valuation.types import Distribution, IndexSetT, Sample, SampleGenerator
from pydvl.valuation.utility.base import UtilityBase


class UniformDistribution:
    def rvs(self):
        return np.random.default_rng().uniform()

    def pdf(self, p: float):
        return 1.0 if 0 <= p <= 1 else 0.0


class AMESampler(StochasticSamplerMixin, PowersetSampler):
    """
    AMESampler implements the two-stage AME sampling scheme.

    For each index i, it first samples a probability p from p_distribution and then
    includes each other index (from the complement) independently with probability p.

    Args:
        p_distribution: An object implementing the
            [Distribution][pydvl.valuation.types.Distribution]
            protocol.
            Defaults to using RNG.uniform(0,1) for sampling and constant density 1.
        batch_size: Number of samples per batch.
        index_iteration: Strategy to iterate over indices (default: SequentialIndexIteration).
        seed: Seed for RNG.
    """

    def __init__(
        self,
        p_distribution: Distribution | None = None,
        batch_size: int = 1,
        index_iteration: Type[IndexIteration] = SequentialIndexIteration,
        seed: Seed | None = None,
    ):
        super().__init__(
            batch_size=batch_size, index_iteration=index_iteration, seed=seed
        )
        self.p_distribution = p_distribution or UniformDistribution()

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        """
        For each index i, sample p ~ p_distribution and then generate a subset of the complement
        by including each element independently with probability p.
        """
        for idx in self.index_iterator(indices):
            comp = complement(indices, idx)
            p = self.p_distribution.rvs()
            subset = random_subset(comp, p, self._rng)
            yield Sample(idx, subset)

    def sample_limit(self, indices: IndexSetT) -> int | None:
        """
        The sample limit is determined by the outer loop over indices.
        """
        return self._index_iterator_cls.length(len(indices))

    def log_weight(self, n: int, subset_len: int) -> float:
        """Computes the log correction weight for a sample of a given subset size.

        For a given index i (with full dataset size n) and a sampled subset S of size k,
        let m = n - 1. The probability of obtaining S under the two-stage scheme is:

            P(S) = (1 / binom(m, k))  \int_0^1 f(p) p^k (1-p)^{m-k} dp.

        This correction ensures that when p_distribution is uniform over (0,1) (f(p)=1),
        we recover the Shapley weighting:

            -log(n) - betaln(k+1, n-k).
        """
        m = self._index_iterator_cls.complement_size(n)
        integrand = (
            lambda p: self.p_distribution.pdf(p)
            * (p**subset_len)
            * ((1 - p) ** (m - subset_len))
        )
        I, abserr = quad(integrand, 0, 1, epsabs=1e-12)  # noqa
        if I <= 0:
            raise ValueError("Computed integral is non-positive.")
        return -np.log(n) - np.log(I)

    def make_strategy(
        self,
        utility: UtilityBase,
        log_coefficient: Callable[[int, int], float] | None = None,
    ) -> EvaluationStrategy:
        # Simply use the standard PowersetEvaluationStrategy.
        return super().make_strategy(utility, log_coefficient)
