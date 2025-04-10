"""
AMESampler implements a generalized Average Marginal Effect sampling scheme.

For each index $i$, it first samples a probability $q$ from a user defined distribution
$D_q$. Then it samples a subset from the complement of $i$, by including each element
independently, following a Bernoulli distribution with parameter $q$.

This two-stage process was introduced in Lin et al. (2022)[^1] together with an
approximation using Lasso. There it was shown, that under a uniform $D_q$, the Monte
Carlo sum of marginal utilities is an unbiased estimator of the Shapley value, without
the need for a correction factor. This is essentially the same as the Owen sampling
scheme.

However, one can also use other distributions for $D_q$, such as a beta distribution. In
this case, the Monte Carlo sum of marginal utilities no longer converges to Shapley
without the correction factor, and even then because of the very different shape of the
distribution, attempting to approximate Shapley values with this method is not advisable.


## References

[^1]: Lin, Jinkun, Anqi Zhang, Mathias Lécuyer, Jinyang Li, Aurojit Panda, and
      Siddhartha Sen. [Measuring the Effect of Training Data on Deep Learning Predictions
      via Randomized Experiments](https://proceedings.mlr.press/v162/lin22h.html). In
      Proceedings of the 39th International Conference on Machine Learning, 13468–504.
      PMLR, 2022.
"""

from __future__ import annotations

from typing import Type

import numpy as np
import scipy.stats.distributions as dist
from scipy.integrate import quad
from scipy.stats import rv_continuous

from pydvl.utils.numeric import complement, random_subset
from pydvl.utils.types import Seed
from pydvl.valuation.samplers.powerset import (
    IndexIteration,
    PowersetSampler,
    SequentialIndexIteration,
)
from pydvl.valuation.samplers.utils import StochasticSamplerMixin
from pydvl.valuation.types import IndexSetT, Sample, SampleGenerator


class AMESampler(StochasticSamplerMixin, PowersetSampler):
    """AMESampler implements the two-stage Average Marginal Effect sampling scheme.

    For each index i, it first samples a probability p from `p_distribution`. Then it
    samples a subset from the complement of i by including each element independently
    following a Bernoulli distribution with parameter p.

    Args:
        p_distribution: A scipy
            [continuous random variable][scipy.stats._distn_infrastructure.rv_continuous]
            Defaults to [scipy.stats.distributions.uniform][].
        batch_size: Number of samples per batch.
        index_iteration: Strategy to iterate over indices.
        seed: Seed or random state.
    """

    def __init__(
        self,
        p_distribution: rv_continuous | None = None,
        batch_size: int = 1,
        index_iteration: Type[IndexIteration] = SequentialIndexIteration,
        seed: Seed | None = None,
    ):
        super().__init__(
            batch_size=batch_size, index_iteration=index_iteration, seed=seed
        )
        self.p_distribution = p_distribution or dist.uniform

    def generate(self, indices: IndexSetT) -> SampleGenerator:
        """For each index i, sample p ~ p_distribution and then generate a subset of
        the complement by including each element independently with probability p.
        """
        for idx in self.index_iterable(indices):
            _complement = complement(indices, [idx])
            p = self.p_distribution.rvs(random_state=self._rng).item()
            subset = random_subset(_complement, p, self._rng)
            yield Sample(idx, subset)

    def sample_limit(self, indices: IndexSetT) -> int | None:
        """The sample limit is determined by the outer loop over indices."""
        return self._index_iterator_cls.length(len(indices))

    def log_weight(self, n: int, subset_len: int) -> float:
        r"""Computes the probability of a sample of a given size.

        Let m = n - 1 or m = n depending on the index iteration strategy. For a given
        index i and a sampled subset S of size k, The probability of obtaining S under
        the two-stage scheme is the result of marginalizing over the distribution of p:

        $$P(S) = \int_0^1 f(p) p^k (1-p)^{m-k} dp.$$

        Args:
            n: The size of the dataset.
            subset_len: The size of the sampled subset.
        Returns:
            The natural logarithm of P(S)
        """
        m = self._index_iterator_cls.complement_size(n)
        integrand = (
            lambda p: self.p_distribution.pdf(p)
            * (p**subset_len)
            * ((1 - p) ** (m - subset_len))
        )
        I, abserr = quad(integrand, 0, 1, epsabs=1e-12)  # noqa
        if I < 0:
            raise ValueError("Computed integral is non-positive.")
        return float(np.log(I) if I > 0 else -np.inf)
