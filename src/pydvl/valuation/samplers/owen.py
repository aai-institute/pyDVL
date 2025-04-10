"""Samplers for Owen Shapley values.

The Owen Shapley value is a sampling-based method to estimate Shapley values. It samples
probability values between 0 and 1 and then draws subsets of the complement of the
current index where each element is sampled with the given probability.

## Possible configurations

The basic sampler is [OwenSampler][pydvl.valuation.samplers.owen.OwenSampler]. It can
be configured with a deterministic grid of probability values or a uniform distribution
between 0 and 1. The former follows the idea of the original paper (Okhrati and Lipani,
2021)[^1].

This strategy for the sampling of probability values can be specified with the
[GridOwenStrategy][pydvl.valuation.samplers.owen.GridOwenStrategy] or
[UniformOwenStrategy][pydvl.valuation.samplers.owen.UniformOwenStrategy] classes.

In addition, and as is the case for all
[PowerSetSamplers][pydvl.valuation.samplers.powerset], one can configure the way the
sampler iterates over indices to be updated. This can be done with the
[IndexIteration][pydvl.valuation.samplers.powerset.IndexIteration] strategy.

When using infinite index iteration, the sampler can be used with a stopping criterion
to estimate Shapley values. This follows more closely the typical usage pattern in PyDVL
than the original sampling method described in Okhrati and Lipani (2021)[^1].

## Antithetic Owen sampling

We also provide an
[AntitheticOwenSampler][pydvl.valuation.samplers.owen.AntitheticOwenSampler], which
draws probability values $q$ between 0 and 0.5 again, either deterministically over a
discrete grid or at uniformly at random, and then generates two samples for each index,
one with the probability $q$ and another with probability $1-q$.

It can be configured in the same manner as the regular Owen sampler.

???+ Example "The four main configurations"
    ```python
    standard_owen = OwenSampler(
        outer_sampling_strategy=GridOwenStrategy(n_samples_outer=100),
        n_samples_inner=2,
        index_iteration=FiniteSequentialIndexIteration,
        )
    antithetic_owen = AntitheticOwenSampler(
        outer_sampling_strategy=GridOwenStrategy(n_samples_outer=100),
        n_samples_inner=2,
        index_iteration=FiniteSequentialIndexIteration,
        )
    infinite_owen = OwenSampler(
        outer_sampling_strategy=UniformOwenStrategy(seed=42),
        n_samples_inner=2,
        index_iteration=RandomIndexIteration,
        seed=42
    )
    infinite_antithetic_owen = AntitheticOwenSampler(
        outer_sampling_strategy=UniformOwenStrategy(seed=42),
        n_samples_inner=2,
        index_iteration=RandomIndexIteration,
        seed=42
    )
    ```

## References

[^1]: <a name="okhrati_multilinear_2021"></a>Okhrati, R., Lipani, A., 2021.
    [A Multilinear Sampling Algorithm to Estimate Shapley
    Values](https://ieeexplore.ieee.org/abstract/document/9412511). In: 2020 25th
    International Conference on Pattern Recognition (ICPR), pp. 7992–7999. IEEE.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type, cast

import numpy as np
from numpy.typing import NDArray

from pydvl.utils import Seed, random_subset
from pydvl.utils.numeric import complement, logcomb
from pydvl.valuation.samplers.powerset import (
    FiniteSequentialIndexIteration,
    IndexIteration,
    PowersetSampler,
)
from pydvl.valuation.samplers.utils import StochasticSamplerMixin
from pydvl.valuation.types import IndexSetT, Sample, SampleGenerator

__all__ = [
    "AntitheticOwenSampler",
    "GridOwenStrategy",
    "OwenSampler",
    "OwenStrategy",
    "UniformOwenStrategy",
]


class OwenStrategy(ABC):
    """Base class for strategies for the Owen sampler to sample probability values."""

    def __init__(self, n_samples_outer: int):
        self.n_samples_outer = n_samples_outer

    @abstractmethod
    def __call__(self, q_stop: float) -> NDArray[np.float64]: ...


class UniformOwenStrategy(OwenStrategy):
    """A strategy for [OwenSampler][pydvl.valuation.samplers.owen.OwenSampler]
    to sample probability values uniformly between 0 and $q_\text{stop}$.

    Args:
        n_samples_outer: The number of probability values $q$ used for the outer loop.
            Since samples are taken anew for each index, a high number will delay
            updating new indices and has no effect on the final accuracy if using an
            infinite index iteration. In general, it only makes sense to change this
            number if using a finite index iteration.
        seed: The seed for the random number generator.
    """

    def __init__(self, n_samples_outer: int, seed: Seed | None = None):
        super().__init__(n_samples_outer=n_samples_outer)
        self.rng = np.random.default_rng(seed)

    def __call__(self, q_stop: float):
        # I doubt the + np.finfo(float).eps nudge will achieve anything...🤷
        return self.rng.uniform(
            0, q_stop + np.finfo(float).eps, size=self.n_samples_outer
        )


class GridOwenStrategy(OwenStrategy):
    """A strategy for [OwenSampler][pydvl.valuation.samplers.owen.OwenSampler] to sample
     probability values on a linear grid.

    Args:
        n_samples_outer: The number of probability values $q$ used for the outer loop.
            These will be linearly spaced between 0 and $q_\text{stop}$.
    """

    def __init__(self, n_samples_outer: int):
        super().__init__(n_samples_outer=n_samples_outer)

    def __call__(self, q_stop: float):
        return np.linspace(start=0, stop=q_stop, num=self.n_samples_outer)


class OwenSampler(StochasticSamplerMixin, PowersetSampler):
    """A sampler for semi-values using the Owen method.

    For each index $i$ we sample `n_samples_outer` probability values $q_j$ between 0
    and 1 and then, for each $j$ we draw `n_samples_inner` subsets of the complement of
    the current index where each element is sampled probability $q_j$.

    The distribution for the outer sampling can be either uniform or deterministic. The
    default is deterministic on a grid, which is the original method described in
    Okhrati and Lipani (2021)<sup><a href="#okhrati_multilinear_2021">1</a></sup>.
    This can be achieved by using the
    [GridOwenStrategy][pydvl.valuation.samplers.owen.GridOwenStrategy] strategy.

    Alternatively, the distribution can be uniform between 0 and 1. This can be achieved
    by using the [UniformOwenStrategy][pydvl.valuation.samplers.owen.UniformOwenStrategy]
    strategy.

    By combining a [UniformOwenStrategy][pydvl.valuation.samplers.owen.UniformOwenStrategy]
    with an infinite
    [IndexIteration][pydvl.valuation.samplers.powerset.IndexIteration] strategy, this
    sampler can be used with a stopping criterion to estimate semi-values. This
    follows more closely the typical usage pattern in PyDVL than the original sampling
    method described in Okhrati and Lipani (2021)<sup><a
    href="#okhrati_multilinear_2021">1</a></sup>.

    ??? Example "Example usage"
        ```python
        sampler = OwenSampler(
            outer_sampling_strategy=GridOwenStrategy(n_samples_outer=200),
            n_samples_inner=8,
            index_iteration=FiniteSequentialIndexIteration,
        )
        ```

    Args:
        n_samples_inner: The number of samples drawn for each probability. In the
            original paper this was fixed to 2 for all experiments.
        batch_size: The batch size of the sampler.
        index_iteration: The index iteration strategy, sequential or random, finite
            or infinite.
        seed: The seed for the random number generator.
    """

    def __init__(
        self,
        outer_sampling_strategy: OwenStrategy,
        n_samples_inner: int = 2,
        batch_size: int = 1,
        index_iteration: Type[IndexIteration] = FiniteSequentialIndexIteration,
        seed: Seed | None = None,
    ):
        super().__init__(
            batch_size=batch_size, index_iteration=index_iteration, seed=seed
        )
        self.n_samples_inner = n_samples_inner
        self.sampling_probabilities = outer_sampling_strategy
        self.q_stop = 1.0

    def generate(self, indices: IndexSetT) -> SampleGenerator:
        for idx in self.index_iterable(indices):
            _complement = complement(indices, [idx])
            for prob in self.sampling_probabilities(self.q_stop):
                for _ in range(self.n_samples_inner):
                    subset = random_subset(_complement, q=prob, seed=self._rng)
                    yield Sample(idx, subset)

    def log_weight(self, n: int, subset_len: int) -> float:
        r"""For each $q_j, j \in \{1, ..., N\}$ in the outer probabilities, the
        probability of drawing a subset $S_k$ of size $k$ is:

        $$ P (| S_{q_j} | = k) = \binom{n}{k} \  q_j^k  (1 - q_j)^{n - k}.$$

        So, if each $q_j$ is chosen with equal weight (or more generally with
        probability $p_j$),then by total probability, the overall probability of
        obtaining a subset of size $k$ is a mixture of the binomials:
        $$
        P (| S | = k) = \sum_{j = 1}^N p_j \ \binom{n}{k} \ q_j^k  (1 - q_j)^{n - k}.
        $$

        In our case $p_j = 1/N$, so that $P(|S|=k) = \frac{1}{N} \sum_{j=1}^N P (|
        S_{q_j} | = k)$. For large enough $N$ this is

        $$
        P(|S|=k) \approx \binom{n}{k} \int_0^1 q^k (1 - q)^{n - k} \, dq = \frac{1}{
        n+1},
        $$

        where we computed the integral using the beta function and its expression as
        products of gamma functions.

        Now, given the symmetry wrt. the indices in the sampling procedure, any given
        set $S$ of size $k$ is equally likely to be drawn. So the probability of a set
        being of size $k$ must be equally divided by the number of sets of that size,
        and the weight of a set of size $k$ is:

        $$ P(S) = \frac{1}{n+1} \binom{n}{|S|}^{-1}. $$

        Args:
            n: Size of the index set.
            subset_len: Size of the subset.
        Returns:
            The logarithm of the weight of a subset of size `subset_len`.
        """
        m = self.complement_size(n)
        return float(-logcomb(m, subset_len) - np.log(m + 1))

    def sample_limit(self, indices: IndexSetT) -> int | None:
        """The number of samples that will be generated by the sampler.

        Args:
            indices:

        Returns:
            0 if there are no indices, `None` if there's no limit and the number of
            samples otherwise.
        """
        if len(indices) == 0:
            return 0
        if not self._index_iterator_cls.is_finite():
            return None

        return (
            cast(int, self._index_iterator_cls.length(len(indices)))
            * self.sampling_probabilities.n_samples_outer
            * self.n_samples_inner
        )


class AntitheticOwenSampler(OwenSampler):
    """A sampler for antithetic Owen shapley values.

    For each sample obtained with the method of
    [OwenSampler][pydvl.valuation.samplers.owen.OwenSampler], a second sample is
    generated by taking the complement of the first sample.

    For the same number of total samples, the antithetic Owen sampler yields usually
    more precise estimates of shapley values than the regular Owen sampler.
    """

    def __init__(
        self,
        outer_sampling_strategy: OwenStrategy,
        n_samples_inner: int = 2,
        batch_size: int = 1,
        index_iteration: Type[IndexIteration] = FiniteSequentialIndexIteration,
        seed: Seed | None = None,
    ):
        super().__init__(
            outer_sampling_strategy=outer_sampling_strategy,
            n_samples_inner=n_samples_inner,
            batch_size=batch_size,
            index_iteration=index_iteration,
            seed=seed,
        )
        self.q_stop = 0.5

    def generate(self, indices: IndexSetT) -> SampleGenerator:
        for sample in super().generate(indices):
            idx, subset = sample
            _exclude = [idx] + subset.tolist()
            _antithetic_subset = complement(indices, _exclude)
            yield sample
            yield Sample(idx, _antithetic_subset)

    def sample_limit(self, indices: IndexSetT) -> int | None:
        x = super().sample_limit(indices)
        return 2 * x if x is not None else None
