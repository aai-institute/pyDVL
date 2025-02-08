"""Samplers for Owen Shapley values.

The Owen Shapley value is a sampling-based method to estimate Shapley values. It samples
probability values between 0 and 1 and then draws subsets of the complement of the
current index where each element is sampled with the given probability.

The basic sampler is [FiniteOwenSampler][pydvl.valuation.samplers.owen.FiniteOwenSampler],
which follows the idea of the original paper (Okhrati and Lipani, 2021)<sup><a
href="#okhrati_multilinear_2021">1</a></sup> and uses a deterministic grid of
probability values for the inner sampling. The number of samples drawn is therefore
constant and equal to `n_samples_outer * n_samples_inner` and
[OwenShapleyValuation][pydvl.valuation.methods.OwenShapleyValuation] should be
instantiated with [NoneStopping][pydvl.valuation.stopping.NoneStopping] as stopping
criterion.

Additionally, we provide [OwenSampler][pydvl.valuation.samplers.owen.OwenSampler], which
samples probability values between 0 and 1 at random indefinitely, and
[AntitheticOwenSampler][pydvl.valuation.samplers.owen.AntitheticOwenSampler], which
draws probability values $q$ between 0 and 0.5 at random and then generates two samples
for each index, one with the probability $q$ and another with probability $1-q$. These
samplers require therefore a stopping criterion to be used with the valuation method.

[^1]: <a name="okhrati_multilinear_2021"></a>Okhrati, R., Lipani, A., 2021.
    [A Multilinear Sampling Algorithm to Estimate Shapley
    Values](https://ieeexplore.ieee.org/abstract/document/9412511). In: 2020 25th
    International Conference on Pattern Recognition (ICPR), pp. 7992–7999. IEEE.

"""

from __future__ import annotations

import math
from typing import Type

import numpy as np

from pydvl.utils import Seed, random_subset
from pydvl.utils.numeric import complement
from pydvl.valuation.samplers.powerset import (
    FiniteSequentialIndexIteration,
    IndexIteration,
    PowersetSampler,
    RandomIndexIteration,
)
from pydvl.valuation.samplers.utils import StochasticSamplerMixin
from pydvl.valuation.types import IndexSetT, Sample, SampleGenerator

__all__ = ["AntitheticOwenSampler", "FiniteOwenSampler", "OwenSampler"]


class FiniteOwenSampler(StochasticSamplerMixin, PowersetSampler):
    """A finite Owen sampler for Shapley values.

    For each index $i$ the Owen sampler loops over a deterministic grid of probabilities
    (containing `n_samples_outer` entries between 0 and 1) and then draws
    `n_samples_inner` subsets of the complement of the current index where each element
    is sampled with the given probability.

    The total number of samples drawn is therefore `n_samples_outer * n_samples_inner`.

    !!! Note
        This finite sampler is intended to reproduce the results of the original paper.
        The infinite [OwenSampler][pydvl.valuation.samplers.OwenSampler] samples
        probability values between 0 and 1 at random indefinitely, which makes for a
        usage pattern that is more in line with the general use of samplers in the
        library.

    Args:
        n_samples_outer: The number of entries in the probability grid used for
            the outer loop in Owen sampling.
        n_samples_inner: The number of samples drawn for each probability. In the
            original paper this was fixed to 2 for all experiments which is why we
            give it a default value of 2.
        batch_size: The batch size of the sampler.
        index_iteration: The index iteration strategy, either sequential or random.
        seed: The seed for the random number generator.

    """

    def __init__(
        self,
        n_samples_outer: int,
        n_samples_inner: int = 2,
        batch_size: int = 1,
        seed: Seed | None = None,
    ):
        super().__init__(
            batch_size=batch_size,
            index_iteration=FiniteSequentialIndexIteration,
            seed=seed,
        )
        self._n_samples_inner = n_samples_inner
        self._n_samples_outer = n_samples_outer
        self._q_stop = 1.0

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        probabilities = np.linspace(
            start=0, stop=self._q_stop, num=self._n_samples_outer
        )
        for idx in self.index_iterator(indices):
            _complement = complement(indices, [idx])
            for prob in probabilities:
                for _ in range(self._n_samples_inner):
                    subset = random_subset(_complement, q=prob, seed=self._rng)
                    yield Sample(idx, subset)

    def weight(self, n: int, subset_len: int) -> float:
        """The probability of drawing a subset of a given length from the complement of
        the current index is 1/(n-1 choose k).
        """
        m = self._index_iterator_cls.complement_size(n)
        return math.comb(m, subset_len) * int(n)

    def sample_limit(self, indices: IndexSetT) -> int | None:
        return len(indices) * self._n_samples_outer * self._n_samples_inner


class OwenSampler(StochasticSamplerMixin, PowersetSampler):
    """A random sampler for Owen shapley values.

    For each index $i$ we sample a probability $q$ from the uniform distribution
    between 0 and 1 and then draw `n_samples_inner` subsets of the complement of the
    current index where each element is sampled with the given probability.

    Indices are iterated at random by default.

    Args:
        n_samples_inner: The number of samples drawn for each probability. In the
            original paper this was fixed to 2 for all experiments which is why we
            give it a default value of 2.
        batch_size: The batch size of the sampler.
        index_iteration: The index iteration strategy, either sequential or random.
        seed: The seed for the random number generator.
    """

    def __init__(
        self,
        n_samples_inner: int = 2,
        batch_size: int = 1,
        index_iteration: Type[IndexIteration] = RandomIndexIteration,
        seed: Seed | None = None,
    ):
        super().__init__(
            batch_size=batch_size, index_iteration=index_iteration, seed=seed
        )
        self._n_samples_inner = n_samples_inner
        self._q_stop = 1.0

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        while True:
            probs = self._rng.uniform(
                0, self._q_stop + np.finfo(float).eps, size=len(indices)
            )
            for idx, prob in zip(self.index_iterator(indices), probs):
                _complement = complement(indices, [idx])
                for _ in range(self._n_samples_inner):
                    subset = random_subset(_complement, q=prob, seed=self._rng)
                    yield Sample(idx, subset)

    def weight(self, n: int, subset_len: int) -> float:
        """The probability of drawing a subset of a given length from the complement of
        the current index is 1/(n-1 choose k).
        """
        m = self._index_iterator_cls.complement_size(n)
        return math.comb(m, subset_len) * int(n)


class AntitheticOwenSampler(OwenSampler):
    """A sampler for antithetic Owen shapley values.

    For each index $i$ we sample a probability $q$ from the uniform distribution
    between 0 and 1 and then draw `n_samples_inner` subsets of the complement of the
    current index where each element is sampled with the given probability.
    For each sample obtained that way, a second sample is generated by taking the
    complement of the first sample.

    Indices are iterated at random by default.

    For the same number of total samples, the antithetic Owen sampler yields usually
    more precise estimates of shapley values than the regular Owen sampler.

    Args:
        n_samples_inner: The number of samples drawn for each probability. In the
            original paper this was fixed to 2 for all experiments which is why we
            give it a default value of 2.
        batch_size: The batch size of the sampler.
        index_iteration: The index iteration strategy, either sequential or random.
        seed: The seed for the random number generator.
    """

    def __init__(
        self,
        n_samples_inner: int = 2,
        batch_size: int = 1,
        index_iteration: Type[IndexIteration] = RandomIndexIteration,
        seed: Seed | None = None,
    ):
        super().__init__(
            n_samples_inner=n_samples_inner,
            batch_size=batch_size,
            index_iteration=index_iteration,
            seed=seed,
        )
        self._q_stop = 0.5

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        for sample in super()._generate(indices):
            idx, subset = sample
            _exclude = [idx] + subset.tolist()
            _antithetic_subset = complement(indices, _exclude)
            yield sample
            yield Sample(idx, _antithetic_subset)
