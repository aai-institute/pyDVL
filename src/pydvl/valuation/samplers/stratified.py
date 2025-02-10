r"""
This module implements stratified samplers.

For each index $i \in N$, stratified samplers first pick a set size $k \in [0, n],$
where $n$ is the total number of indices in the index set $N,$ then sample $S \subset
N_{-i}$ of size $k.$ One can either sample without constraints, or fix the amount of
samples $m_k$ at a given size. Optimal sampling (leading to minimal variance estimators)
involves a dynamic choice of $m_k$ based on the variance of the integrand being
approximated with Monte Carlo (i.e. the marginal utility), but there exist methods
applicable to semi-values which precompute these sizes while still providing reasonable
performance.

??? Note "The number of sets of size $k$"
    Recall that uniform sampling from the powerset $2^{N_{-i}}$ produces a binomial
    distribution of set sizes: the number of sets of size $k$ is $m_k = \binom{n-1}{k},$
    which is the (inverse of the) Shapley coefficient. Therefore, setting for instance
    $m_k = C$ for some constant will drastically reduce the number of sets of size
    $\sym n/2$ while increasing the number of sets of size 1 or $n-1.$ This will then have
    stark implications on the Monte Carlo estimate of semi-values, depending on how
    the marginal utility (i.e. the training of the model) is affected by the size of the
    training set.

This module implements:

* Uniform sampling of $k$ and subsets of that size, with
  [UniformStratifiedSampler][pydvl.valuation.samplers.stratified.UniformStratifiedSampler].
  This has however poor performance because it completely ignores the usual
  characteristics of utility functions in ML.
* Truncated uniform sampling with
  [TruncatedUniformStratifiedSampler][pydvl.valuation.samplers.stratified.TruncatedUniformStratifiedSampler],
  which is the same as above, but restricting $m_k = 0$ if $k \notin [l_n, u_n]$ for
  lower and upper bounds $l_n$ and $u_n$ determined as functions of $n,$ the total
  number of indices. This method was introduced by Watson et al. (2023)[^1] as
  $\delta$-Shapley.
* Variance Reduced Stratified Sampling (VRDS), with
  [VarianceReducedStratifiedSampler][pydvl.valuation.samplers.stratified.VarianceReducedStratifiedSampler].
  Introduced by Wu et al. (2023)[^2], this method uses simple heuristics to
  precompute $m_k,$ then samples sets of that size uniformly from $2^{N_{-i}}.$


## References

[^1]: <a name="watson_accelerated_2023"></a>Watson, Lauren, Zeno Kujawa, Rayna Andreeva,
      Hao-Tsung Yang, Tariq Elahi, and Rik Sarkar. [Accelerated Shapley Value
      Approximation for Data Evaluation](https://doi.org/10.48550/arXiv.2311.05346).
      arXiv, 9 November 2023.
[^2]: <a name="wu_variance_2023"></a>Wu, Mengmeng, Ruoxi Jia, Changle Lin, Wei Huang,
      and Xiangyu Chang. [Variance Reduced Shapley Value Estimation for Trustworthy Data
      Valuation](https://doi.org/10.1016/j.cor.2023.106305). Computers & Operations
      Research 159 (1 November 2023): 106305.
[^3]: <a name="maleki_bounding_2014"></a>Maleki, Sasan, Long Tran-Thanh, Greg Hines,
      Talal Rahwan, and Alex Rogers. [Bounding the Estimation Error of Sampling-Based
      Shapley Value Approximation](https://arxiv.org/abs/1306.4265). arXiv:1306.4265
      [Cs], 12 February 2014.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import lru_cache
from typing import Type

from pydvl.utils import Seed, complement, random_subset_of_size
from pydvl.valuation.samplers.powerset import (
    FiniteSequentialIndexIteration,
    IndexIteration,
    PowersetSampler,
    SequentialIndexIteration,
)
from pydvl.valuation.samplers.utils import StochasticSamplerMixin
from pydvl.valuation.types import IndexSetT, Sample, SampleGenerator

__all__ = [
    "TruncatedUniformStratifiedSampler",
    "UniformStratifiedSampler",
    "VarianceReducedStratifiedSampler",
    "HarmonicSamplesPerSetSize",
    "PowerLawSamplesPerSetSize",
]


class TruncatedUniformStratifiedSampler(StochasticSamplerMixin, PowersetSampler):
    r"""A sampler which first samples set sizes between two bounds, then subsets of that
    size.

    For every index $i,$ this sampler first draws a set size $m_k$ between two bounds.
    Then a set of that size is sampled uniformly from the complement of the current index.

    This sampler was suggested in (Watson et al. 2023)<sup><a
    href="#watson_accelerated_2023">1</a></sup> for $\delta$-Shapley

    Args:
        lower_bound: The lower bound for the set size. If None, the lower bound is 0.
        upper_bound: The upper bound for the set size. If None, the upper bound is set
            to the size of the index set when sampling.
        batch_size: The number of samples to generate per batch. Batches are processed
            together by each subprocess when working in parallel.
        index_iteration: the strategy to use for iterating over indices to update
        seed: The seed for the random number generator.

    Raises:
        ValueError when generating samples If the lower bound is less than 1 or the
            upper bound is less than the lower bound.

    !!! tip "New in version 0.10.0"
    """

    def __init__(
        self,
        *,
        lower_bound: int | None = None,
        upper_bound: int | None = None,
        batch_size: int = 1,
        index_iteration: Type[IndexIteration] = SequentialIndexIteration,
        seed: Seed | None = None,
    ):
        super().__init__(
            batch_size=batch_size, index_iteration=index_iteration, seed=seed
        )
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @contextmanager
    def ensure_correct_bounds(self, n: int):
        """A context manager to ensure that the bounds are correct.

        This locally changes the bounds to the correct values and restores them after.

        Args:
            n: Size of the index set
        """
        save_lower = self.lower_bound
        save_upper = self.upper_bound
        try:
            if self.lower_bound is not None:
                if self.lower_bound < 0 or self.lower_bound > n:
                    raise ValueError(
                        f"Lower bound ({self.lower_bound}) must be between 0 and {n=}, "
                        f"inclusive"
                    )
            else:
                self.lower_bound = 0
            if self.upper_bound is not None:
                if self.upper_bound < self.lower_bound or self.upper_bound > n:
                    raise ValueError(
                        f"Upper bound ({self.upper_bound}) must be between lower bound "
                        f"({self.lower_bound=}) and {n=}, inclusive"
                    )
            else:
                self.upper_bound = n
            yield
        finally:
            self.lower_bound = save_lower
            self.upper_bound = save_upper

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        # For NoIndexIteration, we sample from the full set of indices, but
        # for other index iterations len(complement) < len(indices), so we need
        # to adjust the bounds
        with self.ensure_correct_bounds(
            self._index_iterator_cls.complement_size(len(indices))
        ):
            assert self.lower_bound is not None  # for mypy
            assert self.upper_bound is not None
            for idx in self.index_iterator(indices):
                _complement = complement(indices, [idx])
                k = self._rng.integers(
                    low=self.lower_bound, high=self.upper_bound + 1, size=1
                ).item()
                subset = random_subset_of_size(_complement, size=k, seed=self._rng)
                yield Sample(idx, subset)

    def weight(self, n: int, subset_len: int) -> float:
        r"""The probability of sampling a set of size k is 1/(n-1 choose k) times the
        probability of choosing size k:

        $$\mathbb{P}(S) = \binom{n-1}{k}^{-1} \times
                          \frac{1}{\text{upper_bound} - \text{lower_bound} + 1}$$
        """
        n = self._index_iterator_cls.complement_size(n)
        with self.ensure_correct_bounds(n):
            assert self.lower_bound is not None  # for mypy
            assert self.upper_bound is not None
            if not (self.lower_bound <= subset_len <= self.upper_bound):
                raise ValueError("Subset length out of bounds. This should not happen")

            inv_prob_size = int(self.upper_bound - self.lower_bound + 1)
            return math.comb(n, subset_len) * inv_prob_size


class UniformStratifiedSampler(TruncatedUniformStratifiedSampler):
    """A sampler that samples set sizes uniformly, then subsets of that size.

    For every index, this sampler first draws a set size between 0 and n-1 uniformly,
    where n is the total number of indices. Then a set of that size is sampled uniformly
    from the complement of the current index.

    ??? Info
        Stratified sampling partitions a population by a characteristic and samples
        randomly within each stratum. In model-based data valuation, utility functions
        typically depend on training sample size, so that it makes sense to stratify
        by coalition size.

    !!! Danger
        This sampling scheme yields many more samples of low and high set sizes than a
        [UniformSampler][pydvl.valuation.samplers.UniformSampler] and is therefore
        inadequate for most methods due to the high variance in the estimates that it
        induces. It is included for completeness and for experimentation purposes.

    !!! tip "Renamed in version 0.10.0"
        This used to be called `RandomHierarchicalSampler`.
    """

    def __init__(
        self,
        batch_size: int = 1,
        index_iteration: Type[IndexIteration] = SequentialIndexIteration,
        seed: Seed | None = None,
    ):
        super().__init__(
            lower_bound=0,
            upper_bound=None,
            batch_size=batch_size,
            index_iteration=index_iteration,
            seed=seed,
        )


class SamplesPerSetSizeStrategy(ABC):
    r"""A callable which returns the number of samples to take for a given set size.
    Based on Wu et al. (2023)<sup><a href="#wu_variance_2023">1</a></sup>, Theorem 4.2.

    To be used in the
    [VarianceReducedStratifiedSampler][pydvl.valuation.samplers.VarianceReducedStratifiedSampler].

    Sets the number of sets at size $k$ to be

    $$m(k) = m \frac{f(k)}{\sum_{j=0}^{n} f(j)},$$

    for some choice of $f.$ Implementations of this base class must override the
    method `fun()`. It is provided both the size $k$ and the total number of indices $n$
    as arguments.
    """

    def __init__(self, n_samples_per_index: int):
        """Construct a heuristic for the given number of samples.

        Args:
            n_samples_per_index: Number of samples for the stratified sampler to
                generate, *per index*. If the sampler uses
                [NoIndexIteration][pydvl.valuation.samplers.NoIndexIteration], then this
                is the total number of samples.
        """
        self.n_samples_per_index = n_samples_per_index

    @abstractmethod
    def fun(self, n_indices: int, subset_len: int) -> float:
        """The function $f$ to use in the heuristic.
        Args:
            n_indices: Size of the index set.
            subset_len: Size of the subset.
        """
        ...

    @lru_cache
    def sample_sizes(self, n_indices: int) -> list[int]:
        """Precomputes the number of samples to take for each set size, from 0 up to
        `n_indices` inclusive.

        This method corrects rounding errors taking into account the fractional parts
        so that the total number of samples is respected, while allocating remainders
        in a way that follows the relative sizes of the fractional parts.

        ??? Note
            A naive implementation with e.g.
            ```python
            m_k = [max(1, int(round(m * f(k)/sum(f(j) for j in range(n)), 0)))
                    for k in range(n)]
            ```
            would not respect the total number of samples, and would not distribute
            remainders correctly.

        Args:
            n_indices: number of indices in the index set from which to sample. This is
                typically `len(dataset) - 1` with the usual index iterations.
        Returns:
            The number of samples to take for each set size.
        """
        # Sizes up to the whole index set
        n_indices += 1

        # m_k = m * f(k) / sum_j f(j)
        s = sum(self.fun(n_indices, k) for k in range(n_indices))
        values = [
            self.n_samples_per_index * self.fun(n_indices, k) / s
            for k in range(n_indices)
        ]

        # Round down and distribute remainder by adjusting the largest fractional parts
        int_values = [int(m) for m in values]
        remainder = self.n_samples_per_index - sum(int_values)
        fractional_parts = [(v - int(v), i) for i, v in enumerate(values)]
        fractional_parts.sort(reverse=True, key=lambda x: x[0])
        for i in range(remainder):
            int_values[fractional_parts[i][1]] += 1

        return int_values

    def __call__(self, n_indices: int, subset_len: int) -> int:
        """Returns the number of subsets to sample.

        Args:
            n_indices: Size of the index set to sample from.
            subset_len: Set size to use, from 0 to `n_indices` inclusive

        Returns:
            The number of samples to generate at size `subset_len`.
        """
        try:
            return self.sample_sizes(n_indices)[subset_len]
        except IndexError:
            raise ValueError(
                f"Subset length {subset_len} out of bounds for index set of size "
                f"{n_indices}"
            )


class HarmonicSamplesPerSetSize(SamplesPerSetSizeStrategy):
    r"""Heuristic choice of samples per set size for VRDS.

    Sets the number of sets at size $k$ to be

    $$m_k = m \frac{f(k)}{\sum_{j=0}^{n-1} f(j)},$$

    for

    $$f(k) = \frac{1}{1+k}.$$
    """

    def fun(self, n_indices: int, subset_len: int):
        return 1 / (1 + subset_len)


class PowerLawSamplesPerSetSize(SamplesPerSetSizeStrategy):
    r"""Heuristic choice of samples per set size for VRDS.

    Sets the number of sets at size $k$ to be

    $$m_k = m \frac{f(k)}{\sum_{j=0}^{n-1} f(j)},$$

    for

    $$f(k) = (1+k)^a, $$

    and some exponent $a.$ With $a=1$ one recovers the
    [HarmonicSamplesPerSetSize][pydvl.valuation.samplers.stratified.HarmonicSamplesPerSetSize]
    heuristic.

    Args:
        n_samples_per_index: Total number of samples for VRDS to generate.
        exponent: The exponent to use. Recommended values are between -1 and -0.5.
    """

    def __init__(self, n_samples_per_index: int, exponent: float):
        super().__init__(n_samples_per_index)
        self.exponent = exponent

    def fun(self, n_indices: int, subset_len: int):
        return (1 + subset_len) ** self.exponent


class VarianceReducedStratifiedSampler(StochasticSamplerMixin, PowersetSampler):
    """A sampler stratified by coalition size with variable number of samples per set
    size.

    Variance Reduced Stratified Sampler (VRDS) was suggested by Wu et al. 2023<sup><a
    href="#wu_variance_2023">2</a></sup>, as a generalization of the stratified
    sampler in Maleki et al. (2014)<sup><a href="#maleki_bounding_2014">3</a></sup>.

    ## Choosing the number of samples per set size

    The idea of VRDS is to allow per-set-size configuration of the total number of
    samples in order to reduce the variance coming from the marginal utility evaluations.

    It is known (Wu et al. (2023), Theorem 4.2) that a minimum variance estimator of
    Shapley values samples a number $m_k$ of sets of size $k$ based on the variance of
    the marginal utility at that set size. However, this quantity is unknown in
    practice, so the authors propose a simple heuristic. This function
    (`samples_per_setsize` in the arguments) is deterministic, and in particular does
    not depend on run-time variance estimates, as an adaptive method might do. Section 4
    of Wu et al. (2023) shows a good default choice is based on the harmonic function
    of the set size $k$ (see
    [HarmonicSamplesPerSetSize][pydvl.valuation.samplers.stratified.HarmonicSamplesPerSetSize]).

    Args:
        samples_per_setsize: An object which returns the number of samples to
            take for a given set size. The sampler will generate exactly this many, and
            no more (this is a finite sampler).
        batch_size: The number of samples to generate per batch. Batches are processed
            together by each subprocess when working in parallel.
        index_iteration: the strategy to use for iterating over indices to update.
            Note that anything other than returning index exactly once will break the
            weight computation.
        seed: The seed for the random number generator.

    !!! tip "New in version 0.10.0"
    """

    def __init__(
        self,
        samples_per_setsize: SamplesPerSetSizeStrategy,
        batch_size: int = 1,
        index_iteration: Type[IndexIteration] = FiniteSequentialIndexIteration,
        seed: Seed | None = None,
    ):
        super().__init__(
            batch_size=batch_size, index_iteration=index_iteration, seed=seed
        )
        if not index_iteration.is_finite():
            raise ValueError(
                "VarianceReducedStratifiedSampler requires a finite index iterator"
            )
        self.samples_per_setsize = samples_per_setsize

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        for idx in self.index_iterator(indices):
            from_set = complement(indices, [idx])
            n_indices = len(from_set)
            for k in range(n_indices + 1):
                for _ in range(self.samples_per_setsize(n_indices, k)):
                    subset = random_subset_of_size(from_set, size=k, seed=self._rng)
                    yield Sample(idx, subset)

    @lru_cache
    def total_samples(self, n_indices: int) -> int:
        """The total number of samples depends on the index iteration.
        If it's a regular index iteration, then
        """
        if n_indices == 0:
            return 0
        index_iteration_length = self._index_iterator_cls.length(range(n_indices))  # type: ignore
        if index_iteration_length is None:
            return 0
        index_iteration_length = max(1, index_iteration_length)
        return index_iteration_length * sum(
            self.samples_per_setsize(n_indices, k) for k in range(n_indices + 1)
        )

    def sample_limit(self, indices: IndexSetT) -> int:
        return self.total_samples(len(indices))

    def weight(self, n: int, subset_len: int) -> float:
        r"""The probability of sampling a set of size k is 1/(n choose k) times the
        probability of choosing size k, which is the number of samples for that size
        divided by the total number of samples for all sizes:

        $$\mathbb{P}(S) = \binom{n}{k}^{-1} \times
                       \frac{\text{samples_per_setsize}(k)}{\text{total_samples}(n)}$$
        """
        n = self._index_iterator_cls.complement_size(n)

        # Depending on whether we sample from complements or not, the total number of
        # samples passed to the heuristic has a different interpretation.
        index_iteration_length = self._index_iterator_cls.length(range(n))  # type: ignore
        if index_iteration_length is None:
            index_iteration_length = n
        index_iteration_length = max(1, index_iteration_length)

        return (
            math.comb(n, subset_len)
            * self.total_samples(n)
            / index_iteration_length
            / self.samples_per_setsize(n, subset_len)
        )
