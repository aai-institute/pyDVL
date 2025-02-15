r"""
This module implements stratified samplers.

Stratified samplers change the subset sampling distribution to be a function of set size
with the goal of reducing the variance of the Monte Carlo estimate of the marginal
utility. This is essentially importance sampling, with the heuristic that the utility's
variance is a function of the training set size.

In the simplest case,
[StratifiedSampler][pydvl.valuation.samplers.stratified.StratifiedSampler] takes a
strategy with a fixed number of samples $m_k$ for each set size $k \in [0, n],$ where
$n$ is the total number of indices in the index set $N.$ It iterates through all indices
exactly once (using
[FiniteSequentialIndexIteration][pydvl.valuation.samplers.powerset.FiniteSequentialIndexIteration])
and for each index $i \in N$, iterates through all set sizes $k$, then samples exactly
$m_k$ subsets $S \subset N_{-i}$ of size $k$.

This is the procedure used e.g. in Variance Reduced Stratified Sampling (VRDS),
introduced by Wu et al. (2023)[^2], with a simple heuristic setting $m_k$ to a
decreasing function of $k.$ Their heuristic is a generalization of the ideas in Maleki
et al. (2014)[^3].

??? Example "Constructing a VRDS"
    To create a sampler as in Wu et al. (2023)[^2], one would use the following code:

    ```python
    n_samples_per_index = 1000  # Total number of samples is: n_indices times this
    sampler = StratifiedSampler(
        sample_sizes=HarmonicSampleSize(n_samples=1000),
        sample_sizes_iteration=DeterministicSizeIteration,
        index_iteration=FiniteSequentialIndexIteration,
        )
    ```

## Available strategies

All components described below can be mixed in most ways, but some specific
configurations appear in the literature as follows:

* Constant sample sizes $m_k = c$, but restricting $m_k = 0$ if $k \notin [l_n, u_n]$
  for lower and upper bounds $l_n$ and $u_n$ determined as functions of $n,$ the total
  number of indices. This sampling method was introduced by Watson et al. (2023)[^1] for
  the computation of Shapley values as $\delta$-Shapley.
  ??? Example "Constructing a sampler for $\delta$-Shapley"
      ```python
      sampler = StratifiedSampler(
          sample_sizes=ConstantSampleSize(n_samples=10, lower_bound=1, upper_bound=2),
          sample_sizes_iteration=DeterministicSizeIteration,
          index_iteration=SequentialIndexIteration,
          )
      ```

* Sample sizes decreasing with a power law. Use
  [PowerLawSampleSize][pydvl.valuation.samplers.stratified.PowerLawSampleSize] for the
  strategy. This was also proposed in Wu et al. (2023)[^2]. Empirically they found
  an exponent between -1 and -0.5 to perform well.
  ??? Example "Power law heuristic"
      ```python
      sampler = StratifiedSampler(
            sample_sizes=PowerLawSampleSize(n_samples=1000, exponent=-0.5),
            sample_sizes_iteration=RandomSizeIteration,
            index_iteration=RandomIndexIteration,
            )

* Group Testing Sample Size. This heuristic is used for the stratified sampling
  required by
  [GroupTestingShapleyValuation][pydvl.valuation.methods.gt_shapley.GroupTestingShapleyValuation].


## Iterating over indices and its effect on `n_samples`

As any other sampler,
[StratifiedSampler][pydvl.valuation.samplers.stratified.StratifiedSampler] can iterate
over indices finitely or infinitely many times. It can also use
[NoIndexIteration][pydvl.valuation.samplers.powerset.NoIndexIteration] to sample from
the whole powerset. This is configured with the parameter `index_iteration`.

In the case of finite iterations, the sampler must distribute a finite total number of
samples among all indices. This is done by the
[SampleSizeStrategy][pydvl.valuation.samplers.stratified.SampleSizeStrategy], which
therefore requires an argument `n_samples` to be set to the number of samples **per
index**.

!!! Warning
    On the other hand, if the sampler iterates over the indices indefinitely,
    `n_indices` can be set, but only relative frequencies will matter. As we see next,
    there is another component that will affect how the sampler behaves.

## Iterating over set sizes

Additionally, [StratifiedSampler][pydvl.valuation.samplers.stratified.StratifiedSampler]
must iterate over sample sizes $k \in [0, n]$, and this can be done in multiple ways,
configured via subclasses of
[SampleSizeIteration][pydvl.valuation.samplers.stratified.SampleSizeIteration].

* [DeterministicSizeIteration][pydvl.valuation.samplers.stratified.DeterministicSizeIteration]
  will generate exactly $m_k$ samples for each $k$ before moving to the next $k.$ This
  implies that `n_samples` must be large enough for the computed $m_k$ to be valid.
* [RandomSizeIteration][pydvl.valuation.samplers.stratified.RandomSizeIteration] will
  sample a set size $k$ according to the distribution of sizes given by the strategy.
  When using this in conjunction with an infinite index iteration for the sampler,
  `n_samples` can be safely set to 1 since $m_k$ will be interpreted as a probability.
* [RoundRobinIteration][pydvl.valuation.samplers.stratified.RoundRobinIteration] will
  iterate over set sizes $k$ and generate one sample each time, until reaching $m_k.$

## Choosing set size heuristics

Optimal sampling (leading to minimal variance estimators) involves a dynamic choice of
the number $m_k$ of samples at size $k$ based on the variance of the Monte Carlo
integrand, but Wu et al. (2023)[^2] show that there exist methods applicable to
semi-values which precompute these sizes while still providing reasonable performance.

??? Note "The number of sets of size $k$"
    Recall that uniform sampling from the powerset $2^{N_{-i}}$ produces a binomial
    distribution of set sizes: the number of sets of size $k$ is $m_k = \binom{n-1}{k},$
    which is the (inverse of the) Shapley coefficient. Therefore, setting for instance
    $m_k = C$ for some constant will drastically reduce the number of sets of size
    $\sym n/2$ while increasing the number of sets of size 1 or $n-1.$ This will then
    have stark implications on the Monte Carlo estimate of semi-values, depending on how
    the marginal utility (i.e. the training of the model) is affected by the size of the
    training set.

This heuristic is configured with the argument `sample_size_strategy` of
[StratifiedSampler][pydvl.valuation.samplers.stratifed.StratifiedSampler], which is an
instance of
[SampleSizeStrategy][pydvl.valuation.samplers.stratified.SampleSizeStrategy].


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
from functools import lru_cache
from typing import Generator, Type

import numpy as np
from numpy.typing import NDArray

from pydvl.utils import Seed, complement, logcomb, random_subset_of_size
from pydvl.valuation.samplers.powerset import (
    FiniteSequentialIndexIteration,
    IndexIteration,
    PowersetSampler,
)
from pydvl.valuation.samplers.utils import StochasticSamplerMixin
from pydvl.valuation.types import IndexSetT, Sample, SampleGenerator

__all__ = [
    "StratifiedSampler",
    "HarmonicSampleSize",
    "PowerLawSampleSize",
    "GroupTestingSampleSize",
    "ConstantSampleSize",
    "SampleSizeStrategy",
    "DeterministicSizeIteration",
    "RandomSizeIteration",
    "RoundRobinIteration",
    "SampleSizeIteration",
]


class SampleSizeStrategy(ABC):
    r"""An object to compute the number of samples to take for a given set size.
    Based on Wu et al. (2023)<sup><a href="#wu_variance_2023">1</a></sup>, Theorem 4.2.

    To be used with [StratifiedSampler][pydvl.valuation.samplers.StratifiedSampler].

    Sets the number of sets at size $k$ to be

    $$m(k) = m \frac{f(k)}{\sum_{j=0}^{n} f(j)},$$

    for some choice of $f.$ Implementations of this base class must override the
    method `fun()`. It is provided both the size $k$ and the total number of indices $n$
    as arguments.
    """

    def __init__(self, n_samples: int):
        """Construct a heuristic for the given number of samples.

        Args:
            n_samples: Number of samples for the stratified sampler to generate,
                **per index**. If the sampler uses
                [NoIndexIteration][pydvl.valuation.samplers.NoIndexIteration], then this
                will coincide with the total number of samples.
        """
        self.n_samples = n_samples

    @abstractmethod
    def fun(self, n_indices: int, subset_len: int) -> float:
        """The function $f$ to use in the heuristic.
        Args:
            n_indices: Size of the index set.
            subset_len: Size of the subset.
        """
        ...

    @lru_cache
    def sample_sizes(
        self, n_indices: int, quantize: bool = True
    ) -> NDArray[np.int_] | NDArray[np.float_]:
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
            quantize: Whether to perform the remainder distribution. If `False`, the raw
                floating point values are returned. Useful e.g. for
                [RandomSizeIteration][pydvl.valuation.samplers.stratified.RandomSizeIteration]
                where one needs frequencies. In this case `n_samples` can
                be 1.
        Returns:
            The exact (integer) number of samples to take for each set size, if
            `quantize` is `True`. Otherwise, the fractional number of samples.
        """

        # m_k = m * f(k) / sum_j f(j)
        values = np.empty(n_indices + 1, dtype=float)
        s = 0.0

        for k in range(n_indices + 1):
            val = self.fun(n_indices, k)
            values[k] = val
            s += val

        values *= self.n_samples / s
        if not quantize:
            return values

        # Round down and distribute remainder by adjusting the largest fractional parts
        int_values = np.floor(values).astype(int)
        remainder = self.n_samples - np.sum(int_values)
        fractional_parts = values - int_values
        fractional_parts_indices = np.argsort(-fractional_parts)[:remainder]
        int_values[fractional_parts_indices] += 1
        return int_values


class ConstantSampleSize(SampleSizeStrategy):
    r"""Use a constant number of samples for each set size between two (optional)
    bounds. The total number of samples (per index) is respected.

    Args:
        n_samples: Total number of samples to generate **per index**.
        lower_bound: Lower bound for the set size. If the set size is smaller than this,
            the probability of sampling is 0.
        upper_bound: Upper bound for the set size. If the set size is larger than this,
            the probability of sampling is 0. If `None`, the upper bound is set to the
            number of indices.
    """

    def __init__(
        self,
        n_samples: int,
        lower_bound: int = 0,
        upper_bound: int | None = None,
    ):
        super().__init__(n_samples)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def fun(self, n_indices: int, subset_len: int) -> float:
        if (self.lower_bound is None or self.lower_bound <= subset_len) and (
            self.upper_bound is None or subset_len <= self.upper_bound
        ):
            return 1.0
        return 0.0


class GroupTestingSampleSize(SampleSizeStrategy):
    r"""Heuristic choice of samples per set size used for Group Testing.

    Sets the number of sets at size $k$ to be

    $$m_k = m \frac{f(k)}{\sum_{j=0}^{n-1} f(j)},$$

    for a total number of samples $m$ and:

    $$
    f(k) = \frac{1}{k} + \frac{1}{n-k}, \text{for} k \in \{1, n-1\}.
    $$
    """

    def fun(self, n_indices: int, subset_len: int) -> float:
        if subset_len < 0 or subset_len > n_indices:
            raise ValueError(f"{subset_len=} out of bounds (0, {n_indices=})")

        if subset_len == 0 or subset_len == n_indices:
            return 0
        return 1 / subset_len + 1 / (n_indices - subset_len)


class HarmonicSampleSize(SampleSizeStrategy):
    r"""Heuristic choice of samples per set size for VRDS.

    Sets the number of sets at size $k$ to be

    $$m_k = m \frac{f(k)}{\sum_{j=0}^{n-1} f(j)},$$

    for a total number of samples $m$ and:

    $$f(k) = \frac{1}{1+k}.$$
    """

    def fun(self, n_indices: int, subset_len: int):
        return 1 / (1 + subset_len)


class PowerLawSampleSize(SampleSizeStrategy):
    r"""Heuristic choice of samples per set size for VRDS.

    Sets the number of sets at size $k$ to be

    $$m_k = m \frac{f(k)}{\sum_{j=0}^{n-1} f(j)},$$

    for a total number of samples $m$ and:

    $$f(k) = (1+k)^a, $$

    and some exponent $a.$ With $a=1$ one recovers the
    [HarmonicSamplesPerSetSize][pydvl.valuation.samplers.stratified.HarmonicSamplesPerSetSize]
    heuristic.

    Args:
        n_samples: Total number of samples to generate **per index**.
        exponent: The exponent to use. Recommended values are between -1 and -0.5.
    """

    def __init__(self, n_samples: int, exponent: float):
        super().__init__(n_samples)
        self.exponent = exponent

    def fun(self, n_indices: int, subset_len: int):
        return (1 + subset_len) ** self.exponent


class SampleSizeIteration(ABC):
    """Given a strategy and the number of indices, yield tuples (k, count) that the
    sampler loop will use.
    Args:
        strategy: The strategy to use for computing the number of samples to take.
        n_indices: The number of indices in the index set from which samples are taken.
    """

    def __init__(self, strategy: SampleSizeStrategy, n_indices: int):
        self.strategy = strategy
        self.n_indices = n_indices

    @abstractmethod
    def __iter__(self) -> Generator[tuple[int, int], None, None]: ...


class DeterministicSizeIteration(SampleSizeIteration):
    def __iter__(self) -> Generator[tuple[int, int], None, None]:
        counts = self.strategy.sample_sizes(self.n_indices)
        for k, m_k in enumerate(counts):  # type: int, int
            if m_k > 0:
                yield k, m_k


class RandomSizeIteration(SampleSizeIteration):
    """Draws a set size $k$ following the distribution of sizes given by the
    strategy.
    """

    def __init__(
        self, strategy: SampleSizeStrategy, n_indices: int, seed: Seed | None = None
    ):
        super().__init__(strategy, n_indices)
        self._rng = np.random.default_rng(seed)

    def __iter__(self) -> Generator[tuple[int, int], None, None]:
        # In stochastic mode we interpret the counts as weights to sample one k.
        counts = self.strategy.sample_sizes(self.n_indices, quantize=False)
        total = counts.sum()
        if total == 0:
            raise ValueError("Total sample count is 0; cannot sample stochastically.")
        probs = counts / total
        k = self._rng.choice(np.arange(self.n_indices + 1), p=probs)
        yield k, 1


class RoundRobinIteration(SampleSizeIteration):
    def __iter__(self) -> Generator[tuple[int, int], None, None]:
        counts = self.strategy.sample_sizes(self.n_indices).copy()
        # Continue yielding until every k has been emitted exactly m_k times.
        # For example, if counts == [2, 3, 1] then we want the sequence:
        # (0,1), (1,1), (2,1), (0,1), (1,1), (1,1)
        while any(count > 0 for count in counts):
            for k, count in enumerate(counts):  # type: int, int
                if count > 0:
                    counts[k] -= 1
                    yield k, 1


class StratifiedSampler(StochasticSamplerMixin, PowersetSampler):
    """A sampler stratified by coalition size with variable number of samples per set
    size.

    ## Variance Reduced Stratified Sampler (VRDS)

    Stratified sampling was introduced at least as early as Maleki et al. (2014)<sup><a
    href="#maleki_bounding_2014">3</a></sup>. Wu et al. 2023<sup><a
    href="#wu_variance_2023">2</a></sup>, introduced heuristics adequate for ML tasks.

    ## Choosing the number of samples per set size

    The idea of VRDS is to allow per-set-size configuration of the total number of
    samples in order to reduce the variance coming from the marginal utility evaluations.

    It is known (Wu et al. (2023), Theorem 4.2) that a minimum variance estimator of
    Shapley values samples a number $m_k$ of sets of size $k$ based on the variance of
    the marginal utility at that set size. However, this quantity is unknown in
    practice, so the authors propose a simple heuristic. This function
    (`sample_sizes` in the arguments) is deterministic, and in particular does
    not depend on run-time variance estimates, as an adaptive method might do. Section 4
    of Wu et al. (2023) shows a good default choice is based on the harmonic function
    of the set size $k$ (see
    [HarmonicSamplesPerSetSize][pydvl.valuation.samplers.stratified.HarmonicSamplesPerSetSize]).

    Args:
        sample_sizes: An object which returns the number of samples to
            take for a given set size. If `index_iteration` below is finite, then the
            sampler will generate exactly as many samples of each size as returned by
            this object. If the iteration is infinite, then the `sample_sizes` will be
            used as probabilities of sampling.
        sample_sizes_iteration: How to loop over sample sizes. The main modes are:
            * deterministically. For every k generate m_k samples before moving to k+1.
            * stochastically. Sample sizes k according to the distribution given by
              `sample_sizes`.
            * round-robin. Iterate over k, and generate 1 sample each time, until
              reaching m_k.
            But more can be created by subclassing
            [SampleSizeIteration][pydvl.valuation.samplers.stratified.SampleSizeIteration].
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
        sample_sizes: SampleSizeStrategy,
        sample_sizes_iteration: Type[SampleSizeIteration] = DeterministicSizeIteration,
        batch_size: int = 1,
        index_iteration: Type[IndexIteration] = FiniteSequentialIndexIteration,
        seed: Seed | None = None,
    ):
        super().__init__(
            batch_size=batch_size, index_iteration=index_iteration, seed=seed
        )
        self.sample_sizes_strategy = sample_sizes
        self.sample_sizes_iteration = sample_sizes_iteration

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        for idx in self.index_iterator(indices):
            from_set = complement(indices, [idx])
            n_indices = len(from_set)
            try:
                sample_sizes = self.sample_sizes_iteration(  # type: ignore
                    self.sample_sizes_strategy,
                    n_indices,
                    seed=self._rng,
                )
            except TypeError:
                sample_sizes = self.sample_sizes_iteration(
                    self.sample_sizes_strategy, n_indices
                )
            for k, m_k in sample_sizes:
                for _ in range(m_k):
                    subset = random_subset_of_size(from_set, size=k, seed=self._rng)
                    yield Sample(idx, subset)

    @lru_cache
    def total_samples(self, n_indices: int) -> int:
        index_iteration_length = self._index_iterator_cls.length(n_indices)

        # For infinite iterations, we consider the total of samples after a single whole
        # loop over all indices. For random iterations this will introduce some variance
        if index_iteration_length is None:
            index_iteration_length = 1

        return index_iteration_length * self.sample_sizes_strategy.n_samples

    def sample_limit(self, indices: IndexSetT) -> int | None:
        return self.total_samples(len(indices))

    def weight(self, n: int, subset_len: int) -> float:
        r"""The probability of sampling a set of size k is 1/(n choose k) times the
        probability of choosing size k, which is the number of samples for that size
        divided by the total number of samples for all sizes:

        $$P(S) = \binom{n}{k}^{-1} \ \frac{m_k}{m},$$

        where $m_k$ is the number of samples of size $k$ and $m$ is the total number
        of samples.
        """
        n = self._index_iterator_cls.complement_size(n)

        # Depending on whether we sample from complements or not, the total number of
        # samples passed to the heuristic has a different interpretation.
        index_iteration_length = self._index_iterator_cls.length(n)  # type: ignore
        if index_iteration_length is None:
            index_iteration_length = 1
        index_iteration_length = max(1, index_iteration_length)

        # Note that we can simplify the quotient
        # $$ \frac{m_k}{m} =
        #    \frac{m \frac{f (k)}{\sum_j f (j)}}{m} = \frac{f(k)}{\sum_j f (j)} $$
        # so that in the weight computation we can use the function $f$ directly from
        # the strategy, or equivalently, call `sample_sizes(n, quantize=False)`.
        # This is useful for the stochastic iteration, where we have frequencies
        # and m is possibly 1, so that quantization would yield a bunch of zeros.
        funs = self.sample_sizes_strategy.sample_sizes(n, quantize=False)
        funs /= np.sum(funs)

        return float(
            math.comb(n, subset_len) / index_iteration_length / funs[subset_len]
        )

    def log_weight(self, n: int, subset_len: int) -> float:
        n = self._index_iterator_cls.complement_size(n)
        # Depending on whether we sample from complements or not, the total number of
        # samples passed to the heuristic has a different interpretation.
        index_iteration_length = self._index_iterator_cls.length(n)  # type: ignore
        if index_iteration_length is None:
            index_iteration_length = 1
        index_iteration_length = max(1, index_iteration_length)

        funs = self.sample_sizes_strategy.sample_sizes(n, quantize=False)
        total = np.sum(funs)

        return (
            -logcomb(n, subset_len)
            + math.log(index_iteration_length)
            + math.log(funs[subset_len])
            - math.log(total)
        )
