r"""
This module implements stratified samplers.

Stratified samplers change the subset sampling distribution to be a function of set size
with the goal of reducing the variance of the Monte Carlo estimate of the marginal
utility. They key assumption / heuristic is that the utility's variance is a function of
the training set size.

Stratified sampling was introduced at least as early as Maleki et al. (2014)[^3]. Later
on, Wu et al. 2023[^2], extended these heuristics and proposed some for ML tasks, which
they called VRDS (see below). Watson et al. (2023)[^1] used error estimates for certain
model classes to propose a different heuristic. See [below](#other-known-strategies) and
[$\delta$-Shapley][delta-shapley-intro].

All stratified samplers in pyDVL are implemented by configuring (or subclassing) the
class [StratifiedSampler][pydvl.valuation.samplers.stratified.StratifiedSampler].

In the simplest case,
[StratifiedSampler][pydvl.valuation.samplers.stratified.StratifiedSampler] employs some
strategy with a fixed number of samples $m_k$ for each set size $k \in [0, n],$ where
$n$ is the total number of indices in the index set $N.$ It iterates through all indices
exactly (e.g. exactly once, if using
[FiniteSequentialIndexIteration][pydvl.valuation.samplers.powerset.FiniteSequentialIndexIteration])
and for each index $i \in N$, iterates through all set sizes $k$, then samples exactly
$m_k$ subsets $S \subset N_{-i}$ of size $k$. The iteration over set sizes is configured
with [SampleSizeIteration][pydvl.valuation.samplers.stratified.SampleSizeIteration].

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

This heuristic is configured with the argument `sample_size` of
[StratifiedSampler][pydvl.valuation.samplers.stratified.StratifiedSampler], which is an
instance of
[SampleSizeStrategy][pydvl.valuation.samplers.stratified.SampleSizeStrategy].


## Variance Reduced Stratified Sampler (VRDS)

It is known (Wu et al. (2023), Theorem 4.2)[^2] that a minimum variance estimator of
Shapley values samples $m_k$ sets of size $k$ based on the variance of the marginal
utility at that set size. However, this quantity is unknown in practice, so the authors
propose a simple deterministic heuristic, which in particular does not depend on
run-time variance estimates, as an adaptive method might do. Section 4 of Wu et al.
(2023)[^2] shows a good default choice is based on the harmonic function of the set size
$k$.

This sampler is available through
[VRDSSampler][pydvl.valuation.samplers.stratified.VRDSSampler].

??? Example "Constructing a VRDS"
    It is possible to "manually" repllicate
    [VRDSSampler][pydvl.valuation.samplers.stratified.VRDSSampler] with:

    ```python
    n_samples_per_index = 1000  # Total number of samples is: n_indices times this
    sampler = StratifiedSampler(
        sample_sizes=HarmonicSampleSize(n_samples=1000),
        sample_sizes_iteration=DeterministicSizeIteration,
        index_iteration=FiniteSequentialIndexIteration,
        )
    ```

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

## Other known strategies

All components described above can be mixed in most ways, but some specific
configurations besides VRDS appear in the literature as follows:

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
      ```

* Group Testing Sample Size. This heuristic is used for the stratified sampling
  required by
  [GroupTestingShapleyValuation][pydvl.valuation.methods.gt_shapley.GroupTestingShapleyValuation].


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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Generator, Literal, Type

import numpy as np
from numpy.typing import NDArray

from pydvl.utils import (
    Seed,
    complement,
    logcomb,
    maybe_add_argument,
    random_subset_of_size,
    )
    suppress_warnings,
)
from pydvl.valuation.samplers.powerset import (
    FiniteSequentialIndexIteration,
    IndexIteration,
    PowersetSampler,
)
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

__all__ = [
    "ConstantSampleSize",
    "FiniteSequentialSizeIteration",
    "DeltaShapleyNCSGDConfig",
    "DeltaShapleyNCSGDSampleSize",
    "GroupTestingSampleSize",
    "HarmonicSampleSize",
    "PowerLawSampleSize",
    "RandomSizeIteration",
    "RoundRobinIteration",
    "SampleSizeIteration",
    "SampleSizeStrategy",
    "StratifiedSampler",
    "VRDSSampler",
]

from pydvl.valuation.utility.base import UtilityBase


class SampleSizeStrategy(ABC):
    r"""An object to compute the number of samples to take for a given set size.

    To be used with [StratifiedSampler][pydvl.valuation.samplers.StratifiedSampler].

    Following the structure proposed in Wu et al.
    (2023),<sup><a href="#wu_variance_2023">2</a></sup> this sets the number of sets at
    size $k$ to be:

    $$m(k) = m \frac{f(k)}{\sum_{j=0}^{n} f(j)},$$

    for some choice of $f.$ Implementations of this base class must override the
    method `fun()` implementing $f$. It is provided both the size $k$ and the total
    number of indices $n$ as arguments.

    The argument `n_samples` can be fixed, or it can be set to `None` to indicate that
    it should be left to the strategy to compute. For strategies producing sampling
    probabilities, `n_samples` will be set to 1. For strategies producing integer
    sample sizes, `n_samples` will be set to the total number of samples generated per
    index.

    Args:
        n_samples: Number of samples for the stratified sampler to generate,
            **per index**, i.e. if the sampler iterates over each index exactly
            once, e.g.
            [FiniteSequentialIndexIteration][pydvl.valuation.samplers.powerset.FiniteSequentialIndexIteration],
            then the total number of samples will be `n_samples * n_indices`. Leave
            as `None` to let the strategy compute it whenever possible.
        lower_bound: Lower bound for the set sizes. If the set size is smaller than this,
            the probability of sampling is 0. If `None`, the lower bound is set to 0.
        upper_bound: Upper bound for the set size. If the set size is larger than this,
            the probability of sampling is 0. If `None`, the upper bound is set to the
            number of indices.
    """

    def __init__(
        self,
        n_samples: int | None = None,
        lower_bound: int | None = None,
        upper_bound: int | None = None,
    ):
        if n_samples is not None and n_samples < 0:
            raise ValueError(
                f"Number of samples must be non-negative, got {n_samples=}"
            )
        self.n_samples = n_samples
        if lower_bound is not None and lower_bound < 0:
            raise ValueError(f"Lower bound must be non-negative, got {lower_bound=}")
        if upper_bound is not None and upper_bound < 0:
            raise ValueError(f"Upper bound must be non-negative, got {upper_bound=}")
        if lower_bound is not None and upper_bound is not None:
            if lower_bound > upper_bound:
                raise ValueError(
                    f"Lower bound must be smaller than upper bound, got "
                    f"{lower_bound=}, {upper_bound=}"
                )
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

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
        self, n_indices: int, probs: bool = True
    ) -> NDArray[np.int64] | NDArray[np.float64]:
        """Precomputes the number of samples to take for each set size, from 0 up to
        `n_indices` inclusive.

        If `probs` is `True`, the result is a vector of floats, where each element
        is the probability of sampling a set of size $k.$ This is useful e.g. for
        [RandomSizeIteration][pydvl.valuation.samplers.stratified.RandomSizeIteration]
        where one needs frequencies. In this case `n_samples` can be `None`.

        If `probs` is `False`, the result is a vector of integers, where each
        element $k$ is the number of samples to take for set size $k.$ The sum of all
        elements is equal to `self.n_samples` if provided upon construction, or the sum
        of the values of `fun` for all set sizes if `self.n_samples` is `None`.

        When `probs` is `False`, this method corrects rounding errors taking into
        account the fractional parts so that the total number of samples is respected,
        while allocating remainders in a way that follows the relative sizes of the
        fractional parts.

        !!! warning "Ugly reliance on side effect"
            This method changes `n_samples` if it was `None` to the total number of
            samples computed. We rely on this in some places. It's ugly. (FIXME)

        Args:
            n_indices: number of indices in the index set from which to sample. This is
                typically `len(dataset) - 1` with the usual index iterations.
            probs: Whether to perform the remainder distribution. If `True`, sampling
                probabilities are returned. If `False`, then `n_samples` is used to
                compute the actual number of samples and the values are rounded
                down to the nearest integer, and the remainder is distributed to
                maintain the relative frequencies.
        Returns:
            The exact (integer) number of samples to take for each set size, if
            `probs` is `False`. Otherwise, the fractional number of samples.
        """

        # m_k = m * f(k) / sum_j f(j)
        values = np.zeros(n_indices + 1, dtype=float)
        s = 0.0
        lower = self.lower_bound if self.lower_bound is not None else 0
        upper = self.upper_bound if self.upper_bound is not None else n_indices
        lower = min(lower, n_indices)
        upper = min(upper, n_indices)

        for k in range(lower, upper + 1):
            val = self.fun(n_indices, k)
            values[k] = val
            s += val

        assert n_indices == 0 or s > 0, "Sum of sample sizes must be positive"
        values /= s

        # FIXME: make this hack more explicit
        if self.n_samples is None:
            self.n_samples = int(np.ceil(s))

        if probs:
            return values

        values *= self.n_samples
        # Round down and distribute remainder by adjusting the largest fractional parts
        # A naive implementation with e.g.
        #
        # m_k = [max(1, int(round(m * f(k)/sum(f(j) for j in range(n)), 0)))
        #         for k in range(n)]
        #
        # would not respect the total number of samples, and would not distribute
        # remainders correctly

        int_values: NDArray[np.int64] = np.floor(values).astype(np.int64)
        remainder = self.n_samples - np.sum(int_values)
        fractional_parts = values - int_values
        fractional_parts_indices = np.argsort(-fractional_parts)[:remainder]
        int_values[fractional_parts_indices] += 1
        return int_values


class ConstantSampleSize(SampleSizeStrategy):
    r"""Use a constant number of samples for each set size between two (optional)
    bounds. The total number of samples (per index) is respected.
    """

    def fun(self, n_indices: int, subset_len: int) -> float:
        if self.lower_bound is not None and subset_len < self.lower_bound:
            return 0.0
        if self.upper_bound is not None and subset_len > self.upper_bound:
            return 0.0
        return 1.0


@dataclass
class DeltaShapleyNCSGDConfig:
    """Configuration for Delta-Shapley non-convex SGD sampling.

    This is quite redundant, but convenient for passing around and storing all the
    parameters.

    Args:
        max_loss: Maximum loss.
        lipschitz_loss: Lipschitz constant of the loss
        lipschitz_grad: Lipschitz constant of the gradient of the loss
        lr_factor: Learning rate factor c, assuming $\alpha_t = c/t.$
        n_sgd_iter: Number of SGD iterations.
        n_val: Number of test samples.
        n_train: Number of training samples.
        eps: Epsilon value.
        delta: Delta value.
        version: Version of the bound to use: either the one from the paper or the one
            in the code.
    """

    max_loss: float
    lipschitz_loss: float
    lipschitz_grad: float
    lr_factor: float
    n_sgd_iter: int
    n_val: int
    n_train: int
    eps: float = 0.01
    delta: float = 0.05
    version: Literal["theorem7", "code"] = "theorem7"

    def __post_init__(self):
        assert self.eps > 0
        assert self.delta > 0
        assert self.lipschitz_grad > 0
        assert self.lr_factor > 0
        assert self.max_loss > 0
        assert self.lipschitz_loss > 0
        assert self.n_sgd_iter > 0


# TODO: implement the other bounds?
class DeltaShapleyNCSGDSampleSize(SampleSizeStrategy):
    r"""Heuristic choice of samples per set size for $\delta$-Shapley.

    This implements the non-convex SGD bound from Watson et al.
    (2023)<sup><a href="#watson_accelerated_2023">1</a></sup>.
    """

    def __init__(
        self,
        config: DeltaShapleyNCSGDConfig,
        lower_bound: int | None = None,
        upper_bound: int | None = None,
    ):
        super().__init__(
            n_samples=None, lower_bound=lower_bound, upper_bound=upper_bound
        )
        self.config = config

    def fun(self, n_indices: int, subset_size: int) -> int:
        """Computes the number of samples for the non-convex SGD bound."""
        q = self.config.lipschitz_grad * self.config.lr_factor
        H_1 = self.config.max_loss ** (q / (q + 1))
        H_2 = (2 * self.config.lr_factor * (self.config.lipschitz_loss**2)) ** (
            1 / (q + 1)
        )
        H_3 = self.config.n_sgd_iter ** (q / (q + 1))
        H_4 = (1 + (1 / q)) / (max(subset_size - 1, 1))
        H = H_1 * H_2 * H_3 * H_4

        if self.config.version == "code":
            return int(
                np.ceil(
                    2
                    * np.log((2 * n_indices) / self.config.delta)
                    * (
                        (
                            (H**2 / self.config.n_val)
                            + 2 * self.config.max_loss * self.config.eps / 3
                        )
                        / self.config.eps**2
                    )
                )
            )
        elif self.config.version == "theorem7":
            return int(
                np.ceil(
                    2
                    * np.log((2 * n_indices) / self.config.delta)
                    * (
                        (
                            2 * H**2
                            + 2 * self.config.max_loss * H
                            + 4 * self.config.max_loss * self.config.eps / 3
                        )
                        / self.config.eps**2
                    )
                )
            )
        else:
            raise ValueError(f"Unknown version: {self.config.version}")


class GroupTestingSampleSize(SampleSizeStrategy):
    r"""Heuristic choice of samples per set size used for Group Testing.

    [GroupTestingShapleyValuation][pydvl.valuation.methods.gt_shapley.GroupTestingShapleyValuation]
    uses this strategy for the stratified sampling of samples with which to construct
    the linear problem it requires.

    This heuristic sets the number of sets at size $k$ to be

    $$m_k = m \frac{f(k)}{\sum_{j=0}^{n-1} f(j)},$$

    for a total number of samples $m$ and:

    $$ f(k) = \frac{1}{k} + \frac{1}{n-k}, \text{for} k \in \{1, n-1\}. $$

    For GT Shapley, $m=1$ and $m_k$ is interpreted as a probability of sampling size
    $k.$
    """

    def fun(self, n_indices: int, subset_len: int) -> float:
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

    and some exponent $a.$ With $a=-1$ one recovers the
    [HarmonicSampleSize][pydvl.valuation.samplers.stratified.HarmonicSampleSize]
    heuristic. With $a=-2$ one has the same asymptotic behaviour as the
    $\delta$-Shapley strategies.

    Args:
        n_samples: Total number of samples to generate **per index**.
        exponent: The exponent to use. Recommended values are between -1 and -0.5.
    """

    def __init__(
        self,
        exponent: float,
        n_samples: int | None = None,
        lower_bound: int | None = None,
        upper_bound: int | None = None,
    ):
        super().__init__(n_samples, lower_bound, upper_bound)
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


class FiniteSequentialSizeIteration(SampleSizeIteration):
    """Generates exactly $m_k$ samples for each set size $k$ before moving to the next."""

    def __iter__(self) -> Generator[tuple[int, int], None, None]:
        counts = self.strategy.sample_sizes(self.n_indices, probs=False)
        if self.n_indices > 1 and np.sum(counts) <= 1:
            raise ValueError(
                f"{self.strategy.__class__.__name__} seems to only provide "
                f"probabilities. Ensure you set up the strategy with a fixed "
                f"number of samples per index greater than 1."
            )

        for k, m_k in enumerate(counts):  # type: int, int
            if m_k > 0:
                yield k, max(1, m_k)


class RandomSizeIteration(SampleSizeIteration):
    """Draws a set size $k$ following the distribution of sizes given by the strategy."""

    def __init__(
        self, strategy: SampleSizeStrategy, n_indices: int, seed: Seed | None = None
    ):
        super().__init__(strategy, n_indices)
        self._rng = np.random.default_rng(seed)

    def __iter__(self) -> Generator[tuple[int, int], None, None]:
        # In stochastic mode we interpret the counts as weights to sample one k.
        probs = self.strategy.sample_sizes(self.n_indices, probs=True)
        if np.sum(probs) == 0:
            yield 0, 0
        k = self._rng.choice(np.arange(self.n_indices + 1), p=probs)
        yield k, 1


class RoundRobinIteration(SampleSizeIteration):
    """Generates one sample for each set size $k$ before moving to the next.

    This continues yielding until every size $k$ has been emitted exactly $m_k$ times.
    For example, if `strategy.sample_sizes() == [2, 3, 1]` then we want the sequence:
    (0,1), (1,1), (2,1), (0,1), (1,1), (1,1)

    !!! warning "Only for deterministic sample sizes"
        This iteration is only valid for deterministic sample sizes. In particular, the
        strategy must support `quantize=True` and `n_samples` must be set to the total
        number of samples.
    """

    def __iter__(self) -> Generator[tuple[int, int], None, None]:
        counts = self.strategy.sample_sizes(self.n_indices, probs=False).copy()
        if self.n_indices > 1 and np.sum(counts) <= 1:
            raise ValueError(
                f"{self.strategy.__class__.__name__} seems to only provide "
                f"probabilities. Ensure you set up the strategy with a fixed "
                f"number of samples per index greater than 1."
            )
        while any(count > 0 for count in counts):
            for k, count in enumerate(counts):  # type: int, int
                if count > 0:
                    counts[k] -= 1
                    yield k, 1


class StratifiedSampler(StochasticSamplerMixin, PowersetSampler):
    """A sampler stratified by coalition size with variable number of samples per set
    size.


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
        sample_sizes_iteration: Type[
            SampleSizeIteration
        ] = FiniteSequentialSizeIteration,
        batch_size: int = 1,
        index_iteration: Type[IndexIteration] = FiniteSequentialIndexIteration,
        seed: Seed | None = None,
    ):
        super().__init__(
            batch_size=batch_size, index_iteration=index_iteration, seed=seed
        )
        self.sample_sizes_strategy = sample_sizes
        self.sample_sizes_iteration = maybe_add_argument(sample_sizes_iteration, "seed")

    def generate(self, indices: IndexSetT) -> SampleGenerator:
        m = self._index_iterator_cls.complement_size(len(indices))
        sample_sizes_iterable = self.sample_sizes_iteration(
            self.sample_sizes_strategy, m, seed=self._rng
        )
        for idx in self.index_iterator(indices):
            from_set = complement(indices, [idx])
            for k, m_k in sample_sizes_iterable:
                for _ in range(m_k):
                    subset = random_subset_of_size(from_set, size=k, seed=self._rng)
                    yield Sample(idx, subset)

    def sample_limit(self, indices: IndexSetT) -> int | None:
        index_iteration_length = self._index_iterator_cls.length(len(indices))
        if index_iteration_length is None:
            return None
        # Compute n_samples as side effect (yuk!):
        _ = self.sample_sizes_strategy.sample_sizes(len(indices))
        assert self.sample_sizes_strategy.n_samples is not None
        return index_iteration_length * self.sample_sizes_strategy.n_samples

    def log_weight(self, n: int, subset_len: int) -> float:
        r"""The probability of sampling a set of size k is 1/(n choose k) times the
        probability of the set having size k, which is the number of samples for that
        size divided by the total number of samples for all sizes:

        $$P(S) = \binom{n}{k}^{-1} \ \frac{m_k}{m},$$

        where $m_k$ is the number of samples of size $k$ and $m$ is the total number
        of samples.

        Args:
            n: Size of the index set.
            subset_len: Size of the subset.
        Returns:
            The logarithm of the probability of having sampled a set of size `subset_len`.
        """

        effective_n = self._index_iterator_cls.complement_size(n)
        # Depending on whether we sample from complements or not, the total number of
        # samples passed to the heuristic has a different interpretation.
        index_iteration_length = self._index_iterator_cls.length(effective_n)  # type: ignore
        if index_iteration_length is None:
            index_iteration_length = 1
        index_iteration_length = max(1, index_iteration_length)

        # Note that we can simplify the quotient
        # $$ \frac{m_k}{m} =
        #    \frac{m \frac{f (k)}{\sum_j f (j)}}{m} = \frac{f(k)}{\sum_j f (j)} $$
        # so that in the weight computation we can use the function $f$ directly from
        # the strategy, or equivalently, call `sample_sizes(n, probs=True)`.
        # This is useful for the stochastic iteration, where we are given sampling
        # frequencies for each size instead of counts, and the total number of samples
        # m is 1, so that quantization would yield a bunch of zeros.
        f = self.sample_sizes_strategy.sample_sizes(effective_n, probs=True)
        f_k = f[subset_len]
        assert np.isclose(np.sum(f), 1.0), (
            f"Strategy returned invalid probabilities, adding to {np.sum(f)=}"
        )

        if f_k == 0:
            return -np.inf

        return float(
            -logcomb(effective_n, subset_len)
            + np.log(index_iteration_length)
            + np.log(f_k)
        )


class VRDSSampler(StratifiedSampler):
    """A sampler stratified by coalition size with variable number of samples per set
    size.

    This sampler iterates once per index and generates a fixed mount of subsets of each
    size in its complement.

    This is a convenience subclass of
    [StratifiedSampler][pydvl.valuation.samplers.StratifiedSampler]
    which implements the VRDS heuristic from Wu et al. (2023)<sup><a
    href="#wu_variance_2023">2</a></sup>.

    It is functionally equivalent to a
    [StratifiedSampler][pydvl.valuation.samplers.StratifiedSampler] with
    [HarmonicSampleSize][pydvl.valuation.samplers.stratified.HarmonicSampleSize],
    [DeterministicSizeIteration][pydvl.valuation.samplers.stratified.DeterministicSizeIteration], and
    [FiniteSequentialIndexIteration][pydvl.valuation.samplers.powerset.FiniteSequentialIndexIteration].

    Args:
        n_samples_per_index: The number of samples to generate **per index**. The
            distribution per set size will follow a harmonic function, as defined in
            [HarmonicSampleSize][pydvl.valuation.samplers.stratified.HarmonicSampleSize].
        batch_size: The number of samples to generate per batch. Batches are processed
            together by each subprocess when working in parallel.
        seed: The seed for the random number generator.
    """

    def __init__(
        self,
        n_samples_per_index: int,
        batch_size: int = 1,
        seed: Seed | None = None,
    ):
        super().__init__(
            sample_sizes=HarmonicSampleSize(n_samples=n_samples_per_index),
            sample_sizes_iteration=FiniteSequentialSizeIteration,
            batch_size=batch_size,
            index_iteration=FiniteSequentialIndexIteration,
            seed=seed,
        )
