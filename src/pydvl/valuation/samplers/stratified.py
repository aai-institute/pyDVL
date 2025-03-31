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
classes [StratifiedSampler][pydvl.valuation.samplers.stratified.StratifiedSampler] and
[StratifiedPermutationSampler][pydvl.valuation.samplers.stratified.StratifiedPermutationSampler].

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
    $\sim n/2$ while increasing the number of sets of size 1 or $n-1.$ This will then
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
    It is possible to "manually" replicate
    [VRDSSampler][pydvl.valuation.samplers.stratified.VRDSSampler] with:

    ```python
    n_samples_per_index = 1000  # Total number of samples is: n_indices times this
    sampler = StratifiedSampler(
        sample_sizes=HarmonicSampleSize(n_samples=1000),
        sample_sizes_iteration=FiniteSequentialSizeIteration,
        index_iteration=FiniteSequentialIndexIteration,
        )
    ```

## Iterating over indices and its effect on 'n_samples'

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

* [FiniteSequentialSizeIteration][pydvl.valuation.samplers.stratified.FiniteSequentialSizeIteration]
  will generate exactly $m_k$ samples for each $k$ before moving to the next $k.$ This
  implies that `n_samples` must be large enough for the computed $m_k$ to be valid.
  Alternatively, and preferably, some strategies allow `n_samples = None` to signal them
  to compute the total number of samples.
* [RandomSizeIteration][pydvl.valuation.samplers.stratified.RandomSizeIteration] will
  sample a set size $k$ according to the distribution of sizes given by the strategy.
  When using this in conjunction with an infinite index iteration for the sampler,
  `n_samples` can be safely left to its default `None` since $m_k$ will be interpreted
  as a probability.
* [RoundRobinSizeIteration][pydvl.valuation.samplers.stratified.RoundRobinSizeIteration] will
  iterate over set sizes $k$ and generate one sample each time, until reaching $m_k.$

## Other known strategies

All components described above can be mixed in most ways, but some specific
configurations besides VRDS appear in the literature as follows:

* Sample sizes given by stability bounds related to the algorithm, to ensure good
  approximation of per-set-size marginal utilities. This sampling method was introduced
  by Watson et al. (2023)[^1] for the computation of Shapley values as
  [$\delta$-Shapley][delta-shapley-intro].

  [DeltaShapleyNSGDSampleSize][pydvl.valuation.samplers.stratified.DeltaShapleyNCSGDSampleSize]
  implements the choice of $m_k$ corresponding to non-convex losses minimized with SGD.
  Alas, it requires many parameters to be set which are hard to estimate in practice. An
  alternative is to use
  [PowerLawSampleSize][pydvl.valuation.samplers.stratified.PowerLawSampleSize] (see
  below) with an exponent of -2, which is the order of $m_k$ in $\delta$-Shapley.

    ??? Example "Constructing an alternative sampler for DeltaShapley"
        ```python
        config = DeltaShapleyNCSGDConfig(...)  # See the docs / paper
        sampler = StratifiedSampler(
            sample_sizes=DeltaShapleyNSGDSampleSize(config, lower_bound, upper_bound),
            sample_sizes_iteration=FiniteSequentialSizeIteration,
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
              sample_sizes=PowerLawSampleSize(exponent=-0.5),
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

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache
from typing import Generator, Literal, Type

import numpy as np
from numpy.typing import NDArray

from pydvl.utils import (
    Seed,
    complement,
    logcomb,
    maybe_add_argument,
    random_subset_of_size,
    suppress_warnings,
)
from pydvl.valuation.samplers.permutation import (
    PermutationEvaluationStrategy,
    PermutationSampler,
    TruncationPolicy,
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
from pydvl.valuation.utility.base import UtilityBase

__all__ = [
    "ConstantSampleSize",
    "FiniteSequentialSizeIteration",
    "DeltaShapleyNCSGDConfig",
    "DeltaShapleyNCSGDSampleSize",
    "GroupTestingSampleSize",
    "HarmonicSampleSize",
    "PowerLawSampleSize",
    "RandomSizeIteration",
    "RoundRobinSizeIteration",
    "SampleSizeIteration",
    "SampleSizeStrategy",
    "StratifiedPermutationSampler",
    "StratifiedPermutationEvaluationStrategy",
    "StratifiedSampler",
    "VRDSSampler",
]

logger = logging.getLogger(__name__)


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
            then the total number of samples will be `n_samples * n_indices`. Leave as
            `None` when a fixed number is unnecessary, e.g. when using a stochastic
            sampler and a
            [RandomSizeIteration][pydvl.valuation.samplers.stratified.RandomSizeIteration].
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
        self.n_samples_per_index = n_samples
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

    def effective_bounds(self, n: int) -> tuple[int, int]:
        """Returns the effective bounds for the sample sizes, given the number of
        indices `n` from which sets are sampled.

        !!! note
            The number of indices `n` will typically be `complement_size(len(train))`,
            i.e. what we sometimes denote as `effective_n`.

        Args:
            n: The number of indices from which subsets are drawn.
        Returns:
            A tuple of [lower, upper] bounds for sample sizes (inclusive).
        """
        lower = 0 if self.lower_bound is None else self.lower_bound
        upper = n if self.upper_bound is None else self.upper_bound
        lower = min(lower, n)
        upper = min(upper, n)

        return lower, upper

    @cache
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
        elements is equal to `self.n_samples_per_index` if provided upon construction,
        or the sum of the values of `fun` for all set sizes if
        `self.n_samples_per_index` is `None`.

        When `probs` is `False`, this method corrects rounding errors taking into
        account the fractional parts so that the total number of samples is respected,
        while allocating remainders in a way that follows the relative sizes of the
        fractional parts.

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
        lb, ub = self.effective_bounds(n_indices)

        for k in range(lb, ub + 1):
            val = self.fun(n_indices, k)
            values[k] = val
            s += val

        assert n_indices == 0 or s > 0, "Sum of sample sizes must be positive"
        values /= s

        n_samples = self.n_samples_per_index or int(np.ceil(s))

        if probs:
            return values  # m_k / m

        values *= n_samples
        # Round down and distribute remainder by adjusting the largest fractional parts
        # A naive implementation with e.g.
        #
        # m_k = [max(1, int(round(m * f(k)/sum(f(j) for j in range(n)), 0)))
        #         for k in range(n)]
        #
        # would not respect the total number of samples, and would not distribute
        # remainders correctly
        if n_samples < len(np.nonzero(values)[0]):
            raise ValueError(
                f"Number of samples per index {n_samples} is smaller than the "
                f"number of non-zero sample sizes {len(np.nonzero(values)[0])}. "
                f"Increase `n_samples` when instantiating {str(self)}, or use a"
                f"stochastic size / index iteration and sampler."
            )
        int_values: NDArray[np.int64] = np.floor(values).astype(np.int64)
        remainder = n_samples - np.sum(int_values)
        fractional_parts = values - int_values
        fractional_parts_indices = np.argsort(-fractional_parts, kind="stable")[
            :remainder
        ]
        int_values[fractional_parts_indices] += 1
        return int_values

    def __str__(self):
        return self.__class__.__name__


class ConstantSampleSize(SampleSizeStrategy):
    r"""Use a constant number of samples for each set size between two (optional)
    bounds. The total number of samples (per index) is respected.
    """

    def fun(self, n_indices: int, subset_len: int) -> float:
        return 1.0


# This otherwise unnecessary class can be convenient for passing around and storing
# all the parameters.
@dataclass
class DeltaShapleyNCSGDConfig:
    """Configuration for Delta-Shapley non-convex SGD sampling.

    See Watson et al. (2023)<sup><a href="#watson_accelerated_2023">1</a></sup> for
    details. Given that it can be difficult to estimate these constants, an alternative
    which has a similar decay rate of $O(1/k^2)$ is to use a
    [PowerLawSampleSize][pydvl.valuation.samplers.stratified.PowerLawSampleSize]
    strategy.

    Args:
        max_loss: Maximum of the loss.
        lipschitz_loss: Lipschitz constant of the loss
        lipschitz_grad: Lipschitz constant of the gradient of the loss
        lr_factor: Learning rate factor c, assuming it has the form $\alpha_t = c/t.$
        n_sgd_iter: Number of SGD iterations.
        n_val: Number of test samples.
        n_train: Number of training samples.
        eps: Epsilon value in the epsilon-delta guarantee, i.e. the distance to the
            true value.
        delta: Delta value in the epsilon-delta guarantee, i.e. the probability of
            failure.
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


class RoundRobinSizeIteration(SampleSizeIteration):
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
        effective_n = self.complement_size(len(indices))
        sample_sizes_iterable = self.sample_sizes_iteration(
            self.sample_sizes_strategy, effective_n, seed=self._rng
        )
        for idx in self.index_iterable(indices):
            from_set = complement(indices, [idx])
            for k, m_k in sample_sizes_iterable:
                for _ in range(m_k):
                    subset = random_subset_of_size(from_set, size=k, seed=self._rng)
                    yield Sample(idx, subset)

    def sample_limit(self, indices: IndexSetT) -> int | None:
        index_iteration_length = self._index_iterator_cls.length(len(indices))
        if index_iteration_length is None:
            return None
        m = self.sample_sizes_strategy.sample_sizes(len(indices), probs=False)
        return index_iteration_length * sum(m)

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

        effective_n = self.complement_size(n)

        # Note that we can simplify the quotient
        # $$ \frac{m_k}{m} =
        #    \frac{m \frac{f (k)}{\sum_j f (j)}}{m} = \frac{f(k)}{\sum_j f (j)} $$
        # so that in the weight computation we can use the function $f$ directly from
        # the strategy, or equivalently, call `sample_sizes(n, probs=True)`.
        # This is useful for the stochastic iteration, where we are given sampling
        # frequencies for each size instead of counts, and the total number of samples
        # m is 1, so that quantization would yield a bunch of zeros.
        p = self.sample_sizes_strategy.sample_sizes(effective_n, probs=True)
        p_k = p[subset_len]  # also m_k / m
        if p_k == 0:
            return -np.inf

        return float(np.log(p_k) - logcomb(effective_n, subset_len))


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
    [HarmonicSampleSize][pydvl.valuation.samplers.stratified.HarmonicSamp leSize],
    [FiniteSequentialSizeIteration][pydvl.valuation.samplers.stratified.FiniteSequentialSizeIteration],
    and
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


@dataclass(frozen=True)
class StratifiedPermutation(Sample):
    """A sample for the stratified permutation sampling strategy.

    This is a subclass of [Sample][pydvl.valuation.types.Sample] which adds information
    about the set sizes to sample. It is used by
    [StratifiedPermutationEvaluationStrategy][pydvl.valuation.samplers.stratified.StratifiedPermutationEvaluationStrategy]
    to clip permutations to the required lengths.
    """

    lower_bound: int
    """ The lower bound for the set sizes."""

    upper_bound: int
    """ The upper bound for the set sizes."""


class StratifiedPermutationSampler(PermutationSampler):
    """A stratified permutation sampler.

    !!! warning "Experimental"
        This is just an approximation for now. The number of set sizes generated is only
        roughly equal to that specified by the
        [SampleSizeStrategy][pydvl.valuation.samplers.stratified.SampleSizeStrategy]. In
        particular, there is a **single counter of sizes for all indices**.

    Args:
        sample_sizes: An object which returns the number of samples to take for a given
            set size. This must be able to compute discrete numbers of samples, as
            opposed to only probabilities. Either choose one that generates integer
            sample sizes, or set `n_samples` manually upon its construction.
        truncation: A policy to stop the permutation early.
        seed: Seed for the random number generator.
        batch_size: The number of samples (full permutations) to generate at once.
    """

    def __init__(
        self,
        sample_sizes: SampleSizeStrategy,
        truncation: TruncationPolicy | None = None,
        seed: Seed | None = None,
        batch_size: int = 1,
    ):
        super().__init__(truncation, seed, batch_size)
        self.sample_sizes_strategy = sample_sizes
        logger.warning(
            "StratifiedPermutationSampler is experimental and inexact. "
            "Please use another sampler if you are benchmarking methods."
        )

    @property
    def skip_indices(self) -> IndexSetT:
        return self._skip_indices

    @skip_indices.setter
    def skip_indices(self, indices: IndexSetT):
        raise AttributeError(
            f"Cannot skip converged indices in {self.__class__.__name__}."
        )

    def sample_limit(self, indices: IndexSetT) -> int | None:
        m = self.sample_sizes_strategy.sample_sizes(len(indices), probs=False)
        return len(indices) * sum(m)

    def generate(self, indices: IndexSetT) -> SampleGenerator[StratifiedPermutation]:
        """Generates the permutation samples.

        These samples include information as to what sample sizes can be taken from the
        permutation by the strategy.

        !!! info
            This generator ignores `skip_indices`.

        Args:
            indices: The indices to sample from. If empty, no samples are generated.
        """
        n = len(indices)
        if n == 0:
            return
        sizes = self.sample_sizes_strategy.sample_sizes(n, probs=False)
        n_samples = np.sum(sizes)
        if n_samples <= 1:
            raise ValueError(
                f"{self.sample_sizes_strategy.__class__.__name__} seems to only provide "
                f"probabilities. Ensure you set up the strategy with a fixed "
                f"number of samples per index greater than 1."
            )

        # FIXME: This is just an approximation. On expectation we should produce roughly
        #   the correct number of sizes per index, but we should probably keep track
        #   separately.
        sizes *= n

        while True:
            # Can't have skip indices: if the index set is smaller than the lower bound
            # for the set sizes, the strategy's process() will always return empty
            # evaluations and the method might never stop with criteria depending on the
            # number of updates
            # _indices = np.setdiff1d(indices, self.skip_indices)

            positive = np.where(sizes > 0)[0]
            if len(positive) == 0:
                break
            lb, ub = int(positive[0]), int(positive[-1])
            assert all(sizes[lb : ub + 1] > 0), "Sample size function must be monotonic"
            sizes -= 1

            yield StratifiedPermutation(
                idx=None,
                subset=self._rng.permutation(indices),
                lower_bound=lb,
                upper_bound=ub,
            )

    def log_weight(self, n: int, subset_len: int) -> float:
        """The probability of sampling a set of size `subset_len` from `n` indices.

        See
        [StratifiedSampler.log_weight()][pydvl.valuation.samplers.stratified.StratifiedSampler.log_weight]

        Args:
            n:  Size of the index set.
            subset_len: Size of the subset.

        Returns:
            The logarithm of the probability of having sampled a set of size
                `subset_len`.
        """
        effective_n = self.complement_size(n)
        p = self.sample_sizes_strategy.sample_sizes(effective_n, probs=True)
        p_k = p[subset_len]
        if p_k == 0:
            return -np.inf

        return float(np.log(p_k) - logcomb(effective_n, subset_len))

    def make_strategy(
        self, utility: UtilityBase, coefficient: SemivalueCoefficient | None = None
    ) -> StratifiedPermutationEvaluationStrategy:
        return StratifiedPermutationEvaluationStrategy(
            sampler=self,
            utility=utility,
            coefficient=coefficient,
        )


class StratifiedPermutationEvaluationStrategy(PermutationEvaluationStrategy):
    """Evaluation strategy for the
    [StratifiedPermutationSampler][pydvl.valuation.samplers.stratified.StratifiedPermutationSampler].

    !!! warning "Experimental"
    """

    @suppress_warnings(categories=(RuntimeWarning,), flag="show_warnings")
    def process(
        self, batch: SampleBatch, is_interrupted: NullaryPredicate
    ) -> list[ValueUpdate]:
        r = []
        for sample in batch:
            lb, ub = sample.lower_bound, sample.upper_bound
            self.truncation.reset(self.utility)
            truncated = False
            permutation = sample.subset
            if lb == 0:
                curr = prev = self.utility(None)
            else:
                first = sample.with_idx(None).with_subset(permutation[:lb])
                curr = prev = self.utility(first)
            for i, idx in enumerate(permutation[lb : ub + 1], start=lb):  # type: int, np.int_
                if not truncated:
                    new_sample = sample.with_idx(idx).with_subset(permutation[: i + 1])
                    curr = self.utility(new_sample)
                marginal = curr - prev
                sign = np.sign(marginal)
                log_marginal = -np.inf if marginal == 0 else np.log(marginal * sign)
                log_marginal += self.valuation_coefficient(self.n_indices, i)
                r.append(ValueUpdate(idx, log_marginal, sign))
                prev = curr
                if not truncated and self.truncation(idx, curr, self.n_indices):
                    truncated = True
                if is_interrupted():
                    return r
        return r
