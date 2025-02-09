r"""
This module provides the base implementation for powerset samplers.

These samplers operate in two loops:

1. Outer iteration over all indices. This is configurable with subclasses of
   [IndexIteration][]. At each step we fix an index $i \in N$.
2. Inner iteration over subsets of $N_{-i}$. This step can return one or more subsets,
   sampled in different ways: uniformly, with varying probabilities, in tuples of
   complementary sets, etc.

This scheme follows the usual definition of semi-values as:

$$
v_\text{semi}(i) = \sum_{i=1}^n w(k)
                     \sum_{S \subset D_{-i}^{(k)}} [U(S_{+i})-U(S)],
$$

see [semivalues][pydvl.valuation.methods.semivalue] for reference.


## References

[^1]: <a name="mitchell_sampling_2022"></a>Mitchell, Rory, Joshua Cooper, Eibe
      Frank, and Geoffrey Holmes. [Sampling Permutations for Shapley Value
      Estimation](https://jmlr.org/papers/v23/21-0439.html). Journal of Machine
      Learning Research 23, no. 43 (2022): 1–46.
[^2]: <a name="maleki_bounding_2014"></a>Maleki, Sasan, Long Tran-Thanh, Greg Hines,
      Talal Rahwan, and Alex Rogers. [Bounding the Estimation Error of Sampling-Based
      Shapley Value Approximation](https://arxiv.org/abs/1306.4265). arXiv:1306.4265
      [Cs], 12 February 2014.

"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import lru_cache
from typing import Callable, Generator, Generic, Type, TypeVar

import numpy as np
from numpy.typing import NDArray

from pydvl.utils.numeric import (
    complement,
    powerset,
    random_subset,
    random_subset_of_size,
)
from pydvl.utils.types import Seed
from pydvl.valuation.samplers.base import EvaluationStrategy, IndexSampler
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
    "AntitheticSampler",
    "DeterministicUniformSampler",
    "LOOSampler",
    "PowersetSampler",
    "TruncatedUniformStratifiedSampler",
    "UniformSampler",
    "UniformStratifiedSampler",
    "HarmonicSamplesPerSetSize",
    "PowerLawSamplesPerSetSize",
    "VarianceReducedStratifiedSampler",
    "IndexIteration",
    "SequentialIndexIteration",
    "FiniteSequentialIndexIteration",
    "RandomIndexIteration",
    "FiniteRandomIndexIteration",
    "NoIndexIteration",
    "FiniteNoIndexIteration",
]


logger = logging.getLogger(__name__)


# Careful with MRO when using these and subclassing!
class FiniteIterationMixin:
    @staticmethod
    def length(indices: IndexSetT) -> int | None:
        return len(indices)


class InfiniteIterationMixin:
    @staticmethod
    def length(indices: IndexSetT) -> int | None:
        if len(indices) == 0:
            return 0
        return None


class IndexIteration(ABC):
    def __init__(self, indices: IndexSetT):
        self._indices = indices

    @abstractmethod
    def __iter__(self) -> Generator[IndexT | None, None, None]: ...

    @staticmethod
    @abstractmethod
    def length(indices: IndexSetT) -> int | None:
        """Returns the length of the iteration over the index set

        Args:
            indices: The set of indices to iterate over.

        Returns:
            The length of the iteration. It can be:
                - 0, if the index set is ignored or there are no indices.
                - a positive integer, if the iteration is finite
                - `None` if the iteration never ends.
        """
        ...

    @staticmethod
    @abstractmethod
    def complement_size(n: int) -> int:
        """Returns the size of complements of sets of size n, with respect to the
        indices returned by the iteration.

        If the iteration returns single indices, then this is n-1, if it returns no
        indices, then it is n. If it returned tuples, then n-2, etc.
        """
        ...

    @classmethod
    def is_finite(cls) -> bool:
        return cls.length(np.array([1])) == 1


class SequentialIndexIteration(InfiniteIterationMixin, IndexIteration):
    """Samples indices sequentially, indefinitely."""

    def __iter__(self) -> Generator[IndexT, None, None]:
        while True:
            yield from self._indices

    @staticmethod
    def complement_size(n: int) -> int:
        return n - 1


class FiniteSequentialIndexIteration(FiniteIterationMixin, SequentialIndexIteration):
    """Samples indices sequentially, once."""

    def __iter__(self) -> Generator[IndexT, None, None]:
        if len(self._indices) == 0:
            return
        yield from self._indices


class RandomIndexIteration(
    InfiniteIterationMixin, StochasticSamplerMixin, IndexIteration
):
    """Samples indices at random, indefinitely"""

    def __init__(self, indices: NDArray[IndexT], seed: Seed):
        super().__init__(indices, seed=seed)

    def __iter__(self) -> Generator[IndexT, None, None]:
        if len(self._indices) == 0:
            return
        while True:
            yield self._rng.choice(self._indices, size=1).item()

    @staticmethod
    def complement_size(n: int) -> int:
        return n - 1


class FiniteRandomIndexIteration(FiniteIterationMixin, RandomIndexIteration):
    """Samples indices at random, once"""

    def __iter__(self) -> Generator[IndexT, None, None]:
        if len(self._indices) == 0:
            return
        yield from self._rng.choice(self._indices, size=len(self._indices))


class NoIndexIteration(InfiniteIterationMixin, IndexIteration):
    """An infinite iteration over no indices."""

    def __iter__(self) -> Generator[None, None, None]:
        while True:
            yield None

    @staticmethod
    def complement_size(n: int) -> int:
        return n


class FiniteNoIndexIteration(FiniteIterationMixin, NoIndexIteration):
    """A finite iteration over no indices."""

    def __iter__(self) -> Generator[None, None, None]:
        yield None

    @staticmethod
    def complement_size(n: int) -> int:
        return n


class PowersetSampler(IndexSampler, ABC):
    """An abstract class for samplers which iterate over the powerset of the
    complement of an index in the training set.

    This is done in two nested loops, where the outer loop iterates over the set
    of indices, and the inner loop iterates over subsets of the complement of
    the current index. The outer iteration can be either sequential or at random.
    """

    def __init__(
        self,
        batch_size: int = 1,
        index_iteration: Type[IndexIteration] = SequentialIndexIteration,
    ):
        """
        Args:
            batch_size: The number of samples to generate per batch. Batches are
                processed together by
                [UtilityEvaluator][pydvl.valuation.utility.evaluator.UtilityEvaluator].
            index_iteration: the strategy to use for iterating over indices to update
        """
        super().__init__(batch_size)
        self._index_iterator_cls = index_iteration
        self._index_iterator: IndexIteration | None = None

    def index_iterator(
        self, indices: IndexSetT
    ) -> Generator[IndexT | None, None, None]:
        """Iterates over indices with the method specified at construction."""
        try:
            self._index_iterator = self._index_iterator_cls(indices, seed=self._rng)  # type: ignore
        except (AttributeError, TypeError):
            self._index_iterator = self._index_iterator_cls(indices)
        yield from self._index_iterator

    def make_strategy(
        self,
        utility: UtilityBase,
        coefficient: Callable[[int, int, float], float] | None = None,
    ) -> PowersetEvaluationStrategy:
        return PowersetEvaluationStrategy(self, utility, coefficient)

    @abstractmethod
    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        """Generates samples iterating in sequence over the outer indices, then over
        subsets of the complement of the current index. Each PowersetSampler defines
        its own
        [subset_iterator][pydvl.valuation.samplers.PowersetSampler.subset_iterator] to
        generate the subsets.

        Args:
            indices:"""
        ...

    def weight(self, n: int, subset_len: int) -> float:
        """Correction coming from Monte Carlo integration so that the mean of
        the marginals converges to the value: the uniform distribution over the
        powerset of a set with n-1 elements has mass 1/2^{n-1} over each subset."""
        n = self._index_iterator_cls.complement_size(n)
        return 2**n if n > 0 else 1  # type: ignore


PowersetSamplerT = TypeVar("PowersetSamplerT", bound=PowersetSampler)


class PowersetEvaluationStrategy(
    Generic[PowersetSamplerT], EvaluationStrategy[PowersetSamplerT, ValueUpdate]
):
    def process(
        self, batch: SampleBatch, is_interrupted: NullaryPredicate
    ) -> list[ValueUpdate]:
        updates = []
        for sample in batch:
            u_i = self.utility(sample.with_idx_in_subset())
            u = self.utility(sample)
            marginal = (u_i - u) * self.coefficient(self.n_indices, len(sample.subset))
            updates.append(ValueUpdate(sample.idx, marginal))
            if is_interrupted():
                break
        return updates


class LOOSampler(PowersetSampler):
    """Leave-One-Out sampler.

    In this special case of a powerset samplet, for every index $i$ in the set $S$, the
    sample $(i, S_{-i})$ is returned.

    Args:
        batch_size: The number of samples to generate per batch. Batches are processed
            together by each subprocess when working in parallel.
        index_iteration: the strategy to use for iterating over indices to update. By
            default, a finite sequential index iteration is used, which is what
            [LOOValuation][pydvl.valuation.methods.loo.LOOValuation] expects.
        seed: The seed for the random number generator used in case the index iteration
            is random.

    !!! tip "New in version 0.10.0"
    """

    def __init__(
        self,
        batch_size: int = 1,
        index_iteration: Type[IndexIteration] = FiniteSequentialIndexIteration,
        seed: Seed | None = None,
    ):
        super().__init__(batch_size, index_iteration)
        if issubclass(index_iteration, NoIndexIteration):
            raise ValueError("LOO samplers require a valid index iteration strategy")
        self._rng = np.random.default_rng(seed)

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        for idx in self.index_iterator(indices):
            yield Sample(idx, complement(indices, [idx]))

    def weight(self, n: int, subset_len: int) -> float:
        """This sampler returns only sets of size n-1. There are n such sets, so the
        probability of drawing one is 1/n, or 0 if subset_len != n-1."""
        return n if subset_len == n - 1 else 0

    def make_strategy(
        self,
        utility: UtilityBase,
        coefficient: Callable[[int, int, float], float] | None = None,
    ) -> PowersetEvaluationStrategy[LOOSampler]:
        return LOOEvaluationStrategy(self, utility, coefficient)

    def sample_limit(self, indices: IndexSetT) -> int | None:
        return self._index_iterator_cls.length(indices)


class LOOEvaluationStrategy(PowersetEvaluationStrategy[LOOSampler]):
    """Computes marginal values for LOO."""

    def __init__(
        self,
        sampler: LOOSampler,
        utility: UtilityBase,
        coefficient: Callable[[int, int, float], float] | None = None,
    ):
        super().__init__(sampler, utility, coefficient)
        assert utility.training_data is not None
        self.total_utility = utility(Sample(None, utility.training_data.indices))

    def process(
        self, batch: SampleBatch, is_interrupted: NullaryPredicate
    ) -> list[ValueUpdate]:
        updates = []
        for sample in batch:
            assert sample.idx is not None
            u = self.utility(sample)
            marginal = self.total_utility - u
            marginal *= self.coefficient(self.n_indices, len(sample.subset))
            updates.append(ValueUpdate(sample.idx, marginal))
            if is_interrupted():
                break
        return updates


class DeterministicUniformSampler(PowersetSampler):
    """An iterator to perform uniform deterministic sampling of subsets.

    For every index $i$, each subset of the complement `indices - {i}` is
    returned.

    Args:
        batch_size: The number of samples to generate per batch. Batches are processed
            together by each subprocess when working in parallel.
        index_iteration: the strategy to use for iterating over indices to update. This
            iteration can be either finite or infinite.

    ??? Example
        The code:

        ```python
        from pydvl.valuation.samplers import DeterministicUniformSampler
        import numpy as np
        sampler = DeterministicUniformSampler()
        for idx, s in sampler.generate_batches(np.arange(2)):
            print(f"{idx} - {s}", end=", ")
        ```

        Should produce the output:

        ```
        1 - [], 1 - [2], 2 - [], 2 - [1],
        ```
    """

    def __init__(
        self,
        batch_size: int = 1,
        index_iteration: Type[IndexIteration] = FiniteSequentialIndexIteration,
    ):
        super().__init__(batch_size=batch_size, index_iteration=index_iteration)

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        for idx in self.index_iterator(indices):
            for subset in powerset(complement(indices, [idx])):
                yield Sample(idx, np.asarray(subset, dtype=indices.dtype))

    def sample_limit(self, indices: IndexSetT) -> int | None:
        len_outer = self._index_iterator_cls.length(indices)
        if len(indices) == 0:  # Empty index set
            return 0
        elif len_outer is None:  # Infinite index iteration
            return None
        elif len_outer == 0:  # No iteration over indices
            return int(2 ** len(indices))
        else:  # SequentialIndexIteration or other finite index iteration
            return int(len_outer * 2 ** (len(indices) - 1))


class UniformSampler(StochasticSamplerMixin, PowersetSampler):
    """Draws random samples uniformly from the powerset of the index set.

    Iterating over every index $i$, either in sequence or at random depending on
    the value of ``index_iteration``, one subset of the complement
    ``indices - {i}`` is sampled with equal probability $2^{n-1}$.

    Args:
        batch_size: The number of samples to generate per batch. Batches are processed
            together by each subprocess when working in parallel.
        index_iteration: the strategy to use for iterating over indices to update. This iteration
            can be either finite or infinite.
        seed: The seed for the random number generator.

    ??? Example
        The code
        ```python
        for idx, s in UniformSampler(np.arange(3)):
           print(f"{idx} - {s}", end=", ")
        ```
        Produces the output:
        ```
        0 - [1 4], 1 - [2 3], 2 - [0 1 3], 3 - [], 4 - [2], 0 - [1 3 4], 1 - [0 2]
        (...)
        ```
    """

    def __init__(
        self,
        batch_size: int = 1,
        index_iteration: Type[IndexIteration] = SequentialIndexIteration,
        seed: Seed | None = None,
    ):
        super().__init__(
            batch_size=batch_size, index_iteration=index_iteration, seed=seed
        )

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        for idx in self.index_iterator(indices):
            subset = random_subset(complement(indices, [idx]), seed=self._rng)
            yield Sample(idx, subset)


class AntitheticSampler(StochasticSamplerMixin, PowersetSampler):
    """A sampler that draws samples uniformly and their complements.

    Works as [UniformSampler][pydvl.valuation.samplers.UniformSampler], but for every
    tuple $(i,S)$, it subsequently returns $(i,S^c)$, where $S^c$ is the
    complement of the set $S$ in the set of indices, excluding $i$.
    """

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        for idx in self.index_iterator(indices):
            _complement = complement(indices, [idx])
            subset = random_subset(_complement, seed=self._rng)
            yield Sample(idx, subset)
            yield Sample(idx, complement(_complement, subset))


class TruncatedUniformStratifiedSampler(StochasticSamplerMixin, PowersetSampler):
    r"""A sampler which first samples set sizes between two bounds, then subsets of that
    size.

    For every index, this sampler first draws a set size between the two lower bounds
    where n is the total number of indices. Then a set of that size is sampled uniformly
    from the complement of the current index.

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


class SamplesPerSetSizeHeuristic:
    r"""A function which returns the number of samples to take for a given set size.
    Based on Wu et al. (2023), Theorem 4.2

    To be used in the
    [VarianceReducedStratifiedSampler][pydvl.valuation.samplers.VarianceReducedStratifiedSampler].

    Sets the number of sets at size $k$ to be

    $$m(k) = m \frac{f(k)}{\sum_{j=0}^{n} f(j)},$$

    for some choice of $f$.

    Args:
        n_samples_per_index: Number of samples for VRDS to generate, _per index_. If
            the sampler uses
            [NoIndexIteration][pydvl.valuation.samplers.NoIndexIteration],
            then this is the total number of samples.
    """

    def __init__(self, n_samples_per_index: int):
        self.n_samples_per_index = n_samples_per_index

    @abstractmethod
    def fun(self, subset_len: int) -> float:
        """The function $f$ to use in the heuristic."""
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
        s = sum(self.fun(k) for k in range(n_indices))
        values = [self.n_samples_per_index * self.fun(k) / s for k in range(n_indices)]

        # Round down and distribute remainder by adjusting the largest fractional parts
        int_values = [int(m) for m in values]
        remainder = self.n_samples_per_index - sum(int_values)
        fractional_parts = [(v - int(v), i) for i, v in enumerate(values)]
        fractional_parts.sort(reverse=True, key=lambda x: x[0])
        for i in range(remainder):
            int_values[fractional_parts[i][1]] += 1

        return int_values

    def __call__(self, n_indices: int, subset_len: int) -> int:
        """
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


class HarmonicSamplesPerSetSize(SamplesPerSetSizeHeuristic):
    r"""Heuristic choice of samples per set size for VRDS.

    Sets the number of sets at size $k$ to be

    $$m(k) = m \frac{f(k)}{\sum_{j=0}^{n-1} f(j)},$$

    for

    $$f(k) = \frac{1}{1+k}.$$

    """

    def fun(self, subset_len: int):
        return 1 / (1 + subset_len)


class PowerLawSamplesPerSetSize(SamplesPerSetSizeHeuristic):
    r"""Heuristic choice of samples per set size for VRDS.

    Sets the number of sets at size $k$ to be

    $$m(k) = m \frac{f(k)}{\sum_{j=0}^{n-1} f(j)},$$

    for

    $$f(k) = (1+k)^a, $$

    and some exponent $a$. With $a=1$ one recovers the
    [HarmonicSamplesPerSetSize][pydvl.valuation.samplers.HarmonicSamplesPerSetSize]
    heuristic.

    Args:
        n_samples_per_index: Total number of samples for VRDS to generate.
        exponent: The exponent to use. Recommended values are between -1 and -0.5.
    """

    def __init__(self, n_samples_per_index: int, exponent: float):
        super().__init__(n_samples_per_index)
        self.exponent = exponent

    def fun(self, subset_len):
        return (1 + subset_len) ** self.exponent


class VarianceReducedStratifiedSampler(StochasticSamplerMixin, PowersetSampler):
    """A sampler stratified by coalition size with variable number of samples per set
    size.

    Variance Reduced Stratified Sampler (VRDS) was suggested by Wu et al. 2023<sup><a
    href="#wu_variance_2023">3</a></sup>, as a generalization of the stratified
    sampler in Maleki et al. 2014<sup><a href="#maleki_bounding_2014">4</a></sup>.

    ## Choosing the number of samples per set size

    The idea of VRDS is to allow per-set-size configuration of the total number of
    samples in order to reduce the variance coming from the marginal utility evaluations.

    It is known (Wu et al. 2023, Theorem 4.2) that a minimum variance estimator of
    Shapley values samples a number $m_k$ of sets of size $k$ based on the variance of
    the marginal utility at that set size. However, this quantity is unknown in
    practice, so the authors propose a simple heuristic. This function
    (`samples_per_setsize` in the arguments) is deterministic, and in particular does
    not depend on run-time variance estimates, as an adaptive method might do. Section 4
    of Wu et al. 2023 shows that a choice of $1/(set_size+1)$ is a good default choice.
    where $m$ is the total number of samples
    to generate.

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
        samples_per_setsize: SamplesPerSetSizeHeuristic,
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
