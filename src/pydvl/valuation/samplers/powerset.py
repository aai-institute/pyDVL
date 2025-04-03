r"""
This module provides the base implementation for powerset samplers.

These samplers operate in two loops:

1. Outer iteration over all indices. This is configurable with subclasses of
   [IndexIteration][pydvl.valuation.samplers.powerset.IndexIteration]. At each step we
   fix an index $i \in N$.
2. Inner iteration over subsets of $N_{-i}$. This step can return one or more subsets,
   sampled in different ways: uniformly, with varying probabilities, in tuples of
   complementary sets, etc.

## Index iteration

The type of iteration over indices $i$ and their complements is configured upon
construction of the sampler with the classes
[SequentialIndexIteration][pydvl.valuation.samplers.powerset.SequentialIndexIteration],
[RandomIndexIteration][pydvl.valuation.samplers.powerset.RandomIndexIteration], or their
finite counterparts, when each index must be visited just once (albeit possibly
generating many samples per index).

However, some valuation schemes require iteration over subsets of the whole set (as
opposed to iterating over complements of individual indices). For this purpose, one can
use [NoIndexIteration][pydvl.valuation.samplers.powerset.NoIndexIteration] or its finite
counterpart.


!!! info "See also"
    [General information on semi-values][semi-values-intro] and, an explanation of
    [importance sampling in the context of semi-values][semi-values-sampling].


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
from abc import ABC, abstractmethod
from typing import Generator, Generic, Type, TypeVar

import numpy as np
from numpy.typing import NDArray

from pydvl.utils.functional import suppress_warnings
from pydvl.utils.numeric import (
    complement,
    powerset,
    random_subset,
)
from pydvl.utils.types import Seed
from pydvl.valuation.samplers.base import (
    EvaluationStrategy,
    IndexSampler,
)
from pydvl.valuation.samplers.utils import StochasticSamplerMixin
from pydvl.valuation.types import (
    IndexSetT,
    IndexT,
    NullaryPredicate,
    Sample,
    SampleBatch,
    SampleGenerator,
    SemivalueCoefficient,
    ValueUpdate,
)
from pydvl.valuation.utility.base import UtilityBase

__all__ = [
    "AntitheticSampler",
    "DeterministicUniformSampler",
    "LOOSampler",
    "PowersetSampler",
    "UniformSampler",
    "IndexIteration",
    "SequentialIndexIteration",
    "FiniteSequentialIndexIteration",
    "RandomIndexIteration",
    "FiniteRandomIndexIteration",
    "NoIndexIteration",
    "FiniteNoIndexIteration",
]


logger = logging.getLogger(__name__)


class FiniteIterationMixin:
    """Careful with MRO when using this and subclassing!"""

    @staticmethod
    def length(n_indices: int) -> int | None:
        return n_indices


class InfiniteIterationMixin:
    """Careful with MRO when using this and subclassing!"""

    @staticmethod
    def length(n_indices: int) -> int | None:
        if n_indices == 0:
            return 0
        return None


class IndexIteration(ABC):
    """An index iteration defines the way in which the outer loop over indices in a
    [PowersetSampler][pydvl.valuation.samplers.powerset.PowersetSampler] is done.

    Iterations can be finite, infinite, at random, etc. Subclasses must implement
    certain methods to inform the samplers of the size of the complement, the number
    of iterations, etc.
    """

    def __init__(self, indices: IndexSetT):
        self._indices = indices

    @abstractmethod
    def __iter__(self) -> Generator[IndexT | None, None, None]: ...

    @staticmethod
    @abstractmethod
    def length(n_indices: int) -> int | None:
        """Returns the length of the iteration over the index set

        Args:
            n_indices: The size of the index set.

        Returns:
            The length of the iteration. It can be:
                - a non-negative integer, if the iteration is finite
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
        Args:
            n: The size of the index set.
        """
        ...

    @classmethod
    def is_finite(cls) -> bool:
        return cls.length(1) is not None

    @classmethod
    def is_proper(cls) -> bool:
        return cls.complement_size(1) < 1


class SequentialIndexIteration(InfiniteIterationMixin, IndexIteration):
    """Samples indices sequentially, indefinitely."""

    def __iter__(self) -> Generator[IndexT, None, None]:
        while True:
            yield from self._indices

    @staticmethod
    def complement_size(n: int) -> int:
        return n - 1 if n > 0 else 0


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
        return n - 1 if n > 0 else 0


class FiniteRandomIndexIteration(FiniteIterationMixin, RandomIndexIteration):
    """Samples indices at random, once"""

    def __iter__(self) -> Generator[IndexT, None, None]:
        if len(self._indices) == 0:
            return
        yield from self._rng.permutation(self._indices)


class NoIndexIteration(InfiniteIterationMixin, IndexIteration):
    """An infinite iteration over no indices."""

    def __iter__(self) -> Generator[None, None, None]:
        while True:
            yield None

    @staticmethod
    def complement_size(n: int) -> int:
        return n


class FiniteNoIndexIteration(FiniteIterationMixin, NoIndexIteration):
    """A finite iteration over no indices.

    The iterator will yield `None` once and then stop.
    """

    def __iter__(self) -> Generator[None, None, None]:
        yield None

    @staticmethod
    def length(n_indices: int) -> int | None:
        """Returns 1, as the iteration yields exactly one item (`None`)"""
        return 1

    @staticmethod
    def complement_size(n: int) -> int:
        return n


class PowersetSampler(IndexSampler, ABC):
    """An abstract class for samplers which iterate over the powerset of the
    complement of an index in the training set.

    This is done in two nested loops, where the outer loop iterates over the set
    of indices, and the inner loop iterates over subsets of the complement of
    the current index. The outer iteration can be either sequential or at random.
    Args:
        batch_size: The number of samples to generate per batch. Batches are
            processed together by `process()` in the evaluation strategy
            [PowersetEvaluationStrategy][pydvl.valuation.samplers.powerset.PowersetEvaluationStrategy].
        index_iteration: the strategy to use for iterating over indices to update.
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
                [EvaluationStrategy][pydvl.valuation.samplers.base.EvaluationStrategy].
            index_iteration: the strategy to use for iterating over indices to update
        """
        super().__init__(batch_size)
        self._index_iterator_cls = index_iteration

    def complement_size(self, n: int) -> int:
        return self._index_iterator_cls.complement_size(n)

    @property
    def skip_indices(self):
        """Set of indices to skip in the outer loop."""
        return self._skip_indices

    @skip_indices.setter
    def skip_indices(self, indices: IndexSetT):
        """(Most) Powerset samplers support skipping indices in the outer loop.

        Args:
            indices: The indices to skip.
        """
        self._skip_indices = indices

    def index_iterable(
        self, indices: IndexSetT
    ) -> Generator[IndexT | None, None, None]:
        """Iterates over indices with the method specified at construction."""
        try:
            iterable = self._index_iterator_cls(indices, seed=self._rng)  # type: ignore
        except (AttributeError, TypeError):
            iterable = self._index_iterator_cls(indices)
        for idx in iterable:
            if idx not in self.skip_indices:
                yield idx

    def make_strategy(
        self,
        utility: UtilityBase,
        log_coefficient: SemivalueCoefficient | None,
    ) -> PowersetEvaluationStrategy:
        return PowersetEvaluationStrategy(utility, log_coefficient)

    @abstractmethod
    def generate(self, indices: IndexSetT) -> SampleGenerator:
        """Generates samples over the powerset of `indices`

        Each `PowersetSampler` defines its own way to generate the subsets by
        implementing this method. The outer loop is handled by the
        [index_iterable()][pydvl.valuation.samplers.powerset.PowersetSampler.index_iterable]
        method. Batching is handled by the
        [generate_batches()][pydvl.valuation.samplers.base.IndexSampler.generate_batches]
        method.

        Args:
            indices: The set from which to generate samples.
        Returns:
            A generator that yields samples over the powerset of `indices`.
        """
        ...

    def log_weight(self, n: int, subset_len: int) -> float:
        """Probability of sampling a set S as a function of total number of indices and
         set size.

        The uniform distribution over the powerset of a set with $n$ elements has mass
        $1/2^{n}$ over each subset.

        Args:
            n: The size of the index set. Note that the actual size of the set being
                sampled will often be n-1, as one index might be removed from the set.
                See [IndexIteration][pydvl.valuation.samplers.powerset.IndexIteration]
                for more.
            subset_len: The size of the subset being sampled

        Returns:
            The natural logarithm of the probability of sampling a set of the given
                size, when the index set has size `n`, under the
                [IndexIteration][pydvl.valuation.samplers.powerset.IndexIteration] given
                upon construction.

        """
        m = self.complement_size(n)
        return float(-m * np.log(2))

    @abstractmethod
    def sample_limit(self, indices: IndexSetT) -> int | None:
        """Returns the number of samples that can be generated from the index set.

        This will depend, among other things, on the type of
        [IndexIteration][pydvl.valuation.samplers.powerset.IndexIteration].

        Args:
            indices: The set of indices to sample from.
        Returns:
            The number of samples that can be generated from the index set.
        """
        ...


PowersetSamplerT = TypeVar("PowersetSamplerT", bound=PowersetSampler)


class PowersetEvaluationStrategy(
    Generic[PowersetSamplerT], EvaluationStrategy[PowersetSamplerT, ValueUpdate]
):
    """The standard strategy for evaluating the utility of subsets of a set.

    This strategy computes the marginal value of each subset of the complement of an
    index in the training set. The marginal value is the difference between the utility
    of the subset and the utility of the subset with the index added back in.

    It is the standard strategy for the direct implementation of semi-values, when
    sampling is done over the powerset of the complement of an index.
    """

    @suppress_warnings(categories=(RuntimeWarning,), flag="show_warnings")
    def process(
        self, batch: SampleBatch, is_interrupted: NullaryPredicate
    ) -> list[ValueUpdate]:
        updates = []
        for sample in batch:
            u_i = self.utility(sample.with_idx_in_subset())
            u = self.utility(sample)
            marginal = u_i - u
            sign = np.sign(marginal)
            log_marginal = -np.inf if marginal == 0 else np.log(marginal * sign)
            log_marginal += self.valuation_coefficient(
                self.n_indices, len(sample.subset)
            )
            updates.append(ValueUpdate(sample.idx, log_marginal, sign))
            if is_interrupted():
                break
        return updates


class LOOSampler(PowersetSampler):
    """Leave-One-Out sampler.

    In this special case of a powerset sampler, for every index $i$ in the set $S$, the
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
        if not self._index_iterator_cls.is_proper():
            raise ValueError("LOO samplers require a proper index iteration strategy")
        self._rng = np.random.default_rng(seed)

    def generate(self, indices: IndexSetT) -> SampleGenerator:
        for idx in self.index_iterable(indices):
            yield Sample(idx, complement(indices, [idx]))

    def log_weight(self, n: int, subset_len: int) -> float:
        """This sampler returns only sets of size n-1. There are n such sets, so the
        probability of drawing one is 1/n, or 0 if subset_len != n-1."""
        return float(-np.log(n) if subset_len == n - 1 else -np.inf)

    def make_strategy(
        self,
        utility: UtilityBase,
        log_coefficient: SemivalueCoefficient | None,
    ) -> PowersetEvaluationStrategy[LOOSampler]:
        return LOOEvaluationStrategy(utility, log_coefficient)

    def sample_limit(self, indices: IndexSetT) -> int | None:
        return self._index_iterator_cls.length(len(indices))


class LOOEvaluationStrategy(PowersetEvaluationStrategy[LOOSampler]):
    """Computes marginal values for LOO.

    Upon construction, the total utility is computed once. Then, the utility for every
    sample processed in [process()] is subtracted from it and returned as value update.

    Args:
        utility: The utility function to use.
        coefficient: The coefficient to use. If `None`, the correction of importance
            sampling is disabled.
    """

    def __init__(
        self,
        utility: UtilityBase,
        coefficient: SemivalueCoefficient | None,
    ):
        super().__init__(utility, coefficient)
        assert utility.training_data is not None
        self.total_utility = utility(Sample(None, utility.training_data.indices))

    @suppress_warnings(categories=(RuntimeWarning,), flag="show_warnings")
    def process(
        self, batch: SampleBatch, is_interrupted: NullaryPredicate
    ) -> list[ValueUpdate]:
        updates = []
        for sample in batch:
            assert sample.idx is not None
            marginal = self.total_utility - self.utility(sample)
            sign = np.sign(marginal)
            log_marginal = -np.inf if marginal == 0 else np.log(marginal * sign)
            log_marginal += self.valuation_coefficient(
                self.n_indices, len(sample.subset)
            )
            updates.append(ValueUpdate(sample.idx, log_marginal, sign))
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

    def generate(self, indices: IndexSetT) -> SampleGenerator:
        for idx in self.index_iterable(indices):
            for subset in powerset(complement(indices, [idx])):
                yield Sample(idx, np.asarray(subset, dtype=indices.dtype))

    def sample_limit(self, indices: IndexSetT) -> int | None:
        len_outer = self._index_iterator_cls.length(len(indices))
        if len(indices) == 0:  # Empty index set
            return 0
        if len_outer is None:  # Infinite index iteration
            return None

        return int(len_outer * 2 ** (self.complement_size(len(indices))))


class UniformSampler(StochasticSamplerMixin, PowersetSampler):
    """Draws random samples uniformly from the powerset of the index set.

    Iterating over every index $i$, either in sequence or at random depending on
    the value of `index_iteration`, one subset of the complement
    `indices - {i}` is sampled with equal probability $2^{n-1}$.

    Args:
        batch_size: The number of samples to generate per batch. Batches are processed
            together by each subprocess when working in parallel.
        index_iteration: the strategy to use for iterating over indices to update.
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

    def generate(self, indices: IndexSetT) -> SampleGenerator:
        for idx in self.index_iterable(indices):
            subset = random_subset(complement(indices, [idx]), seed=self._rng)
            yield Sample(idx, subset)


class AntitheticSampler(StochasticSamplerMixin, PowersetSampler):
    """A sampler that draws samples uniformly, followed by their complements.

    Works as [UniformSampler][pydvl.valuation.samplers.UniformSampler], but for every
    tuple $(i,S)$, it subsequently returns $(i,S^c)$, where $S^c$ is the
    complement of the set $S$ in the set of indices, excluding $i$.

    By symmetry, the probability of sampling a set $S$ is the same as the probability of
    sampling its complement $S^c$, so that $p(S)$ in `log_weight` is the same as in the
    [PowersetSampler][pydvl.valuation.samplers.powerset.PowersetSampler] class.
    """

    def generate(self, indices: IndexSetT) -> SampleGenerator:
        for idx in self.index_iterable(indices):
            _complement = complement(indices, [idx])
            subset = random_subset(_complement, seed=self._rng)
            yield Sample(idx, subset)
            yield Sample(idx, complement(_complement, subset))
