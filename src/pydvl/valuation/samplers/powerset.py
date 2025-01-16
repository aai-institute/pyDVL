r"""
Powerset samplers.

TODO: explain the formulation and the different samplers.

## Stochastic samplers

...


## References

[^1]: <a name="mitchell_sampling_2022"></a>Mitchell, Rory, Joshua Cooper, Eibe
      Frank, and Geoffrey Holmes. [Sampling Permutations for Shapley Value
      Estimation](https://jmlr.org/papers/v23/21-0439.html). Journal of Machine
      Learning Research 23, no. 43 (2022): 1–46.
[^2]: <a name="watson_accelerated_2023"></a>Watson, Lauren, Zeno Kujawa, Rayna Andreeva,
      Hao-Tsung Yang, Tariq Elahi, and Rik Sarkar. [Accelerated Shapley Value
      Approximation for Data Evaluation](https://doi.org/10.48550/arXiv.2311.05346).
      arXiv, 9 November 2023.
[^3]: <a name="wu_variance_2023"></a>Wu, Mengmeng, Ruoxi Jia, Changle Lin, Wei Huang,
      and Xiangyu Chang. [Variance Reduced Shapley Value Estimation for Trustworthy Data
      Valuation](https://doi.org/10.1016/j.cor.2023.106305). Computers & Operations
      Research 159 (1 November 2023): 106305.
[^4]: <a name="maleki_bounding_2014"></a>Maleki, Sasan, Long Tran-Thanh, Greg Hines,
      Talal Rahwan, and Alex Rogers. [Bounding the Estimation Error of Sampling-Based
      Shapley Value Approximation](https://arxiv.org/abs/1306.4265). arXiv:1306.4265
      [Cs], 12 February 2014.

"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Generator, Iterable, Type

import numpy as np
from numpy.typing import NDArray

from pydvl.utils.numeric import powerset, random_subset, random_subset_of_size
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
    "OwenSampler",
    "AntitheticOwenSampler",
    "UniformStratifiedSampler",
    "VarianceReducedStratifiedSampler",
    "IndexIteration",
    "SequentialIndexIteration",
    "RandomIndexIteration",
    "NoIndexIteration",
]


logger = logging.getLogger(__name__)


class IndexIteration(ABC):
    def __init__(self, indices: NDArray[IndexT]):
        self._indices = indices

    @abstractmethod
    def __iter__(self) -> Generator[IndexT | None, None, None]: ...

    @staticmethod
    @abstractmethod
    def length(indices: IndexSetT) -> int | None: ...


class SequentialIndexIteration(IndexIteration):
    def __iter__(self) -> Generator[IndexT, None, None]:
        yield from self._indices

    @staticmethod
    def length(indices: IndexSetT) -> int:
        return len(indices)


class RandomIndexIteration(StochasticSamplerMixin, IndexIteration):
    def __init__(self, indices: NDArray[IndexT], seed: Seed):
        super().__init__(indices, seed=seed)

    def __iter__(self) -> Generator[IndexT, None, None]:
        if len(self._indices) == 0:
            return

        while True:
            yield self._rng.choice(self._indices, size=1).item()

    @staticmethod
    def length(indices: IndexSetT) -> int | None:
        if len(indices) == 0:
            out = 0
        else:
            out = None
        return out


class NoIndexIteration(IndexIteration):
    def __iter__(self) -> Generator[None, None, None]:
        yield None

    @staticmethod
    def length(indices: IndexSetT) -> int:
        return 0


def complement(include: IndexSetT, exclude: Iterable[IndexT]) -> NDArray[IndexT]:
    """Returns the complement of the set of indices excluding the given
    indices.

    Args:
        include: The set of indices to consider.
        exclude: The indices to exclude from the complement.

    Returns:
        The complement of the set of indices excluding the given indices.
    """
    _exclude = [i for i in exclude if i is not None]
    return np.setxor1d(include, _exclude).astype(np.int_)


class PowersetSampler(IndexSampler, ABC):
    """
    An abstract class for samplers which iterate over the powerset of the
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
            index_iteration: the order in which indices are iterated over
        """
        super().__init__(batch_size)
        self._index_iteration = index_iteration

    def index_iterator(
        self, indices: IndexSetT
    ) -> Generator[IndexT | None, None, None]:
        """Iterates over indices with the method specified at construction."""
        if issubclass(self._index_iteration, StochasticSamplerMixin):
            # To-Do: Need to do something more elegant here
            seed = self._rng.integers(0, 2**32, dtype=np.uint32).item()  # type: ignore
            yield from self._index_iteration(indices, seed)  # type: ignore
        else:
            yield from self._index_iteration(indices)

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

    @staticmethod
    def weight(n: int, subset_len: int) -> float:
        """Correction coming from Monte Carlo integration so that the mean of
        the marginals converges to the value: the uniform distribution over the
        powerset of a set with n-1 elements has mass 2^{n-1} over each subset."""
        return float(2 ** (n - 1)) if n > 0 else 1.0


class PowersetEvaluationStrategy(EvaluationStrategy[PowersetSampler, ValueUpdate]):
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


class LOOSampler(IndexSampler):
    """Leave-One-Out sampler.
    For every index $i$ in the set $S$, the sample $(i, S_{-i})$ is returned.

    !!! tip "New in version 0.10.0"
    """

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        for idx in indices:
            yield Sample(idx, complement(indices, [idx]))

    @staticmethod
    def weight(n: int, subset_len: int) -> float:
        return 1.0

    def make_strategy(
        self,
        utility: UtilityBase,
        coefficient: Callable[[int, int, float], float] | None = None,
    ) -> EvaluationStrategy:
        return LOOEvaluationStrategy(self, utility, coefficient)

    def sample_limit(self, indices: IndexSetT) -> int:
        return len(indices)


class LOOEvaluationStrategy(EvaluationStrategy[LOOSampler, ValueUpdate]):
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

    !!! Note
        Outer indices are iterated over sequentially

    ??? Example
        ``` pycon
        >>> sampler = DeterministicUniformSampler()
        >>> for idx, s in sampler.generate_batches(np.arange(2)):
        >>>    print(f"{idx} - {s}", end=", ")
        1 - [], 1 - [2], 2 - [], 2 - [1],
        ```
    """

    def __init__(
        self,
        index_iteration: Type[
            SequentialIndexIteration | NoIndexIteration
        ] = SequentialIndexIteration,
        batch_size: int = 1,
    ):
        super().__init__(index_iteration=index_iteration, batch_size=batch_size)

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        for idx in self.index_iterator(indices):
            for subset in powerset(  # type:ignore
                complement(indices, [idx] if idx is not None else [])
            ):
                yield Sample(idx, np.asarray(subset, dtype=indices.dtype))

    def sample_limit(self, indices: IndexSetT) -> int | None:
        len_outer = self._index_iteration.length(indices)
        # empty index set
        if len(indices) == 0:
            out = 0
        # infinite index iteration
        elif len_outer is None:
            out = None
        # NoIndexIteration, i.e. sampling from all indices and not just the complement
        # of the current outer index
        elif len_outer == 0:
            out = 2 ** len(indices)
        # SequentialIndexIteration or other finite index iteration
        else:
            out = len_outer * 2 ** (len(indices) - 1)
        return out


class UniformSampler(StochasticSamplerMixin, PowersetSampler):
    """An iterator to perform uniform random sampling of subsets.

    Iterating over every index $i$, either in sequence or at random depending on
    the value of ``index_iteration``, one subset of the complement
    ``indices - {i}`` is sampled with equal probability $2^{n-1}$. The
    iterator never ends.

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
        while True:
            for idx in self.index_iterator(indices):
                subset = random_subset(
                    complement(indices, [idx] if idx is not None else []),
                    seed=self._rng,
                )
                yield Sample(idx, subset)


class OwenSampler(StochasticSamplerMixin, PowersetSampler):
    """A sampler for Owen shapley values.

    For each index $i$ the Owen sampler loops over a deterministic grid of probabilities
    (containing `n_samples_outer` entries between 0 and 1) and then draws
    `n_samples_inner` subsets of the complement of the current index where each element
    is sampled with the given probability.

    The total number of samples drawn is therefore `n_samples_outer * n_samples_inner`.

    Args:
        n_samples_outer: The number of entries in the probability grid used for
            the outer loop in Owen sampling.
        n_samples_inner: The number of samples drawn for each probability. In the
            original paper this was fixed to 2 for all experiments which is why we
            give it a default value of 2.
        batch_size: The batch size of the sampler.
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
            batch_size=batch_size, index_iteration=SequentialIndexIteration, seed=seed
        )
        self._n_samples_inner = n_samples_inner
        self._n_samples_outer = n_samples_outer
        self._q_stop = 1.0

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        probabilities = np.linspace(
            start=0, stop=self._q_stop, num=self._n_samples_outer
        )
        for idx in self.index_iterator(indices):
            _complement = complement(indices, [idx] if idx is not None else [])
            for prob in probabilities:
                for _ in range(self._n_samples_inner):
                    subset = random_subset(_complement, q=prob, seed=self._rng)
                    yield Sample(idx, subset)

    @staticmethod
    def weight(n: int, subset_len: int) -> float:
        return 1.0

    def sample_limit(self, indices: IndexSetT) -> int:
        return len(indices) * self._n_samples_outer * self._n_samples_inner


class AntitheticOwenSampler(OwenSampler):
    """A sampler for antithetic Owen shapley values.

    For each index $i$, the antithetic Owen sampler loops over a deterministic grid of
    probabilities (containing `n_samples_outer` entries between 0 and 0.5) and then
    draws `n_samples_inner` subsets of the complement of the current index where each
    element is sampled with the given probability. For each sample obtained that way,
    a second sample is generated by taking the complement of the first sample.

    The total number of samples drawn is therefore
    `2 * n_samples_outer * n_samples_inner`.

    For the same number of total samples, the antithetic Owen sampler yields usually
    more precise estimates of shapley values than the regular Owen sampler.

    Args:
        n_samples_outer: The number of entries in the probability grid used for
            the outer loop in Owen sampling.
        n_samples_inner: The number of samples drawn for each probability. In the
            original paper this was fixed to 2 for all experiments which is why we
            give it a default value of 2.
        batch_size: The batch size of the sampler.
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
            n_samples_outer=n_samples_outer,
            n_samples_inner=n_samples_inner,
            batch_size=batch_size,
            seed=seed,
        )
        self._q_stop = 0.5

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        for sample in super()._generate(indices):
            idx, subset = sample
            _exclude = [idx] if idx is not None else []
            _exclude += subset.tolist()
            _antithetic_subset = complement(indices, _exclude)
            yield sample
            yield Sample(idx, _antithetic_subset)

    def sample_limit(self, indices: IndexSetT) -> int:
        return 2 * super().sample_limit(indices)


class AntitheticSampler(StochasticSamplerMixin, PowersetSampler):
    """An iterator to perform uniform random sampling of subsets, and their
    complements.

    Works as [UniformSampler][pydvl.valuation.samplers.UniformSampler], but for every
    tuple $(i,S)$, it subsequently returns $(i,S^c)$, where $S^c$ is the
    complement of the set $S$ in the set of indices, excluding $i$.
    """

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        while True:
            for idx in self.index_iterator(indices):
                _complement = complement(indices, [idx] if idx is not None else [])
                subset = random_subset(_complement, seed=self._rng)
                yield Sample(idx, subset)
                yield Sample(idx, complement(_complement, subset))

    # FIXME: is a uniform 2^{1-n} weight correct here too?


class UniformStratifiedSampler(StochasticSamplerMixin, PowersetSampler):
    """For every index, sample a set size, then a set of that size."""

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        while True:
            for idx in self.index_iterator(indices):
                k = int(self._rng.choice(np.arange(len(indices)), size=1).item())
                subset = random_subset_of_size(
                    complement(indices, [idx] if idx is not None else []),
                    size=k,
                    seed=self._rng,
                )
                yield Sample(idx, subset)

    # FIXME: is a uniform 2^{1-n} weight correct here too?


class TruncatedUniformStratifiedSampler(UniformStratifiedSampler):
    r"""A sampler which samples set sizes between two bounds.

    This sampler was suggested in (Watson et al. 2023)<sup><a
    href="#watson_accelerated_2023">1</a></sup> for $\delta$-Shapley

    !!! tip "New in version 0.10.0"
    """

    def __init__(
        self,
        *,
        lower_bound: int,
        upper_bound: int,
        index_iteration: Type[IndexIteration] = SequentialIndexIteration,
        seed: Seed | None = None,
    ):
        super().__init__(index_iteration=index_iteration, seed=seed)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        while True:
            for idx in self.index_iterator(indices):
                k = self._rng.integers(
                    low=self.lower_bound, high=self.upper_bound + 1, size=1
                ).item()
                subset = random_subset_of_size(
                    complement(indices, [idx] if idx is not None else []),
                    size=k,
                    seed=self._rng,
                )
                yield Sample(idx, subset)

    # FIXME: is a uniform 2^{1-n} weight correct here too?


class VarianceReducedStratifiedSampler(StochasticSamplerMixin, PowersetSampler):
    r"""VRDS sampler.

    This sampler was suggested in (Wu et al. 2023)<sup><a
    href="#wu_variance_2023">3</a></sup>, a generalization of the stratified
    sampler in (Maleki et al. 2014)<sup><a href="#maleki_bounding_2014">4</a></sup>

    Args:
        samples_per_setsize: A function which returns the number of samples to
            take for a given set size.
        index_iteration: the order in which indices are iterated over

    !!! tip "New in version 0.10.0"
    """

    def __init__(
        self,
        samples_per_setsize: Callable[[int], int],
        index_iteration: Type[IndexIteration] = SequentialIndexIteration,
    ):
        super().__init__(index_iteration=index_iteration)
        self.samples_per_setsize = samples_per_setsize
        # HACK: closure around the argument to avoid weight() being an instance method
        # FIXME: is this the correct weight anyway?
        self.weight = lambda n, subset_len: samples_per_setsize(subset_len)  # type: ignore

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        while True:
            for idx in self.index_iterator(indices):
                from_set = complement(indices, [idx] if idx is not None else [])
                for k in range(1, len(from_set)):
                    for _ in range(self.samples_per_setsize(k)):
                        subset = random_subset_of_size(from_set, size=k, seed=self._rng)
                        yield Sample(idx, subset)

    @staticmethod
    def weight(n: int, subset_len: int) -> float:
        raise NotImplementedError  # This should never happen
