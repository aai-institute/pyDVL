r"""
Samplers iterate over subsets of indices.

The classes in this module are used to iterate over indices and subsets of their
complement in the whole set, as required for the computation of marginal utility
for semi-values. The elements returned when iterating over any subclass of
[PowersetSampler][pydvl.value.sampler.PowersetSampler] are tuples of the form
`(idx, subset)`, where `idx` is the index of the element being added to the
subset, and `subset` is the subset of the complement of `idx`.
The classes in this module are used to iterate over an index set $I$ as required
for the computation of marginal utility for semi-values. The elements returned
when iterating over any subclass of :class:`PowersetSampler` are tuples of the
form $(i, S)$, where $i$ is an index of interest, and $S \subset I \setminus \{i\}$
is a subset of the complement of $i$.

The iteration happens in two nested loops. An outer loop iterates over $I$, and
an inner loop iterates over the powerset of $I \setminus \{i\}$. The outer
iteration can be either sequential or at random.

!!! Note
    This is the natural mode of iteration for the combinatorial definition of
    semi-values, in particular Shapley value. For the computation using
    permutations, adhering to this interface is not ideal, but we stick to it for
    consistency.

The samplers are used in the [semivalues][pydvl.value.semivalues] module to
compute any semi-value, in particular Shapley and Beta values, and Banzhaf
indices.

## Slicing of samplers

The samplers can be sliced for parallel computation. For those which are
embarrassingly parallel, this is done by slicing the set of "outer" indices and
returning new samplers over those slices. This includes all truly powerset-based
samplers, such as [DeterministicUniformSampler][pydvl.value.sampler.DeterministicUniformSampler]
and [UniformSampler][pydvl.value.sampler.UniformSampler]. In contrast, slicing a
[PermutationSampler][pydvl.value.sampler.PermutationSampler] creates a new
sampler which iterates over the same indices.


## References

[^1]: <a name="mitchell_sampling_2022"></a>Mitchell, Rory, Joshua Cooper, Eibe
      Frank, and Geoffrey Holmes. [Sampling Permutations for Shapley Value
      Estimation](http://jmlr.org/papers/v23/21-0439.html). Journal of Machine
      Learning Research 23, no. 43 (2022): 1–46.
[^2]: <a name="wang_data_2023"></a>Wang, J.T. and Jia, R., 2023.
    [Data Banzhaf: A Robust Data Valuation Framework for Machine Learning](https://proceedings.mlr.press/v206/wang23e.html).
    In: Proceedings of The 26th International Conference on Artificial Intelligence and Statistics, pp. 6388-6421.

"""

from __future__ import annotations

import abc
import math
from enum import Enum
from itertools import permutations
from typing import (
    Generic,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import numpy as np
from numpy.typing import NDArray

from pydvl.utils.numeric import powerset, random_subset, random_subset_of_size
from pydvl.utils.types import IndexT, Seed

__all__ = [
    "AntitheticSampler",
    "DeterministicPermutationSampler",
    "DeterministicUniformSampler",
    "MSRSampler",
    "PermutationSampler",
    "PowersetSampler",
    "RandomHierarchicalSampler",
    "SampleT",
    "StochasticSampler",
    "StochasticSamplerMixin",
    "UniformSampler",
]

SampleT = Tuple[IndexT, NDArray[IndexT]]
Sequence.register(np.ndarray)


class PowersetSampler(abc.ABC, Iterable[SampleT], Generic[IndexT]):
    r"""Samplers are custom iterables over subsets of indices.

    Calling ``iter()`` on a sampler returns an iterator over tuples of the form
    $(i, S)$, where $i$ is an index of interest, and $S \subset I \setminus \{i\}$
    is a subset of the complement of $i$.

    This is done in two nested loops, where the outer loop iterates over the set
    of indices, and the inner loop iterates over subsets of the complement of
    the current index. The outer iteration can be either sequential or at random.

    !!! Note
        Samplers are **not** iterators themselves, so that each call to ``iter()``
        e.g. in a for loop creates a new iterator.

    ??? Example
        ``` pycon
        >>>for idx, s in DeterministicUniformSampler(np.arange(2)):
        >>>    print(s, end="")
        [][2,][][1,]
        ```

    # Methods required in subclasses

    Samplers must implement a [weight()][pydvl.value.sampler.PowersetSampler.weight]
    function to be used as a multiplier in Monte Carlo sums, so that the limit
    expectation coincides with the semi-value.

    # Slicing of samplers

    The samplers can be sliced for parallel computation. For those which are
    embarrassingly parallel, this is done by slicing the set of "outer" indices
    and returning new samplers over those slices.
    """

    class IndexIteration(Enum):
        Sequential = "sequential"
        Random = "random"

    def __init__(
        self,
        indices: NDArray[IndexT],
        index_iteration: IndexIteration = IndexIteration.Sequential,
        outer_indices: NDArray[IndexT] | None = None,
        **kwargs,
    ):
        """
        Args:
            indices: The set of items (indices) to sample from.
            index_iteration: the order in which indices are iterated over
            outer_indices: The set of items (indices) over which to iterate
                when sampling. Subsets are taken from the complement of each index
                in succession. For embarrassingly parallel computations, this set
                is sliced and the samplers are used to iterate over the slices.
        """
        self._indices = indices
        self._index_iteration = index_iteration
        self._outer_indices = outer_indices if outer_indices is not None else indices
        self._n = len(indices)
        self._n_samples = 0

    @property
    def indices(self) -> NDArray[IndexT]:
        return self._indices

    @indices.setter
    def indices(self, indices: NDArray[IndexT]):
        raise AttributeError("Cannot set indices of sampler")

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @n_samples.setter
    def n_samples(self, n: int):
        raise AttributeError("Cannot reset a sampler's number of samples")

    def complement(self, exclude: Sequence[IndexT]) -> NDArray[IndexT]:
        return np.setxor1d(self._indices, exclude)  # type: ignore

    def iterindices(self) -> Iterator[IndexT]:
        """Iterates over indices in the order specified at construction.

        FIXME: this is probably not very useful, but I couldn't decide
          which method is better
        """
        if self._index_iteration is PowersetSampler.IndexIteration.Sequential:
            for idx in self._outer_indices:
                yield idx
        elif self._index_iteration is PowersetSampler.IndexIteration.Random:
            while True:
                yield np.random.choice(self._outer_indices, size=1).item()

    @overload
    def __getitem__(self, key: slice) -> PowersetSampler[IndexT]: ...

    @overload
    def __getitem__(self, key: list[int]) -> PowersetSampler[IndexT]: ...

    def __getitem__(self, key: slice | list[int]) -> PowersetSampler[IndexT]:
        if isinstance(key, slice) or isinstance(key, Iterable):
            return self.__class__(
                self._indices,
                index_iteration=self._index_iteration,
                outer_indices=self._outer_indices[key],
            )
        raise TypeError("Indices must be an iterable or a slice")

    def __len__(self) -> int:
        """Returns the number of outer indices over which the sampler iterates."""
        return len(self._outer_indices)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return f"{self.__class__.__name__}({self._indices}, {self._outer_indices})"

    @abc.abstractmethod
    def __iter__(self) -> Iterator[SampleT]: ...

    @classmethod
    @abc.abstractmethod
    def weight(cls, n: int, subset_len: int) -> float:
        r"""Factor by which to multiply Monte Carlo samples, so that the
        mean converges to the desired expression.

        By the Law of Large Numbers, the sample mean of $\Delta_i(S_j)$
        converges to the expectation under the distribution from which $S_j$ is
        sampled.

        $$ \frac{1}{m}  \sum_{j = 1}^m \Delta_i (S_j) c (S_j) \longrightarrow
           \underset{S \sim \mathcal{D}_{- i}}{\mathbb{E}} [\Delta_i (S) c (
           S)]$$

        We add a factor $c(S_j)$ in order to have this expectation coincide with
        the desired expression.
        """
        ...


class StochasticSamplerMixin:
    """Mixin class for samplers which use a random number generator."""

    def __init__(self, *args, seed: Optional[Seed] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._rng = np.random.default_rng(seed)


class DeterministicUniformSampler(PowersetSampler[IndexT]):
    def __init__(self, indices: NDArray[IndexT], *args, **kwargs):
        """An iterator to perform uniform deterministic sampling of subsets.

        For every index $i$, each subset of the complement `indices - {i}` is
        returned.

        !!! Note
            Indices are always iterated over sequentially, irrespective of
            the value of `index_iteration` upon construction.

        ??? Example
            ``` pycon
            >>> for idx, s in DeterministicUniformSampler(np.arange(2)):
            >>>    print(f"{idx} - {s}", end=", ")
            1 - [], 1 - [2], 2 - [], 2 - [1],
            ```

        Args:
            indices: The set of items (indices) to sample from.
        """
        # Force sequential iteration
        kwargs.update({"index_iteration": PowersetSampler.IndexIteration.Sequential})
        super().__init__(indices, *args, **kwargs)

    def __iter__(self) -> Iterator[SampleT]:
        for idx in self.iterindices():
            # FIXME: type ignore just necessary for CI ??
            for subset in powerset(self.complement([idx])):  # type: ignore
                yield idx, np.array(subset)
                self._n_samples += 1

    @classmethod
    def weight(cls, n: int, subset_len: int) -> float:
        return float(2 ** (n - 1)) if n > 0 else 1.0


class UniformSampler(StochasticSamplerMixin, PowersetSampler[IndexT]):
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

    def __iter__(self) -> Iterator[SampleT]:
        while True:
            for idx in self.iterindices():
                subset = random_subset(self.complement([idx]), seed=self._rng)
                yield idx, subset
                self._n_samples += 1
            if self._n_samples == 0:  # Empty index set
                break

    @classmethod
    def weight(cls, n: int, subset_len: int) -> float:
        """Correction coming from Monte Carlo integration so that the mean of
        the marginals converges to the value: the uniform distribution over the
        powerset of a set with n-1 elements has mass 2^{n-1} over each subset."""
        return float(2 ** (n - 1)) if n > 0 else 1.0


class MSRSampler(StochasticSamplerMixin, PowersetSampler[IndexT]):
    """An iterator to perform sampling of random subsets.

    This sampler does not return any index, it only returns subsets of the data.
    This sampler is used in (Wang et. al.)<sup><a href="wang_data_2023">2</a></sup>.
    """

    def __iter__(self) -> Iterator[SampleT]:
        if len(self) == 0:
            return
        while True:
            subset = random_subset(self.indices, seed=self._rng)
            yield None, subset
            self._n_samples += 1

    @classmethod
    def weight(cls, n: int, subset_len: int) -> float:
        return 1.0


class AntitheticSampler(StochasticSamplerMixin, PowersetSampler[IndexT]):
    """An iterator to perform uniform random sampling of subsets, and their
    complements.

    Works as [UniformSampler][pydvl.value.sampler.UniformSampler], but for every
    tuple $(i,S)$, it subsequently returns $(i,S^c)$, where $S^c$ is the
    complement of the set $S$ in the set of indices, excluding $i$.
    """

    def __iter__(self) -> Iterator[SampleT]:
        while True:
            for idx in self.iterindices():
                _complement = self.complement([idx])
                subset = random_subset(_complement, seed=self._rng)
                yield idx, subset
                self._n_samples += 1
                yield idx, np.setxor1d(_complement, subset)
                self._n_samples += 1
            if self._n_samples == 0:  # Empty index set
                break

    @classmethod
    def weight(cls, n: int, subset_len: int) -> float:
        return float(2 ** (n - 1)) if n > 0 else 1.0


class PermutationSampler(StochasticSamplerMixin, PowersetSampler[IndexT]):
    """Sample permutations of indices and iterate through each returning
    increasing subsets, as required for the permutation definition of
    semi-values.

    This sampler does not implement the two loops described in
    [PowersetSampler][pydvl.value.sampler.PowersetSampler]. Instead, for a
    permutation `(3,1,4,2)`, it returns in sequence the tuples of index and
    sets:  `(3, {})`, `(1, {3})`, `(4, {3,1})` and `(2, {3,1,4})`.

    Note that the full index set is never returned.

    !!! Warning
        This sampler requires caching to be enabled or computation
        will be doubled wrt. a "direct" implementation of permutation MC
    """

    def __iter__(self) -> Iterator[SampleT]:
        while True:
            permutation = self._rng.permutation(self._indices)
            for i, idx in enumerate(permutation):
                yield idx, permutation[:i]
                self._n_samples += 1
            if self._n_samples == 0:  # Empty index set
                break

    def __getitem__(self, key: slice | list[int]) -> PowersetSampler[IndexT]:
        """Permutation samplers cannot be split across indices, so we return
        a copy of the full sampler."""
        return super().__getitem__(slice(None))

    @classmethod
    def weight(cls, n: int, subset_len: int) -> float:
        return n * math.comb(n - 1, subset_len) if n > 0 else 1.0


class AntitheticPermutationSampler(PermutationSampler[IndexT]):
    """Samples permutations like
    [PermutationSampler][pydvl.value.sampler.PermutationSampler], but after
    each permutation, it returns the same permutation in reverse order.

    This sampler was suggested in (Mitchell et al. 2022)<sup><a
    href="#mitchell_sampling_2022">1</a></sup>

    !!! tip "New in version 0.7.1"
    """

    def __iter__(self) -> Iterator[SampleT]:
        while True:
            permutation = self._rng.permutation(self._indices)
            for perm in permutation, permutation[::-1]:
                for i, idx in enumerate(perm):
                    yield idx, perm[:i]
                    self._n_samples += 1

            if self._n_samples == 0:  # Empty index set
                break


class DeterministicPermutationSampler(PermutationSampler[IndexT]):
    """Samples all n! permutations of the indices deterministically, and
    iterates through them, returning sets as required for the permutation-based
    definition of semi-values.

    !!! Warning
        This sampler requires caching to be enabled or computation
        will be doubled wrt. a "direct" implementation of permutation MC

    !!! Warning
        This sampler is not parallelizable, as it always iterates over the whole
        set of permutations in the same order. Different processes would always
        return the same values for all indices.
    """

    def __iter__(self) -> Iterator[SampleT]:
        for permutation in permutations(self._indices):
            for i, idx in enumerate(permutation):
                yield idx, np.array(permutation[:i], dtype=self._indices.dtype)
                self._n_samples += 1


class RandomHierarchicalSampler(StochasticSamplerMixin, PowersetSampler[IndexT]):
    """For every index, sample a set size, then a set of that size.

    !!! Todo
        This is unnecessary, but a step towards proper stratified sampling.
    """

    def __iter__(self) -> Iterator[SampleT]:
        while True:
            for idx in self.iterindices():
                k = self._rng.choice(np.arange(len(self._indices)), size=1).item()
                subset = random_subset_of_size(
                    self.complement([idx]), size=k, seed=self._rng
                )
                yield idx, subset
                self._n_samples += 1
            if self._n_samples == 0:  # Empty index set
                break

    @classmethod
    def weight(cls, n: int, subset_len: int) -> float:
        return float(2 ** (n - 1)) if n > 0 else 1.0


# TODO Replace by Intersection[StochasticSamplerMixin, PowersetSampler[T]]
# See https://github.com/python/typing/issues/213
StochasticSampler = Union[
    UniformSampler[IndexT],
    PermutationSampler[IndexT],
    AntitheticSampler[IndexT],
    RandomHierarchicalSampler[IndexT],
    MSRSampler[IndexT],
]
