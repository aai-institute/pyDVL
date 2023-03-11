from __future__ import annotations

import abc
import math
from collections.abc import Generator, Iterable, Sequence
from enum import Enum
from typing import Any, TypeVar, overload

import numpy as np
from numpy.typing import NDArray

from pydvl.utils.numeric import powerset, random_subset, random_subset_of_size

T = TypeVar("T", bound=np.generic)


class PowersetSampler(abc.ABC, Iterable[T]):
    """Samplers iterate over subsets of indices.

    For each element in the set, the complementary set is considered and
    sets from its power set are generated.

    :Example:

    .. code::python
       for idx, s in DeterministicSampler([1,2], 4):
           print(s)

    will print the arrays

    ``[]``, ``[2]``, ``[]``, ``[1]``

    In addition, samplers must define a :meth:`weight` function to be used as a
    multiplier in Monte Carlo sums, so that the limit expectation coincides with
    the semi-value.
    """

    class IndexIteration(Enum):
        Sequential = "sequential"
        Random = "random"

    def __init__(
        self,
        indices: NDArray[T],
        index_iteration: IndexIteration = IndexIteration.Sequential,
    ):
        """
        :param indices: The set of items (indices) to sample from.
        :param index_iteration: the order in which indices are iterated over
        """
        self._indices = indices
        self._index_iteration = index_iteration
        self._n = len(indices)
        self._n_samples = 0

    @property
    def indices(self) -> NDArray[T]:
        return self._indices

    @indices.setter
    def indices(self, indices: NDArray[T]):
        raise AttributeError("Cannot set indices of sampler")

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @n_samples.setter
    def n_samples(self, n: int):
        raise AttributeError("Cannot reset a sampler's number of samples")

    @abc.abstractmethod
    def __iter__(self):
        ...

    @abc.abstractmethod
    def weight(self, subset: Sequence[T]) -> float:
        r"""Factor by which to multiply Monte Carlo samples, so that the
        mean converges to the desired expression.

        By the Law of Large Numbers, the sample mean of $\delta_i(S_j)$ converges
        to the expectation under the distribution from which $S_j$ is sampled.

        $$ \frac{1}{m}  \sum_{j = 1}^m \delta_i (S_j) c (S_j) \longrightarrow
           \underset{S \sim \mathcal{D}_{- i}}{\mathbb{E}} [\delta_i (S) c (S)]$$

        We add a factor $c(S_j)$ in order to have this expectation coincide with
        the desired expression.
        """
        ...

    def complement(self, exclude: Sequence[T]) -> NDArray[T]:
        return np.setxor1d(self._indices, exclude)

    def iterindices(self) -> Generator[T, Any, None]:
        """Iterates over indices in the order specified at construction.

        FIXME: this is probably not very useful, but I couldn't decide
          which method is better
        """
        if self._index_iteration is PowersetSampler.IndexIteration.Sequential:
            for idx in self._indices:
                yield idx
        elif self._index_iteration is PowersetSampler.IndexIteration.Random:
            while True:
                yield np.random.choice(self._indices, size=1).item()

    @overload
    def __getitem__(self, key: slice) -> "PowersetSampler[T]":
        ...

    @overload
    def __getitem__(self, key: list[int]) -> "PowersetSampler[T]":
        ...

    def __getitem__(self, key: slice | list[int]) -> "PowersetSampler[T]":
        if isinstance(key, slice) or isinstance(key, Iterable):
            return self.__class__(self._indices[key])
        raise TypeError("Indices must be an iterable or a slice")

    def __len__(self) -> int:
        return self._n


class DeterministicSampler(PowersetSampler[T]):
    def __init__(self, indices: NDArray[T]):
        """Uniform deterministic sampling of subsets.

        For every index $i$, each subset of `indices - {i}` has equal
        probability $2^{n-1}$.

        :param indices: The set of items (indices) to sample from.
        """
        super().__init__(indices, PowersetSampler.IndexIteration.Sequential)

    def __iter__(self) -> Generator[tuple[T, NDArray[T]], Any, None]:
        for idx in self.iterindices():
            for subset in powerset(self.complement([idx])):
                yield idx, subset
                self._n_samples += 1

    def weight(self, subset: Sequence[T]) -> float:
        """Deterministic sampling should be used only for exact computations,
        where there is no need for a correcting factor in Monte Carlo sums."""
        return 1.0


class UniformSampler(PowersetSampler[T]):
    def __iter__(self) -> Generator[tuple[T, NDArray[T]], Any, None]:
        while True:
            for idx in self.iterindices():
                subset = random_subset(self.complement([idx]))
                yield idx, subset
                self._n_samples += 1
            if self._n_samples == 0:  # Empty index set
                break

    def weight(self, subset: Sequence[T]) -> float:
        """Correction coming from Monte Carlo integration so that the mean of
        the marginals converges to the value: the uniform distribution over the
        powerset of a set with n-1 elements has mass 2^{n-1} over each subset.
        The factor 1 / n corresponds to the one in the Shapley definition."""
        return float(2 ** (self._n - 1)) if self._n > 0 else 1.0


class AntitheticSampler(PowersetSampler[T]):
    def __iter__(self) -> Generator[tuple[T, NDArray[T]], None, None]:
        while True:
            for idx in self.iterindices():
                subset = random_subset(self.complement([idx]))
                yield idx, subset
                self._n_samples += 1
                yield idx, self.complement(np.concatenate((subset, [idx])))
                self._n_samples += 1
            if self._n_samples == 0:  # Empty index set
                break

    def weight(self, subset: Sequence[T]) -> float:
        return float(2 ** (self._n - 1)) if self._n > 0 else 1.0


class PermutationSampler(PowersetSampler[T]):
    """Sample permutations of indices and iterate through each returning sets,
    as required for the permutation definition of semi-values.

    .. warning::
       This sampler requires caching to be enabled or computation
       will be doubled wrt. a "direct" implementation of permutation MC
    """

    def __iter__(self) -> Generator[tuple[T, NDArray[T]], None, None]:
        while True:
            permutation = np.random.permutation(self._indices)
            for i, idx in enumerate(permutation):
                yield idx, permutation[:i]
                self._n_samples += 1
            if self._n_samples == 0:  # Empty index set
                break

    def __getitem__(self, key: slice | list[int]) -> "PowersetSampler[T]":
        """Permutation samplers cannot be split across indices, so we return
        a copy of the full sampler."""
        return super().__getitem__(slice(None))

    def weight(self, subset: Sequence[T]) -> float:
        return self._n * math.comb(self._n - 1, len(subset)) if self._n > 0 else 1.0


class RandomHierarchicalSampler(PowersetSampler[T]):
    """For every index, sample a set size, then a set of that size.

    .. todo::
       This is unnecessary, but a step towards proper stratified sampling.
    """

    def __iter__(self) -> Generator[tuple[T, NDArray[T]], None, None]:
        while True:
            for idx in self.iterindices():
                k = np.random.choice(np.arange(len(self._indices)), size=1).item()
                subset = random_subset_of_size(self.complement([idx]), size=k)
                yield idx, subset
                self._n_samples += 1
            if self._n_samples == 0:  # Empty index set
                break

    def weight(self, subset: Sequence[T]) -> float:
        return 2 ** (self._n - 1) if self._n > 0 else 1.0
