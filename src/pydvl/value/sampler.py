from enum import Enum
from typing import Any, Generator, Generic, Optional, Sequence, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray

from pydvl.utils import powerset, random_powerset

T = TypeVar("T")


class Sampler(Generic[T]):
    """Samplers iterate over subsets.

    For each element in the whole set, the complementary set is considered, and
    at most ``max_subsets`` from its power set are generated.


    :Example:

    .. code::python
       for idx, s in DeterministicSampler([1,2], 4):
           print(s)

    will print the arrays

    ``[]``, ``[2]``, ``[]``, ``[1]``

    In addition, samplers define a :meth:`weight` function to be used as a
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
        """
        self._indices = indices
        self._index_iteration = index_iteration
        self._n = len(indices)

    def __iter__(self):
        raise NotImplementedError()

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
        raise NotImplementedError()

    def complement(self, exclude: Sequence[T], *args, **kwargs) -> NDArray[T]:
        return np.setxor1d(self._indices, exclude)

    def indices(self) -> Generator[T, Any, None]:
        """
        FIXME: this is probably not very useful, but I couldn't decide
          which method is better
        """
        if self._index_iteration is Sampler.IndexIteration.Sequential:
            for idx in self._indices:
                yield idx
        elif self._index_iteration is Sampler.IndexIteration.Random:
            while True:
                yield np.random.choice(self._indices, size=1).item()


class DeterministicSampler(Sampler[T]):
    def __init__(self, indices: NDArray[T]):
        """Uniform deterministic sampling of subsets.

        For every index $i$, each subset of `indices - {i}` has equal
        probability $2^{n-1}$.

        :param indices: The set of items (indices) to sample from.
        """
        super().__init__(indices, Sampler.IndexIteration.Sequential)

    def __iter__(self) -> Generator[T, Any, None]:
        for idx in self.indices():
            for subset in powerset(self.complement(idx)):
                yield idx, subset

    def weight(self, subset: Sequence[T]) -> float:
        """Deterministic sampling should be used only for exact computations,
        where there is no need for a correcting factor in Monte Carlo sums."""
        return 1.0


class UniformSampler(Sampler[T]):
    def __iter__(self) -> Generator[Tuple[T], Any, None]:
        while True:
            for idx in self.indices():
                for subset in random_powerset(self.complement([idx]), max_subsets=1):
                    yield idx, subset

    def weight(self, subset: Sequence[T]) -> float:
        """Correction coming from Monte Carlo integration so that the mean of
        the marginals converges to the value: the uniform distribution over the
        powerset of a set with n-1 elements has mass 2^{n-1} over each subset.
        The factor 1 / n corresponds to the one in the Shapley definition."""
        return 2 ** (self._n - 1) / self._n


class AntitheticSampler(Sampler[T]):
    def complement(
        self, exclude: Sequence[T], exclude_idx: Optional[T] = None, *args, **kwargs
    ) -> NDArray[np.int_]:
        tmp = super().complement(exclude)
        if exclude_idx is None:
            return tmp
        return np.setxor1d(tmp, [exclude_idx])

    def __iter__(self) -> Generator[Tuple[T], Any, None]:
        while True:
            for idx in self.indices():
                for subset in random_powerset(self.complement([idx]), max_subsets=1):
                    yield idx, subset
                    yield idx, self.complement(subset, idx)

    def weight(self, subset: Sequence[T]) -> float:
        return 2 ** (self._n - 1) / self._n


class PermutationSampler(Sampler[T]):
    """Sample permutations of indices and iterate through each returning sets,
    as required for the permutation definition of Shapley value.

    .. warning::
       This sampler requires caching to be enabled or computation
       will be doubled wrt. a "direct" implementation of permutation MC

    """

    def __iter__(self) -> Generator[Tuple[T], Any, None]:
        while True:
            permutation = np.random.permutation(self._indices)
            for i, idx in enumerate(permutation):
                yield idx, permutation[:i]

    def weight(self, subset: Sequence[T]) -> float:
        return 1.0


class HierarchicalSampler(Sampler[T]):
    """Sample a set size, then a set of that size.

    .. todo::
       This is unnecessary, but a step towards proper stratified sampling.

    """

    def __iter__(self) -> Generator[Tuple[T], Any, None]:
        while True:
            for idx in self.indices():
                k = np.random.choice(np.arange(len(self._indices)), size=1).item()
                for subset in random_powerset(  # FIXME: not implemented
                    self.complement([idx]), size=k, max_subsets=1
                ):
                    yield idx, subset

    def weight(self, subset: Sequence[T]) -> float:
        return 2 ** (self._n - 1) / self._n


class OwenSampler(Sampler[T]):
    class Algorithm(Enum):
        Standard = "standard"
        Antithetic = "antithetic"

    def __init__(self, indices: NDArray[T], method: Algorithm, n_steps: int):
        super().__init__(indices)
        q_stop = {
            OwenSampler.Algorithm.Standard: 1.0,
            OwenSampler.Algorithm.Antithetic: 0.5,
        }
        self.q_steps = np.linspace(start=0, stop=q_stop[method], num=n_steps)

    def complement(self, exclude: Sequence[T], *args, **kwargs):
        return np.setxor1d(self._indices, exclude)

    def __iter__(self) -> Generator[Tuple[np.float_, T, T], Any, None]:
        while True:
            for idx in self.indices():
                for j, q in enumerate(self.q_steps):
                    for subset in random_powerset(
                        self.complement([idx]), q=q, max_subsets=1
                    ):
                        yield q, idx, subset

    def weight(self, subset: Sequence[T]) -> float:
        raise NotImplementedError("Compute the right weight")
