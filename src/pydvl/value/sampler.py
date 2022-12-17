from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
    Generic,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np

from pydvl.utils import powerset, random_powerset

if TYPE_CHECKING:
    from numpy.typing import NDArray

T = TypeVar("T")


class Sampler(Generic[T]):
    """Samplers iterate over subsets.

    For each element in the whole set, the complementary set is considered, and
    at most ``max_subsets`` from its power set are generated.


    .. example::

    .. code::python
       for idx, s in DeterministicSampler([1,2], 4):
           print(s)

    will print the arrays

    ``[]``, ``[2]``, ``[]``, ``[1]``

    """

    class IndexIteration(Enum):
        Sequential = "sequential"
        Random = "random"

    def __init__(
        self,
        indices: "NDArray[T]",
        index_iteration: IndexIteration = IndexIteration.Sequential,
    ):
        """
        :param indices: The set of items (indices) to sample from.
        """
        self._indices = indices
        self._index_iteration = index_iteration

    def __iter__(self):
        raise NotImplementedError()

    def complement(self, exclude: Sequence[T], *args, **kwargs) -> "NDArray[T]":
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
    def __init__(self, indices: "NDArray[T]"):
        """
        :param indices: The set of items (indices) to sample from.
        """
        super().__init__(indices, Sampler.IndexIteration.Sequential)

    def __iter__(self) -> Generator[T, Any, None]:
        for idx in self.indices():
            for subset in powerset(self.complement(idx)):
                yield idx, subset


class UniformSampler(Sampler[T]):
    def __iter__(self) -> Generator[Tuple[T], Any, None]:
        while True:
            for idx in self.indices():
                for subset in random_powerset(self.complement([idx]), max_subsets=1):
                    yield idx, subset


class AntitheticSampler(Sampler[T]):
    def complement(
        self, exclude: Sequence[T], exclude_idx: Optional[T] = None, *args, **kwargs
    ) -> "NDArray[np.int_]":
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


class PermutationSampler(Sampler[T]):
    """
    .. warning::
       This sampler requires caching to be enabled or computation
       will be doubled wrt. a "direct" implementation of permutation MC
    """

    def __iter__(self) -> Generator[Tuple[T], Any, None]:
        while True:
            permutation = np.random.permutation(self._indices)
            for i, idx in enumerate(permutation):
                yield idx, permutation[:i]


class HierarchicalSampler(Sampler[T]):
    def __iter__(self) -> Generator[Tuple[T], Any, None]:
        while True:
            for idx in self.indices():
                k = np.random.choice(np.arange(len(self._indices)), size=1).item()
                for subset in random_powerset(  # FIXME: not implemented
                    self.complement([idx]), size=k, max_subsets=1
                ):
                    yield idx, subset


class OwenSampler(Sampler[T]):
    class Algorithm(Enum):
        Standard = "standard"
        Antithetic = "antithetic"

    def __init__(self, indices: "NDArray[T]", method: Algorithm, n_steps: int):
        super().__init__(indices)
        q_stop = {
            OwenSampler.Algorithm.Standard: 1.0,
            OwenSampler.Algorithm.Antithetic: 0.5,
        }
        self.q_steps = np.linspace(start=0, stop=q_stop[method], num=n_steps)

    def complement(self, exclude: Sequence[T], *args, **kwargs):
        return np.setxor1d(self._indices, exclude)

    def __iter__(self) -> Generator[Tuple[np.float, T, T], Any, None]:
        while True:
            for idx in self.indices():
                for j, q in enumerate(self.q_steps):
                    for subset in random_powerset(
                        self.complement([idx]), q=q, max_subsets=1
                    ):
                        yield q, idx, subset
