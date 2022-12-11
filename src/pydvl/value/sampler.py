from enum import Enum
from typing import (
    Any,
    Generator,
    Generic,
    Optional,
    Sequence,
    TYPE_CHECKING,
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

    For each index, the complement set indices is considered, and at most
    ``max_subsets`` from the power set are generated.


    .. example::

    .. code::python
       for idx, s in DeterministicSampler([1,2], 4):
           print(s)

    will print the arrays

    ``[]``, ``[2]``, ``[]``, ``[1]``

    """

    def __init__(self, indices: "NDArray[T]", max_subsets: Optional[int] = None):
        self._indices = indices
        self._max_subsets = max_subsets

    def __iter__(self):
        raise NotImplementedError()

    def complement(self, exclude: Sequence[T], *args, **kwargs) -> Sequence[T]:
        return np.setxor1d(self._indices, exclude)


class DeterministicSampler(Sampler[T]):
    def __iter__(self) -> Generator[T, Any, None]:
        for idx in self._indices:
            for subset in powerset(self.complement(idx)):
                yield idx, subset


class UniformSampler(Sampler[T]):
    def __iter__(self) -> Generator[Tuple[T], Any, None]:
        for idx in self._indices:
            for subset in random_powerset(
                self.complement([idx]), max_subsets=self._max_subsets
            ):
                yield idx, subset


class AntitheticSampler(Sampler[T]):
    # FIXME: overload with different args
    def complement(
        self, exclude: Sequence[T], exclude_idx: Optional[T] = None, *args, **kwargs
    ) -> "NDArray[np.int_]":
        tmp = super().complement(exclude)
        if exclude_idx is None:
            return tmp
        return np.setxor1d(tmp, [exclude_idx])

    def __iter__(self) -> Generator[Tuple[T], Any, None]:
        for idx in self._indices:
            for subset in random_powerset(
                self.complement([idx]), max_subsets=self._max_subsets
            ):
                yield idx, subset
                yield idx, self.complement(subset, idx)


class PermutationSampler(Sampler[T]):
    """
    .. warning::
       This sampler requires caching to be enabled or computation
       will be doubled wrt. a "direct" implementation of permutation MC
    """

    def __iter__(self) -> Generator[Tuple[T], Any, None]:
        permutation = np.random.permutation(self._indices)
        for i, idx in enumerate(permutation):
            yield idx, permutation[:i]


class HierarchicalSampler(Sampler[T]):
    def __iter__(self) -> Generator[Tuple[T], Any, None]:
        for idx in self._indices:
            for k in range(len(self._indices)):
                for subset in random_powerset(
                    self.complement([idx]),
                    size=k,  # FIXME: not implemented
                    max_subsets=1 + self._max_subsets // k,
                ):
                    yield idx, subset


class OwenSampler(Generic[T]):
    class Algorithm(Enum):
        Standard = "standard"
        Antithetic = "antithetic"

    def __init__(
        self,
        indices: "NDArray[T]",
        method: Algorithm,
        n_steps: int,
        max_subsets: Optional[int] = None,
    ):
        q_stop = {
            OwenSampler.Algorithm.Standard: 1.0,
            OwenSampler.Algorithm.Antithetic: 0.5,
        }
        self.q_steps = np.linspace(start=0, stop=q_stop[method], num=n_steps)
        self._indices = indices
        self._max_subsets = max_subsets

    def complement(self, exclude: Sequence[T]):
        return np.setxor1d(self._indices, exclude)

    def __iter__(self) -> Generator[Tuple[np.float, T, T], Any, None]:
        for idx in self._indices:
            for j, q in enumerate(self.q_steps):
                for subset in random_powerset(
                    self.complement([idx]), q=q, max_subsets=self._max_subsets
                ):
                    yield q, idx, subset
