import operator
import numpy as np


from enum import Enum
from functools import reduce
from itertools import chain, combinations
from typing import Collection, Generator, Iterator, List, Sequence, TypeVar
from valuation.utils import memcached
from valuation.utils.parallel import MapReduceJob, map_reduce

T = TypeVar('T')


def vanishing_derivatives(x: np.ndarray, min_values: int, atol: float) -> int:
    """ Returns the number of rows whose empirical derivatives have converged
        to zero, up to an absolute tolerance of atol.
    """
    last_values = x[:, -min_values - 1:]
    d = np.diff(last_values, axis=1)
    zeros = np.isclose(d, 0.0, atol=atol).sum(axis=1)
    return int(np.sum(zeros >= min_values / 2))


def powerset(it: Sequence[T]) -> Iterator[Collection[T]]:
    """ Returns an iterator for the power set of the argument.

    Subsets are generated in sequence by growing size. See `random_powerset()`
    for random sampling.

    >>> powerset([1,2])
    () (1,) (2,) (1,2)
    """
    s = list(it)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def lower_bound_hoeffding(delta: float, eps: float, score_range: float) -> int:
    """ Minimum number of samples n required for MonteCarlo Shapley to obtain
    an (ε,δ)-approximation.

    That is: with probability 1-δ, the estimate will be ε-close to the true
    quantity, if at least n samples are taken.
    """
    return int(np.ceil(np.log(2 / delta) * score_range ** 2 / (2 * eps ** 2)))


class PowerSetDistribution(Enum):
    UNIFORM = 'uniform'
    WEIGHTED = 'weighted'


def random_powerset(s: np.ndarray,
                    max_subsets: int = None,
                    dist: PowerSetDistribution = PowerSetDistribution.WEIGHTED,
                    num_jobs: int = 1) \
        -> Generator[np.ndarray, None, None]:
    """ Uniformly samples a subset from the power set of the argument, without
    pre-generating all subsets and in no order.

    This function accepts arbitrarily large values for n. However, values
    in the tens of thousands can take very long to compute, hence the ability
    to run in parallel with num_jobs.

    See `powerset()` if you wish to deterministically generate all subsets.

    :param s: set to sample from
    :param max_subsets: if set, stop the generator after this many steps.
    :param dist: whether to sample from the "true" distribution, i.e. weighted
        by the number of sets of size k, or "uniformly", taking e.g. the empty
        set to be as likely as any other
    :param num_jobs: Duh. Must be >= 1
    """
    if not isinstance(s, np.ndarray):
        raise TypeError

    n = len(s)
    total = 1
    if max_subsets is None:
        max_subsets = np.inf

    @memcached(threshold=0.5)
    def subset_probabilities(n: int) -> List[float]:
        def sub(sizes: List[int]) -> List[float]:
            # FIXME: is the normalization ok?
            return [np.math.comb(n, j) / 2 ** n for j in sizes]

        job = MapReduceJob.from_fun(sub,
                                    lambda r: reduce(operator.add, r, []))
        ret = map_reduce(job, list(range(n + 1)), num_jobs=num_jobs)
        return ret[0]

    while total <= max_subsets:
        if dist == PowerSetDistribution.WEIGHTED:
            k = np.random.choice(np.arange(n + 1), p=subset_probabilities(n))
        else:
            k = np.random.choice(np.arange(n + 1))
        subset = np.random.choice(s, replace=False, size=k)
        yield subset
        total += 1


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """ Spearman correlation for integer, distinct ranks.
    :return: A float in [-1,1]: -1 for reversed ranks, 1 for perfect match, 0
        for independent ranks
    """
    lx = len(x)
    ly = len(y)
    if lx == 0 or ly == 0 or lx != ly:
        raise ValueError("Ranks must be non empty and same length")
    if len(np.unique(x)) != lx or len(np.unique(y)) != ly:
        raise ValueError("Ranks must be unique")
    for a in x, y:
        if min(a) < 0 or min(a) > 1 or max(a) - min(a) > len(a):
            raise ValueError("Ranks must be in range [0,n-1] or [1,n]")
    try:
        if x.dtype != int or y.dtype != int:
            raise ValueError("Ranks must be integers")
    except AttributeError:
        raise TypeError("Input must be numpy.ndarray")

    return 1 - 6*np.sum((x - y)**2)/(lx**3 - lx)
