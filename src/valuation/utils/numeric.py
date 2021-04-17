import numpy as np

from functools import lru_cache
from itertools import chain, combinations
from typing import Generator, Iterator, Iterable, List, TypeVar
from sklearn.metrics import check_scoring
from valuation.utils.dataset import Dataset
from valuation.utils.types import Scorer, SupervisedModel
from valuation import _logger

T = TypeVar('T')


def vanishing_derivatives(x: np.ndarray, min_values: int, eps: float) -> int:
    """ Returns the number of rows whose empirical derivatives have converged
        to zero, up to a tolerance of eps.
    """
    last_values = x[:, -min_values - 1:]
    d = np.diff(last_values, axis=1)
    zeros = np.isclose(d, 0.0, atol=eps).sum(axis=1)
    return int(np.sum(zeros >= min_values / 2))


def powerset(it: Iterable[T]) -> Iterator[Iterable[T]]:
    """ Returns an iterator for the power set of the argument.

    Subsets are generated in sequence by growing size. See `random_powerset()`
    for random sampling.

    >>> powerset([1,2])
    () (1,) (2,) (1,2)
    """
    s = list(it)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


# FIXME: make usage of the cache optional for cases where it is not necessary
# TODO: benchmark this, make maxsize configurable?
@lru_cache(maxsize=4096)
def utility(model: SupervisedModel,
            data: Dataset,
            indices: Iterable[int],
            scoring: Scorer,
            catch_errors: bool = True) \
        -> float:
    """ Fits the model on a subset of the training data and scores it on the
    test data. Results are memoized to avoid duplicate computation. This is
    useful in particular when computing utilities of permutations of indices.

    :param model: Any supervised model
    :param data: a split Dataset
    :param indices: a subset of indices from data.x_train.index. The type must
      be hashable for the caching to work, e.g. wrap the argument with
      `frozenset` (rather than `tuple` since order should not matter)
    :param catch_errors: set to True to return np.nan if fit() fails. This hack
        helps when a step in a pipeline fails if there are too few data points
    :param scoring: Same as in sklearn's `cross_validate()`: a string, a scorer
        callable or None for the default `model.score()`. Greater values must
        be better. If they are not, a negated version can be used (see
        `make_scorer`)
    :return: 0 if no indices are passed, otherwise the value the scorer on the
        test data.
    """
    if not indices:
        return 0.0
    scorer = check_scoring(model, scoring)
    x = data.x_train[list(indices)]
    y = data.y_train[list(indices)]
    try:
        model.fit(x, y)
        return scorer(model, data.x_test, data.y_test)
    except Exception as e:
        if catch_errors:
            _logger.warning(str(e))
            return np.nan
        else:
            raise e


def lower_bound_hoeffding(delta: float, eps: float, score_range: float) -> int:
    """ Minimum number of samples required for MonteCarlo Shapley to obtain
    an (eps,delta) approximation.
    That is, with probability 1-delta, the estimate will be epsilon close to
     the true quantity, if at least so many monte carlo samples are taken.
    """
    return int(np.ceil(np.log(2 / delta) * score_range ** 2 / (2 * eps ** 2)))


def random_subset_indices(n: int, k: int = None) -> List[int]:
    """ Uniformly samples a subset of indices in the range [0,n).

    :param n: number of indices.
    :param k: size of the subset. None for all sizes (empty included)
    """
    if k is not None:
        raise NotImplementedError
    if n <= 0:
        return []

    # FIXME: the normalization is wrong
    # if k is None:
    #     k = np.random.choice(np.arange(n+1),
    #                          p=[np.math.comb(n, j)/2**n for j in range(n+1)])
    # return np.random.randint(n, size=k).tolist()
    from random import getrandbits
    r = getrandbits(n)
    indices = []
    for b in range(n):
        if r & 1:
            indices.append(b)
        r = r >> 1
    return indices


def random_powerset(s: np.ndarray, max_subsets: int = None, k: int = None) \
        -> Generator[np.ndarray, None, None]:
    """ Uniformly samples a subset from the power set of the argument, without
    pre-generating all subsets and in no order.

    See `powerset()` if you wish to deterministically generate all subsets.
    :param s:
    :param max_subsets: if set, stop the generator after this many steps.
    :param k: sample only subsets of size k. `None` for all subsets.
    """
    if not isinstance(s, np.ndarray):
        raise TypeError

    if k is not None:
        raise NotImplementedError
    n = len(s)
    total = 1
    if max_subsets is None:
        max_subsets = np.inf
    while total <= max_subsets:
        subset = random_subset_indices(n, k)
        yield s[subset]
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
