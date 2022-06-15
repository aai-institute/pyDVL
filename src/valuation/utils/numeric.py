import operator
from enum import Enum
from functools import reduce
from itertools import chain, combinations
from typing import (
    Callable,
    Collection,
    Generator,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
)

import numpy as np

from valuation.utils import logger, memcached
from valuation.utils.caching import ClientConfig
from valuation.utils.parallel import MapReduceJob, map_reduce

T = TypeVar("T")


def mcmc_is_linear_function(
    A: Callable[[np.ndarray], np.ndarray], v: np.ndarray, verify_samples: int = 1000
):
    """Assumes nothing. Stochastically checks for property sum_i a_i * f(v_i) == f(sum_i a_i v_i)."""

    dim = v.shape[1]
    weights = np.random.uniform(size=[verify_samples, 1, dim])
    sample_vectors = [np.random.uniform(size=[verify_samples, dim]) for _ in range(dim)]
    lin_sample_vectors = [A(v) for v in sample_vectors]
    x = (weights * np.stack(sample_vectors, axis=-1)).sum(-1)
    A_x = A(x)
    sum_A_v = (weights * np.stack(lin_sample_vectors, axis=-1)).sum(-1)
    diff_value = np.max(np.abs(sum_A_v - A_x), axis=1)
    return np.max(diff_value) <= 1e-10


def mcmc_is_linear_function_positive_definite(
    A: Callable[[np.ndarray], np.ndarray], v: np.ndarray, verify_samples: int = 1000
):
    """Assumes linear function. Stochastically checks for property v.T @ f(v) >= 0"""

    dim = v.shape[1]
    add_v = np.random.uniform(size=[verify_samples, dim])
    v = np.concatenate((v, add_v), axis=0)
    product = np.einsum("ia,ia->i", v, A(v))
    is_positive_definite = np.sum(product <= 1e-7) == 0
    return is_positive_definite


def vanishing_derivatives(x: np.ndarray, min_values: int, atol: float) -> int:
    """Returns the number of rows whose empirical derivatives have converged
    to zero, up to an absolute tolerance of atol.
    """
    last_values = x[:, -min_values - 1 :]
    d = np.diff(last_values, axis=1)
    zeros = np.isclose(d, 0.0, atol=atol).sum(axis=1)
    return int(np.sum(zeros >= min_values / 2))


def powerset(it: Sequence[T]) -> Iterator[Collection[T]]:
    """Returns an iterator for the power set of the argument.

    Subsets are generated in sequence by growing size. See `random_powerset()`
    for random sampling.

    >>> powerset([1,2])
    () (1,) (2,) (1,2)
    """
    s = list(it)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def lower_bound_hoeffding(delta: float, eps: float, score_range: float) -> int:
    """Lower bound on the number of samples required for MonteCarlo Shapley to
     obtain an (ε,δ)-approximation.

    That is: with probability 1-δ, the estimate will be ε-close to the true
    quantity, if at least n samples are taken.
    """
    return int(np.ceil(np.log(2 / delta) * score_range**2 / (2 * eps**2)))


class PowerSetDistribution(Enum):
    UNIFORM = "uniform"
    WEIGHTED = "weighted"


def random_powerset(
    s: np.ndarray,
    max_subsets: int = None,
    dist: PowerSetDistribution = PowerSetDistribution.WEIGHTED,
    num_jobs: int = 1,
    *,
    client_config: Optional[ClientConfig] = None
) -> Generator[np.ndarray, None, None]:
    """Uniformly samples a subset from the power set of the argument, without
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
    :param client_config: Memcached client configuration
    """
    if not isinstance(s, np.ndarray):
        raise TypeError

    n = len(s)
    total = 1
    if max_subsets is None:
        max_subsets = np.inf

    @memcached(client_config=client_config, threshold=0.5)
    def subset_probabilities(n: int) -> List[float]:
        def sub(sizes: List[int]) -> List[float]:
            # FIXME: is the normalization ok?
            return [np.math.comb(n, j) / 2**n for j in sizes]

        job = MapReduceJob.from_fun(sub, lambda r: reduce(operator.add, r, []))
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
    """Spearman correlation for integer, distinct ranks.
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

    return 1 - 6 * np.sum((x - y) ** 2) / (lx**3 - lx)


def random_matrix_with_condition_number(
    n: int, condition_number: float, positive_definite: bool = False
) -> np.ndarray:
    """
    https://gist.github.com/bstellato/23322fe5d87bb71da922fbc41d658079#file-random_mat_condition_number-py
    https://math.stackexchange.com/questions/1351616/condition-number-of-ata
    """

    if positive_definite:
        condition_number = np.sqrt(condition_number)

    log_condition_number = np.log(condition_number)
    exp_vec = np.linspace(
        -log_condition_number / 4.0, log_condition_number * (n + 1) / (4 * (n - 1)), n
    )
    s = np.exp(exp_vec)
    S = np.diag(s)
    U, _ = np.linalg.qr((np.random.rand(n, n) - 5.0) * 200)
    V, _ = np.linalg.qr((np.random.rand(n, n) - 5.0) * 200)
    P = U.dot(S).dot(V.T)
    return P if not positive_definite else P @ P.T  # cond(P @ P.T) = cond(P) ** 2


def linear_regression_analytical_grads(
    A: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    Calculates the analytical derivative of L with respect to vect(A). The loss function is the mean squared error,
    precisely L(x, y) = np.mean((A @ x - y) ** 2).
    """
    n = A.shape[0]
    residuals = x @ A.T - y
    grads = []

    for i in range(len(x)):
        grad = np.kron(residuals[i], x[i])
        grads.append(grad)

    test_grads = np.stack(grads, axis=0)
    return (2 / n) * test_grads


def linear_regression_analytical_hessian(
    A: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    Calculates the analytical hessian of L with respect to vect(A). The loss function is the mean squared error,
    precisely L(x, y) = np.mean((A @ x - y) ** 2).
    """
    n, m = tuple(A.shape)
    inner_hessians = (2 / n) * np.einsum("ia,ib->iab", x, x)
    inner_hessian = np.mean(inner_hessians, axis=0)
    return np.kron(np.eye(n), inner_hessian)
