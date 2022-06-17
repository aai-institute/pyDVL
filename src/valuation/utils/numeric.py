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

    >>> from valuation.utils.numeric import powerset
    >>> list(powerset([1,2]))
    [(), (1,), (2,), (1, 2)]
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
    enable_cache: bool = False,
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

    def subset_probabilities(n: int) -> List[float]:
        def sub(sizes: List[int]) -> List[float]:
            return [np.math.comb(n, j) / 2**n for j in sizes]

        job = MapReduceJob.from_fun(sub, lambda r: reduce(operator.add, r, []))
        ret = map_reduce(job, list(range(n + 1)), num_jobs=num_jobs)
        return ret[0]

    if enable_cache:
        _subset_probabilities = memcached(client_config=client_config, threshold=0.5)(
            subset_probabilities
        )
    else:
        _subset_probabilities = subset_probabilities

    while total <= max_subsets:
        if dist == PowerSetDistribution.WEIGHTED:
            k = np.random.choice(np.arange(n + 1), p=_subset_probabilities(n))
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


def linear_regression_analytical_derivative_d_theta(
    A: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    Calculates the analytical derivative for batches of L with respect to vect(A). The loss function is the mse loss,
    precisely L(x, y) = 0.5 * (r(x, y)^T r(x, y)) / n of the residuals r(x, y) = A @ x + b - y. The first derivative worth the
    parameters \theta = \vect{A} and b are given in vectorized form by d_theta L(x, y) = np.kron(A @ x + b - y, x) / n
    and d_b L(x, y) = res(x, y) / n.

    :param A: A np.ndarray of shape [NxM].
    :param b: A np.ndarray of shape [N].
    :param x: A np.ndarray of shape [BxM],
    :param y: A np.nparray of shape [BxN].
    :returns. A np.ndarray of shape [Bx((N+1)*M)], where each row vector is [d_theta L(x, y), d_b L(x, y)]
    """
    n, m = list(A.shape)
    residuals = x @ A.T + b - y
    kron_product = np.expand_dims(residuals, axis=2) * np.expand_dims(x, axis=1)
    test_grads = np.reshape(kron_product, [-1, n * m])
    full_grads = np.concatenate((test_grads, residuals), axis=1)
    return full_grads / n


def linear_regression_analytical_derivative_d2_theta(
    A: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    Calculates the analytical derivative for batches of L with respect to vect(A). The loss function is the mse loss,
    precisely L(x, y) = 0.5 * (r(x, y)^T r(x, y)) / n of the residuals r(x, y) = A @ x + b - y. The second derivative worth
    the parameters \theta = \vect{A} and b are given in vectorized form by
    d^2_theta L(x, y) = np.kron(I, xx^T) / n,
    d^2_b L(x, y) = eye(n) / n and
    d_theta d_b L(x, y) = d_b d_theta L(x, y) = np.kron(I, x) / n.


    :param A: A np.ndarray of shape [NxM].
    :param b: A np.ndarray of shape [N].
    :param x: A np.ndarray of shape [BxM],
    :param y: A np.nparray of shape [BxN].
    :returns. A np.ndarray of shape [((N+1)*M)x((N+1)*M)], representing the Hessian. It gets averaged over all samples.
    """
    n, m = tuple(A.shape)
    d2_theta = np.einsum("ia,ib->iab", x, x)
    d2_theta = np.mean(d2_theta, axis=0)
    d2_theta = np.kron(np.eye(n), d2_theta)
    d2_b = np.eye(n)
    mean_x = np.mean(x, axis=0, keepdims=True)
    d_theta_d_b = np.kron(np.eye(n), mean_x)
    top_matrix = np.concatenate((d2_theta, d_theta_d_b.T), axis=1)
    bottom_matrix = np.concatenate((d_theta_d_b, d2_b), axis=1)
    full_matrix = np.concatenate((top_matrix, bottom_matrix), axis=0)
    return full_matrix / n


def linear_regression_analytical_derivative_d_x_d_theta(
    A: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    Calculates the analytical derivative for batches of L with respect to vect(A). The loss function is the mse loss,
    precisely L(x, y) = 0.5 * (r(x, y)^T r(x, y)) / n of the residuals r(x, y) = A @ x + b - y. The second derivative worth
    the parameters \theta = \vect{A} and b are given in vectorized form by
    d_x_d_theta L(x, y) =

    :param A: A np.ndarray of shape [NxM].
    :param b: A np.ndarray of shape [N].
    :param x: A np.ndarray of shape [BxM],
    :param y: A np.nparray of shape [BxN].
    :returns. A np.ndarray of shape [Bx((N+1)*M)xM], representing the derivative.
    """
    n, m = tuple(A.shape)
    residuals = x @ A.T + b - y
    b = len(x)
    outer_product_matrix = np.einsum("ab,ic->iacb", A, x)
    outer_product_matrix = np.reshape(outer_product_matrix, [b, m * n, m])
    tiled_identity = np.tile(np.expand_dims(np.eye(m), axis=0), [b, n, 1])
    outer_product_matrix += tiled_identity * np.expand_dims(
        np.repeat(residuals, m, axis=1), axis=2
    )
    b_part_derivative = np.tile(np.expand_dims(A, axis=0), [b, 1, 1])
    full_derivative = np.concatenate((outer_product_matrix, b_part_derivative), axis=1)
    return full_derivative / n


def upweighting_influences_linear_regression_analytical(
    A: np.ndarray,
    b: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    train_x: np.ndarray,
    train_y: np.ndarray,
):
    """
    Calculate the influences of the training set onto the validation set for a linear model Ax+b=y.

    :param A: A np.ndarray of shape [NxM]
    :param b: A np.ndarray of shape [N]
    :param test_x: A np.ndarray of shape [BxM]
    :param test_y: A np.ndarray of shpae [BxN]
    :param train_x: A np.ndarray of shape [CxM]
    :param test_y: A np.ndarray of shape [CxN]
    :returns: A np.ndarray of shape [BxC] with the influences of the training points on the test points.
    """
    test_grads_analytical = linear_regression_analytical_derivative_d_theta(
        A, b, test_x, test_y
    )
    train_grads_analytical = linear_regression_analytical_derivative_d_theta(
        A, b, train_x, train_y
    )
    hessian_analytical = linear_regression_analytical_derivative_d2_theta(
        A, b, train_x, train_y
    )
    s_test_analytical = np.linalg.solve(hessian_analytical, test_grads_analytical.T).T
    return -np.einsum("ia,ja->ij", s_test_analytical, train_grads_analytical)
