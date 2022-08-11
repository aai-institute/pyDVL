"""
Contains

- shapley related stuff.
- analytical derivatives for MSE and Linear Regression.
- methods for sampling datasets.
- code for calculating decision boundary in BinaryLogisticRegression.
"""


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
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

from valuation.utils import memcached
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
        _subset_probabilities = memcached(
            client_config=client_config, cache_threshold=0.5
        )(subset_probabilities)
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
    linear_model: Tuple[np.ndarray, np.ndarray], x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    :param linear_model: A tuple of np.ndarray' of shape [NxM] and [N] representing A and b respectively.
    :param x: A np.ndarray of shape [BxM].
    :param y: A np.nparray of shape [BxN].
    :returns: A np.ndarray of shape [Bx((N+1)*M)], where each row vector is [d_theta L(x, y), d_b L(x, y)]
    """

    A, b = linear_model
    n, m = list(A.shape)
    residuals = x @ A.T + b - y
    kron_product = np.expand_dims(residuals, axis=2) * np.expand_dims(x, axis=1)
    test_grads = np.reshape(kron_product, [-1, n * m])
    full_grads = np.concatenate((test_grads, residuals), axis=1)
    return full_grads / n


def linear_regression_analytical_derivative_d2_theta(
    linear_model: Tuple[np.ndarray, np.ndarray], x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    :param linear_model: A tuple of np.ndarray' of shape [NxM] and [N] representing A and b respectively.
    :param x: A np.ndarray of shape [BxM],
    :param y: A np.nparray of shape [BxN].
    :returns: A np.ndarray of shape [((N+1)*M)x((N+1)*M)], representing the Hessian. It gets averaged over all samples.
    """
    A, b = linear_model
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
    linear_model: Tuple[np.ndarray, np.ndarray], x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    :param linear_model: A tuple of np.ndarray of shape [NxM] and [N] representing A and b respectively.
    :param x: A np.ndarray of shape [BxM].
    :param y: A np.nparray of shape [BxN].
    :returns: A np.ndarray of shape [Bx((N+1)*M)xM], representing the derivative.
    """

    A, b = linear_model
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


def sample_classification_dataset_using_gaussians(
    mus: np.ndarray, sigma: float, num_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample from a uniform Gaussian mixture model.
    :param mus: 2d-matrix [CxD] with the means of the components in the rows.
    :param sigma: Standard deviation of each dimension of each component.
    :param num_samples: The number of samples to generate.
    :returns: A tuple of matrix x of shape [NxD] and target vector y of shape [N].
    """
    num_features = mus.shape[1]
    num_classes = mus.shape[0]
    gaussian_cov = sigma * np.eye(num_features)
    gaussian_chol = np.linalg.cholesky(gaussian_cov)
    y = np.random.randint(num_classes, size=num_samples)
    x = (
        np.einsum(
            "ij,kj->ki",
            gaussian_chol,
            np.random.normal(size=[num_samples, num_features]),
        )
        + mus[y]
    )
    return x, y


def decision_boundary_fixed_variance_2d(
    mu_1: np.ndarray, mu_2: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Closed-form solution for decision boundary dot(a, b) + b = 0 with fixed variance.
    :param mu_1: First mean.
    :param mu_2: Second mean.
    :returns: A callable which converts a continuous line (-infty, infty) to the decision boundary in feature space.
    """
    a = np.asarray([[0, 1], [-1, 0]]) @ (mu_2 - mu_1)
    b = (mu_1 + mu_2) / 2
    a = a.reshape([1, -1])
    return lambda z: z.reshape([-1, 1]) * a + b


def min_distance_points_to_line_2d(
    p: np.ndarray, a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """
    Closed-form solution for minimum distance of point to line specified by dot(a, x) + b = 0.
    :param p: A 2-dimensional matrix [NxD] representing the points.
    :param a: A 1-dimensional vector [D] representing the slope.
    :param b: The offset of the line.
    :returns: A 1-dimensional vector [N] with the shortest distance for each point to the line.
    """
    a = np.reshape(a, [2, 1])
    r = np.abs(p @ a + b) / np.sqrt(np.sum(a**2))
    return r[:, 0]


def get_running_avg_variance(
    previous_avg: Union[float, np.ndarray],
    previous_variance: Union[float, np.ndarray],
    new_value: Union[float, np.ndarray],
    count: int,
):
    """The method uses Welford's algorithm to calculate the running average and variance of
    a set of numbers.

    :param previous_avg: average value at previous step
    :param previous_variance: variance at previous step
    :param new_value: new value in the series of numbers
    :param count: number of points seen so far
    :return: new_average, new_variance, calculated with the new number
    """
    new_average = (new_value + count * previous_avg) / (count + 1)
    new_variance = previous_variance + (
        (new_value - previous_avg) * (new_value - new_average) - previous_variance
    ) / (count + 1)
    return new_average, new_variance
