"""
Contains

- shapley related stuff.
- analytical derivatives for MSE and Linear Regression.
- methods for sampling datasets.
- code for calculating decision boundary in BinaryLogisticRegression.
"""


import math
from enum import Enum
from itertools import chain, combinations
from typing import (
    TYPE_CHECKING,
    Collection,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
)

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FloatOrArray = TypeVar("FloatOrArray", float, "NDArray")

__all__ = [
    "powerset",
    "random_powerset",
    "linear_regression_analytical_derivative_d2_theta",
    "linear_regression_analytical_derivative_d_theta",
    "linear_regression_analytical_derivative_d_x_d_theta",
    "top_k_value_accuracy",
]

T = TypeVar("T")


def powerset(it: "NDArray") -> Iterator[Collection[T]]:
    """Returns an iterator for the power set of the argument.

    Subsets are generated in sequence by growing size. See `random_powerset()`
    for random sampling.

    >>> from pydvl.utils.numeric import powerset
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
    s: "NDArray",
    max_subsets: Optional[int] = None,
    dist: PowerSetDistribution = PowerSetDistribution.WEIGHTED,
) -> Generator["NDArray", None, None]:
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
    """
    if not isinstance(s, np.ndarray):
        raise TypeError

    n = len(s)
    total = 1
    if max_subsets is None:
        max_subsets = np.iinfo(np.int32).max

    def subset_probabilities(n: int) -> List[float]:
        return [math.comb(n, j) / 2**n for j in range(n + 1)]

    _subset_probabilities = subset_probabilities

    while total <= max_subsets:
        if dist == PowerSetDistribution.WEIGHTED:
            k = np.random.choice(np.arange(n + 1), p=_subset_probabilities(n))
        else:
            k = np.random.choice(np.arange(n + 1))
        subset = np.random.choice(s, replace=False, size=k)
        yield subset
        total += 1


def spearman(x: "NDArray", y: "NDArray") -> float:
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

    return 1 - 6 * float(np.sum((x - y) ** 2)) / (lx**3 - lx)


def random_matrix_with_condition_number(
    n: int, condition_number: float, positive_definite: bool = False
) -> "NDArray":
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
    # cond(P @ P.T) = cond(P) ** 2
    return P if not positive_definite else P @ P.T  # type: ignore


def linear_regression_analytical_derivative_d_theta(
    linear_model: Tuple["NDArray", "NDArray"], x: "NDArray", y: "NDArray"
) -> "NDArray":
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
    return full_grads / n  # type: ignore


def linear_regression_analytical_derivative_d2_theta(
    linear_model: Tuple["NDArray", "NDArray"], x: "NDArray", y: "NDArray"
) -> "NDArray":
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
    return full_matrix / n  # type: ignore


def linear_regression_analytical_derivative_d_x_d_theta(
    linear_model: Tuple["NDArray", "NDArray"], x: "NDArray", y: "NDArray"
) -> "NDArray":
    """
    :param linear_model: A tuple of np.ndarray of shape [NxM] and [N] representing A and b respectively.
    :param x: A np.ndarray of shape [BxM].
    :param y: A np.nparray of shape [BxN].
    :returns: A np.ndarray of shape [Bx((N+1)*M)xM], representing the derivative.
    """

    A, b = linear_model
    N, M = tuple(A.shape)
    residuals = x @ A.T + b - y
    B = len(x)
    outer_product_matrix = np.einsum("ab,ic->iacb", A, x)
    outer_product_matrix = np.reshape(outer_product_matrix, [B, M * N, M])
    tiled_identity = np.tile(np.expand_dims(np.eye(M), axis=0), [B, N, 1])
    outer_product_matrix += tiled_identity * np.expand_dims(
        np.repeat(residuals, M, axis=1), axis=2
    )
    b_part_derivative = np.tile(np.expand_dims(A, axis=0), [B, 1, 1])
    full_derivative = np.concatenate((outer_product_matrix, b_part_derivative), axis=1)
    return full_derivative / N  # type: ignore


def min_distance_points_to_line_2d(
    p: "NDArray", a: "NDArray", b: "NDArray"
) -> Tuple["NDArray", "NDArray"]:
    """
    Closed-form solution for minimum distance of point to line specified by dot(a, x) + b = 0.
    :param p: A 2-dimensional matrix [NxD] representing the points.
    :param a: A 1-dimensional vector [D] representing the slope.
    :param b: The offset of the line.
    :returns: A 1-dimensional vector [N] with the shortest distance for each point to the line.
    """
    a = np.reshape(a, [2, 1])
    r = np.abs(p @ a + b) / np.sqrt(np.sum(a**2))
    return r[:, 0]  # type: ignore


def get_running_avg_variance(
    previous_avg: "FloatOrArray",
    previous_variance: "FloatOrArray",
    new_value: "FloatOrArray",
    count: int,
) -> Tuple["FloatOrArray", "FloatOrArray"]:
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


def top_k_value_accuracy(y_true: "NDArray", y_pred: "NDArray", k: int = 3) -> float:
    """Computes the top-k accuracy for the estimated values by comparing indices of the highest k values

    :param y_true: Exact/true value
    :param y_pred: Predicted/estimated value
    :param k: Number of the highest values used to compute accuracy
    """
    top_k_exact_values = np.argsort(y_true)[-k:]
    top_k_pred_values = np.argsort(y_pred)[-k:]
    top_k_accuracy = len(np.intersect1d(top_k_exact_values, top_k_pred_values)) / k
    return top_k_accuracy
