"""
This module contains routines for numerical computations used across the
library.
"""
from __future__ import annotations

from itertools import chain, combinations
from typing import Collection, Generator, Iterator, Optional, Tuple, TypeVar, overload

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "running_moments",
    "linear_regression_analytical_derivative_d2_theta",
    "linear_regression_analytical_derivative_d_theta",
    "linear_regression_analytical_derivative_d_x_d_theta",
    "num_samples_permutation_hoeffding",
    "powerset",
    "random_matrix_with_condition_number",
    "random_subset",
    "random_powerset",
    "random_subset_of_size",
    "top_k_value_accuracy",
]

T = TypeVar("T", bound=np.generic)


def powerset(s: NDArray[T]) -> Iterator[Collection[T]]:
    """Returns an iterator for the power set of the argument.

     Subsets are generated in sequence by growing size. See
     :func:`random_powerset` for random sampling.

    >>> import numpy as np
    >>> from pydvl.utils.numeric import powerset
    >>> list(powerset(np.array((1,2))))
    [(), (1,), (2,), (1, 2)]

    Args:
         s: The set to use
     Returns:
        An iterator
    """
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def num_samples_permutation_hoeffding(eps: float, delta: float, u_range: float) -> int:
    """Lower bound on the number of samples required for MonteCarlo Shapley to
    obtain an (ε,δ)-approximation.

    That is: with probability 1-δ, the estimated value for one data point will
    be ε-close to the true quantity, if at least this many permutations are
    sampled.

    Args:
        eps: ε > 0
        delta: 0 < δ <= 1
        u_range: Range of the :class:`~pydvl.utils.utility.Utility` function
    Returns:
        Number of _permutations_ required to guarantee ε-correct Shapley
        values with probability 1-δ
    """
    return int(np.ceil(np.log(2 / delta) * 2 * u_range**2 / eps**2))


def random_subset(s: NDArray[T], q: float = 0.5) -> NDArray[T]:
    """Returns one subset at random from ``s``.

    Args:
        s: set to sample from
        q: Sampling probability for elements. The default 0.5 yields a
            uniform distribution over the power set of s.
    Returns:
        the subset
    """
    rng = np.random.default_rng()
    selection = rng.uniform(size=len(s)) > q
    return s[selection]


def random_powerset(
    s: NDArray[T], n_samples: Optional[int] = None, q: float = 0.5
) -> Generator[NDArray[T], None, None]:
    """Samples subsets from the power set of the argument, without
    pre-generating all subsets and in no order.

    See `powerset()` if you wish to deterministically generate all subsets.

    To generate subsets, `len(s)` Bernoulli draws with probability `q` are
    drawn. The default value of `q = 0.5` provides a uniform distribution over
    the power set of `s`. Other choices can be used e.g. to implement
    :func:`Owen sampling
    <pydvl.value.shapley.montecarlo.owen_sampling_shapley>`.

    Args:
        s: set to sample from
        n_samples: if set, stop the generator after this many steps.
            Defaults to `np.iinfo(np.int32).max`
        q: Sampling probability for elements. The default 0.5 yields a
            uniform distribution over the power set of s.

    Returns:
        Samples from the power set of s

    :raises: ValueError: if the element sampling probability is not in [0,1]

    """
    if q < 0 or q > 1:
        raise ValueError("Element sampling probability must be in [0,1]")

    total = 1
    if n_samples is None:
        n_samples = np.iinfo(np.int32).max
    while total <= n_samples:
        yield random_subset(s, q)
        total += 1


def random_subset_of_size(s: NDArray[T], size: int) -> NDArray[T]:
    """Samples a random subset of given size uniformly from the powerset
    of ``s``.

    Args:
        s: Set to sample from
        size: Size of the subset to generate
    Returns:
        The subset

    :raises ValueError: If size > len(s)
    """
    if size > len(s):
        raise ValueError("Cannot sample subset larger than set")
    rng = np.random.default_rng()
    return rng.choice(s, size=size, replace=False)


def random_matrix_with_condition_number(n: int, condition_number: float) -> "NDArray":
    """Constructs a square matrix with a given condition number.

    Taken from:
    https://gist.github.com/bstellato/23322fe5d87bb71da922fbc41d658079#file-random_mat_condition_number-py

    Also see:
    https://math.stackexchange.com/questions/1351616/condition-number-of-ata.

    Args:
        n: size of the matrix
        condition_number: duh
    Returns:
        An (n,n) matrix with the requested condition number.
    """
    if n < 2:
        raise ValueError("Matrix size must be at least 2")

    if condition_number <= 1:
        raise ValueError("Condition number must be greater than 1")

    log_condition_number = np.log(condition_number)
    exp_vec = np.arange(
        -log_condition_number / 4.0,
        log_condition_number * (n + 1) / (4 * (n - 1)),
        log_condition_number / (2.0 * (n - 1)),
    )
    exp_vec = exp_vec[:n]
    s: np.ndarray = np.exp(exp_vec)
    S = np.diag(s)
    U, _ = np.linalg.qr((np.random.rand(n, n) - 5.0) * 200)
    V, _ = np.linalg.qr((np.random.rand(n, n) - 5.0) * 200)
    P: np.ndarray = U.dot(S).dot(V.T)
    P = P.dot(P.T)
    return P


def linear_regression_analytical_derivative_d_theta(
    linear_model: Tuple["NDArray", "NDArray"], x: "NDArray", y: "NDArray"
) -> "NDArray":
    """
    Args:
        linear_model: A tuple of np.ndarray' of shape [NxM] and [N] representing A and b respectively.
        x: A np.ndarray of shape [BxM].
        y: A np.nparray of shape [BxN].
    Returns:
        A np.ndarray of shape [Bx((N+1)*M)], where each row vector is [d_theta L(x, y), d_b L(x, y)]
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
    Args:
        linear_model: A tuple of arrays of shape [NxM] and [N] representing A
            and b respectively.
        x: An array of shape [BxM],
        y: An array of shape [BxN].
    Returns:
        An array of shape [((N+1)*M)x((N+1)*M)], representing the Hessian.
        It gets averaged over all samples.
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
    Args:
        linear_model: A tuple of np.ndarray of shape [NxM] and [N] representing A and b respectively.
        x: A np.ndarray of shape [BxM].
        y: A np.nparray of shape [BxN].
    Returns:
        A np.ndarray of shape [Bx((N+1)*M)xM], representing the derivative.
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


@overload
def running_moments(
    previous_avg: float, previous_variance: float, count: int, new_value: float
) -> Tuple[float, float]:
    ...


@overload
def running_moments(
    previous_avg: NDArray[np.float_],
    previous_variance: NDArray[np.float_],
    count: int,
    new_value: NDArray[np.float_],
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    ...


def running_moments(
    previous_avg: float | NDArray[np.float_],
    previous_variance: float | NDArray[np.float_],
    count: int,
    new_value: float | NDArray[np.float_],
) -> Tuple[float | NDArray[np.float_], float | NDArray[np.float_]]:
    """Uses Welford's algorithm to calculate the running average and variance of
     a set of numbers.

    See `Welford's algorithm in wikipedia <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm>`_

    !!! Warning
       This is not really using Welford's correction for numerical stability
       for the variance. (FIXME)

    !!! Todo
       This could be generalised to arbitrary moments. See `this paper <https://www.osti.gov/biblio/1028931>`_

    Args:
        previous_avg: average value at previous step
        previous_variance: variance at previous step
        count: number of points seen so far
        new_value: new value in the series of numbers
    Returns:
        new_average, new_variance, calculated with the new count
    """
    # broadcasted operations seem not to be supported by mypy, so we ignore the type
    new_average = (new_value + count * previous_avg) / (count + 1)  # type: ignore
    new_variance = previous_variance + (
        (new_value - previous_avg) * (new_value - new_average) - previous_variance
    ) / (count + 1)
    return new_average, new_variance


def top_k_value_accuracy(
    y_true: NDArray[np.float_], y_pred: NDArray[np.float_], k: int = 3
) -> float:
    """Computes the top-k accuracy for the estimated values by comparing indices
    of the highest k values.

    Args:
        y_true: Exact/true value
        y_pred: Predicted/estimated value
        k: Number of the highest values taken into account
    Returns:
        Accuracy
    """
    top_k_exact_values = np.argsort(y_true)[-k:]
    top_k_pred_values = np.argsort(y_pred)[-k:]
    top_k_accuracy = len(np.intersect1d(top_k_exact_values, top_k_pred_values)) / k
    return top_k_accuracy
