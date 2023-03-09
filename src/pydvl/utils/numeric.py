"""
This module contains routines for numerical computations used across the
library.
"""
from __future__ import annotations

import logging
import os
import random
import time
from itertools import chain, combinations
from typing import (
    Collection,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    cast,
    overload,
)

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "running_moments",
    "num_samples_permutation_hoeffding",
    "powerset",
    "random_matrix_with_condition_number",
    "random_subset",
    "random_powerset",
    "random_powerset_group_conditional",
    "random_subset_of_size",
    "top_k_value_accuracy",
]


logger = logging.getLogger(__name__)


T = TypeVar("T", bound=np.generic)


def powerset(s: NDArray[T]) -> Iterator[Collection[T]]:
    """Returns an iterator for the power set of the argument.

     Subsets are generated in sequence by growing size. See
     :func:`random_powerset` for random sampling.

    >>> import numpy as np
    >>> from pydvl.utils.numeric import powerset
    >>> list(powerset(np.array((1,2))))
    [(), (1,), (2,), (1, 2)]

     :param s: The set to use
     :return: An iterator
     :raises TypeError: If the argument is not an ``Iterable``.
    """
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def num_samples_permutation_hoeffding(eps: float, delta: float, u_range: float) -> int:
    """Lower bound on the number of samples required for MonteCarlo Shapley to
    obtain an (ε,δ)-approximation.

    That is: with probability 1-δ, the estimated value for one data point will
    be ε-close to the true quantity, if at least this many permutations are
    sampled.

    :param eps: ε > 0
    :param delta: 0 < δ <= 1
    :param u_range: Range of the :class:`~pydvl.utils.utility.Utility` function
    :return: Number of _permutations_ required to guarantee ε-correct Shapley
        values with probability 1-δ
    """
    return int(np.ceil(np.log(2 / delta) * 2 * u_range**2 / eps**2))


def random_subset(s: NDArray[T], q: float = 0.5) -> NDArray[T]:
    """Returns one subset at random from ``s``.

    :param s: set to sample from
    :param q: Sampling probability for elements. The default 0.5 yields a
        uniform distribution over the power set of s.
    :return: the subset
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

    :param s: set to sample from
    :param n_samples: if set, stop the generator after this many steps.
        Defaults to `np.iinfo(np.int32).max`
    :param q: Sampling probability for elements. The default 0.5 yields a
        uniform distribution over the power set of s.

    :return: Samples from the power set of s
    :raises: TypeError: if the data `s` is not a NumPy array
    :raises: ValueError: if the element sampling probability is not in [0,1]

    """
    if not isinstance(s, np.ndarray):
        raise TypeError("Set must be an NDArray")
    if q < 0 or q > 1:
        raise ValueError("Element sampling probability must be in [0,1]")

    total = 1
    if n_samples is None:
        n_samples = np.iinfo(np.int32).max
    while total <= n_samples:
        yield random_subset(s, q)
        total += 1


def random_powerset_group_conditional(
    s: NDArray[T],
    groups: NDArray[np.int_],
    min_elements_per_group: int = 1,
) -> Generator[NDArray[T], None, None]:
    """
    Draw infinite random group-conditional subsets from the passed set s. It is ensured
    that in each sampled set, each unique group is represented at least ``min_elements``
    times. The groups are specified as integers for all elements of the set separately.

    :param s: Vector of size N representing the set to sample elements from.
    :param groups: Vector of size N containing the group as an integer for each element.
    :param min_elements_per_group: The minimum number of elements for each group.

    :return: Generated draw from the power set of s with ``min_elements`` of each group.
    :raises: TypeError: If the data ``s`` or ``groups`` is not a NumPy array.
    :raises: ValueError: If the length of ``s``and ``groups`` different or
        ``min_elements`` is smaller than 0.
    """
    if not isinstance(s, np.ndarray):
        raise TypeError("Set must be an NDArray")

    if not isinstance(groups, np.ndarray):
        raise TypeError("Labels must be an NDArray")

    if len(groups) != len(s):
        raise ValueError("Set and labels have to be of same size.")

    if min_elements_per_group < 0:
        raise ValueError(
            f"Parameter min_elements={min_elements_per_group} needs to be bigger or equal to 0."
        )

    if min_elements_per_group == 0:
        logger.warning(
            "It is recommended to ensure at least one element of each group is"
            " contained in the sampled and yielded set."
        )

    rng = np.random.default_rng()
    unique_labels = np.unique(groups)

    while True:
        subsets: List[NDArray[T]] = []
        for label in unique_labels:
            label_indices = np.asarray(np.where(groups == label)[0])
            subset_length = int(
                rng.integers(
                    min(min_elements_per_group, len(label_indices)),
                    len(label_indices) + 1,
                )
            )
            if subset_length > 0:
                subsets.append(random_subset_of_size(s[label_indices], subset_length))

        if len(subsets) > 0:
            subset = np.concatenate(tuple(subsets))
            rng.shuffle(subset)
            yield subset
        else:
            yield np.array([])


def random_subset_of_size(s: NDArray[T], size: int) -> NDArray[T]:
    """Samples a random subset of given size uniformly from the powerset
    of ``s``.

    :param s: Set to sample from
    :param size: Size of the subset to generate
    :return: The subset
    :raises ValueError: If size > len(s)
    """
    if size > len(s):
        raise ValueError("Cannot sample subset larger than set")
    rng = np.random.default_rng()
    return rng.choice(s, size=size, replace=False)


def random_matrix_with_condition_number(n: int, condition_number: float) -> NDArray:
    """Constructs a square matrix with a given condition number.

    Taken from:
    https://gist.github.com/bstellato/23322fe5d87bb71da922fbc41d658079#file-random_mat_condition_number-py

    Also see:
    https://math.stackexchange.com/questions/1351616/condition-number-of-ata.

    :param n: size of the matrix
    :param condition_number: duh
    :return: An (n,n) matrix with the requested condition number.
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

    See `Welford's algorithm in wikipedia
    <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm>`_

    .. warning::
       This is not really using Welford's correction for numerical stability
       for the variance. (FIXME)

    .. todo::
       This could be generalised to arbitrary moments. See `this paper
       <https://www.osti.gov/biblio/1028931>`_


    :param previous_avg: average value at previous step
    :param previous_variance: variance at previous step
    :param count: number of points seen so far
    :param new_value: new value in the series of numbers
    :return: new_average, new_variance, calculated with the new count
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

    :param y_true: Exact/true value
    :param y_pred: Predicted/estimated value
    :param k: Number of the highest values taken into account
    :return: Accuracy
    """
    top_k_exact_values = np.argsort(y_true)[-k:]
    top_k_pred_values = np.argsort(y_pred)[-k:]
    top_k_accuracy = len(np.intersect1d(top_k_exact_values, top_k_pred_values)) / k
    return top_k_accuracy
