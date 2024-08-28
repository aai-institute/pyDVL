"""
This module contains routines for numerical computations used across the
library.
"""
from __future__ import annotations

from itertools import chain, combinations
from typing import (
    Collection,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    overload,
)

import numpy as np
from numpy.typing import NDArray

from pydvl.utils.types import Seed

__all__ = [
    "running_moments",
    "num_samples_permutation_hoeffding",
    "powerset",
    "random_matrix_with_condition_number",
    "random_subset",
    "random_powerset",
    "random_powerset_label_min",
    "random_subset_of_size",
    "top_k_value_accuracy",
]

T = TypeVar("T", bound=np.generic)


def powerset(s: NDArray[T]) -> Iterator[Collection[T]]:
    """Returns an iterator for the power set of the argument.

     Subsets are generated in sequence by growing size. See
     [random_powerset()][pydvl.utils.numeric.random_powerset] for random
     sampling.

    ??? Example
        ``` pycon
        >>> import numpy as np
        >>> from pydvl.utils.numeric import powerset
        >>> list(powerset(np.array((1,2))))
        [(), (1,), (2,), (1, 2)]
        ```

    Args:
         s: The set to use

    Returns:
        An iterator over all subsets of the set of indices `s`.
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
        u_range: Range of the [Utility][pydvl.utils.utility.Utility] function

    Returns:
        Number of _permutations_ required to guarantee ε-correct Shapley
            values with probability 1-δ
    """
    return int(np.ceil(np.log(2 / delta) * 2 * u_range**2 / eps**2))


def random_subset(
    s: NDArray[T], q: float = 0.5, seed: Optional[Seed] = None
) -> NDArray[T]:
    """Returns one subset at random from ``s``.

    Args:
        s: set to sample from
        q: Sampling probability for elements. The default 0.5 yields a
            uniform distribution over the power set of s.
        seed: Either an instance of a numpy random number generator or a seed
            for it.

    Returns:
        The subset
    """
    rng = np.random.default_rng(seed)
    selection = rng.uniform(size=len(s)) < q
    return s[selection]


def random_powerset(
    s: NDArray[T],
    n_samples: Optional[int] = None,
    q: float = 0.5,
    seed: Optional[Seed] = None,
) -> Generator[NDArray[T], None, None]:
    """Samples subsets from the power set of the argument, without
    pre-generating all subsets and in no order.

    See [powerset][pydvl.utils.numeric.powerset] if you wish to deterministically generate all subsets.

    To generate subsets, `len(s)` Bernoulli draws with probability `q` are
    drawn. The default value of `q = 0.5` provides a uniform distribution over
    the power set of `s`. Other choices can be used e.g. to implement
    [owen_sampling_shapley][pydvl.value.shapley.owen.owen_sampling_shapley].

    Args:
        s: set to sample from
        n_samples: if set, stop the generator after this many steps.
            Defaults to `np.iinfo(np.int32).max`
        q: Sampling probability for elements. The default 0.5 yields a
            uniform distribution over the power set of s.
        seed: Either an instance of a numpy random number generator or a seed for it.

    Returns:
        Samples from the power set of `s`.

    Raises:
        ValueError: if the element sampling probability is not in [0,1]

    """
    if q < 0 or q > 1:
        raise ValueError("Element sampling probability must be in [0,1]")

    rng = np.random.default_rng(seed)
    total = 1
    if n_samples is None:
        n_samples = np.iinfo(np.int32).max
    while total <= n_samples:
        yield random_subset(s, q, seed=rng)
        total += 1


def random_powerset_label_min(
    s: NDArray[T],
    labels: NDArray[np.int_],
    min_elements_per_label: int = 1,
    seed: Optional[Seed] = None,
) -> Generator[NDArray[T], None, None]:
    """Draws random subsets from `s`, while ensuring that at least
    `min_elements_per_label` elements per label are included in the draw. It can be used
    for classification problems to ensure that a set contains information for all labels
    (or not if `min_elements_per_label=0`).

    Args:
        s: Set to sample from
        labels: Labels for the samples
        min_elements_per_label: Minimum number of elements for each label.
        seed: Either an instance of a numpy random number generator or a seed for it.

    Returns:
        Generated draw from the powerset of s with `min_elements_per_label` for each
        label.

    Raises:
        ValueError: If `s` and `labels` are of different length or
            `min_elements_per_label` is smaller than 0.
    """
    if len(labels) != len(s):
        raise ValueError("Set and labels have to be of same size.")

    if min_elements_per_label < 0:
        raise ValueError(
            f"Parameter min_elements={min_elements_per_label} needs to be bigger or "
            f"equal to 0."
        )

    rng = np.random.default_rng(seed)
    unique_labels = np.unique(labels)

    while True:
        subsets: List[NDArray[T]] = []
        for label in unique_labels:
            label_indices = np.asarray(np.where(labels == label)[0])
            subset_size = int(
                rng.integers(
                    min(min_elements_per_label, len(label_indices)),
                    len(label_indices) + 1,
                )
            )
            if subset_size > 0:
                subsets.append(
                    random_subset_of_size(s[label_indices], subset_size, seed=rng)
                )

        if len(subsets) > 0:
            subset = np.concatenate(tuple(subsets))
            rng.shuffle(subset)
            yield subset
        else:
            yield np.array([], dtype=s.dtype)


def random_subset_of_size(
    s: NDArray[T], size: int, seed: Optional[Seed] = None
) -> NDArray[T]:
    """Samples a random subset of given size uniformly from the powerset
    of `s`.

    Args:
        s: Set to sample from
        size: Size of the subset to generate
        seed: Either an instance of a numpy random number generator or a seed for it.

    Returns:
        The subset

    Raises
        ValueError: If size > len(s)
    """
    if size > len(s):
        raise ValueError("Cannot sample subset larger than set")
    rng = np.random.default_rng(seed)
    return rng.choice(s, size=size, replace=False)


def random_matrix_with_condition_number(
    n: int, condition_number: float, seed: Optional[Seed] = None
) -> NDArray:
    """Constructs a square matrix with a given condition number.

    Taken from:
    [https://gist.github.com/bstellato/23322fe5d87bb71da922fbc41d658079#file-random_mat_condition_number-py](
    https://gist.github.com/bstellato/23322fe5d87bb71da922fbc41d658079#file-random_mat_condition_number-py)

    Also see:
    [https://math.stackexchange.com/questions/1351616/condition-number-of-ata](
    https://math.stackexchange.com/questions/1351616/condition-number-of-ata).

    Args:
        n: size of the matrix
        condition_number: duh
        seed: Either an instance of a numpy random number generator or a seed for it.

    Returns:
        An (n,n) matrix with the requested condition number.
    """
    if n < 2:
        raise ValueError("Matrix size must be at least 2")

    if condition_number <= 1:
        raise ValueError("Condition number must be greater than 1")

    rng = np.random.default_rng(seed)
    log_condition_number = np.log(condition_number)
    exp_vec = np.arange(
        -log_condition_number / 4.0,
        log_condition_number * (n + 1) / (4 * (n - 1)),
        log_condition_number / (2.0 * (n - 1)),
    )
    exp_vec = exp_vec[:n]
    s: np.ndarray = np.exp(exp_vec)
    S = np.diag(s)
    U, _ = np.linalg.qr((rng.uniform(size=(n, n)) - 5.0) * 200)
    V, _ = np.linalg.qr((rng.uniform(size=(n, n)) - 5.0) * 200)
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
    previous_avg: NDArray[np.float64],
    previous_variance: NDArray[np.float64],
    count: int,
    new_value: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    ...


def running_moments(
    previous_avg: float | NDArray[np.float64],
    previous_variance: float | NDArray[np.float64],
    count: int,
    new_value: float | NDArray[np.float64],
) -> Tuple[float | NDArray[np.float64], float | NDArray[np.float64]]:
    """Uses Welford's algorithm to calculate the running average and variance of
     a set of numbers.

    See [Welford's algorithm in wikipedia](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)

    !!! Warning
        This is not really using Welford's correction for numerical stability
        for the variance. (FIXME)

    !!! Todo
        This could be generalised to arbitrary moments. See [this paper](https://www.osti.gov/biblio/1028931)

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
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64], k: int = 3
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
