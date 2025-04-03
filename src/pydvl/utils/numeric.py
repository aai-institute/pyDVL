"""
This module contains routines for numerical computations used across the library.
"""

from __future__ import annotations

from itertools import chain, combinations
from typing import (
    Collection,
    Generator,
    Iterator,
    Optional,
    Sequence,
    TypeVar,
)

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

from pydvl.utils.types import Seed

__all__ = [
    "complement",
    "logcomb",
    "logexp",
    "log_running_moments",
    "logsumexp_two",
    "num_samples_permutation_hoeffding",
    "powerset",
    "random_matrix_with_condition_number",
    "random_subset",
    "random_powerset",
    "random_powerset_label_min",
    "random_subset_of_size",
    "running_moments",
    "top_k_value_accuracy",
]

T = TypeVar("T", bound=np.generic)


def complement(
    include: NDArray[T], exclude: NDArray[T] | Sequence[T | None]
) -> NDArray[T]:
    """Returns the complement of the set of indices excluding the given
    indices.

    Args:
        include: The set of indices to consider.
        exclude: The indices to exclude from the complement. These must be a subset
            of `include`. If an index is `None` it is ignored.

    Returns:
        The complement of the set of indices excluding the given indices.
    """
    _exclude = np.array([i for i in exclude if i is not None], dtype=include.dtype)
    return np.setdiff1d(include, _exclude).astype(np.int_)


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
        subsets: list[NDArray[T]] = []
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


def running_moments(
    previous_avg: float,
    previous_variance: float,
    count: int,
    new_value: float,
    unbiased: bool = True,
) -> tuple[float, float]:
    """Calculates running average and variance of a series of numbers.

    See [Welford's algorithm in
    wikipedia](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)

    !!! Warning
        This is not really using Welford's correction for numerical stability
        for the variance. (FIXME)

    !!! Todo
        This could be generalised to arbitrary moments. See [this
        paper](https://www.osti.gov/biblio/1028931)

    Args:
        previous_avg: average value at previous step.
        previous_variance: variance at previous step.
        count: number of points seen so far,
        new_value: new value in the series of numbers.
        unbiased: whether to use the unbiased variance estimator (same as `np.var` with
            `ddof=1`).
    Returns:
        new_average, new_variance, calculated with the new count
    """
    delta = new_value - previous_avg
    new_average = previous_avg + delta / (count + 1)

    if unbiased:
        if count > 0:
            new_variance = (
                previous_variance + delta**2 / (count + 1) - previous_variance / count
            )
        else:
            new_variance = 0.0
    else:
        new_variance = previous_variance + (
            delta * (new_value - new_average) - previous_variance
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


def logcomb(n: int, k: int) -> float:
    r"""Computes the log of the binomial coefficient (n choose k).

    $$
    \begin{array}{rcl}
        \log\binom{n}{k} & = & \log(n!) - \log(k!) - \log((n-k)!) \\
                         & = & \log\Gamma(n+1) - \log\Gamma(k+1) - \log\Gamma(n-k+1).
    \end{array}
    $$

    Args:
        n: Total number of elements
        k: Number of elements to choose
    Returns:
        The log of the binomial coefficient
        """
    if k < 0 or k > n or n < 0:
        raise ValueError(f"Invalid arguments: n={n}, k={k}")
    return float(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))


def logexp(x: float, a: float) -> float:
    """Computes log(x^a).

    Args:
        x: Base
        a: Exponent
    Returns
        a * log(x)
    """
    return float(a * np.log(x))


def logsumexp_two(log_a: float, log_b: float) -> float:
    r"""Numerically stable computation of log(exp(log_a) + exp(log_b)).

    Uses standard log sum exp trick:

    $$
    \log(\exp(\log a) + \exp(\log b)) = m + \log(\exp(\log a - m) + \exp(\log b - m)),
    $$

    where $m = \max(\log a, \log b)$.

    Args:
        log_a: Log of the first value
        log_b: Log of the second value
    Returns:
        The log of the sum of the exponentials
    """
    assert log_a < np.inf and log_b < np.inf, f"log_a={log_a}, log_b={log_b}"

    if log_a == -np.inf:
        return log_b
    if log_b == -np.inf:
        return log_a
    m = max(log_a, log_b)
    return float(m + np.log(np.exp(log_a - m) + np.exp(log_b - m)))


def log_running_moments(
    previous_log_sum_pos: float,
    previous_log_sum_neg: float,
    previous_log_sum2: float,
    count: int,
    new_log_value: float,
    new_sign: int,
    unbiased: bool = True,
) -> tuple[float, float, float, float, float]:
    """
    Update running moments when the new value is provided in log space,
    allowing for negative values via an explicit sign.

    Here the actual value is x = new_sign * exp(new_log_value). Rather than
    updating the arithmetic sum S = sum(x) and S2 = sum(x^2) directly, we maintain:

       L_S+ = log(sum_{i: x_i >= 0} x_i)
       L_S- = log(sum_{i: x_i < 0} |x_i|)
       L_S2 = log(sum_i x_i^2)

    The running mean is then computed as:

         mean = exp(L_S+) - exp(L_S-)

    and the second moment is:

         second_moment = exp(L_S2 - log(count))

    so that the variance is:

         variance = second_moment - mean^2

    For the unbiased (sample) estimator, we scale the variance by count/(count-1)
    when count > 1 (and define variance = 0 when count == 1).

    Args:
        previous_log_sum_pos: running log(sum of positive contributions), or -inf if none.
        previous_log_sum_neg: running log(sum of negative contributions in absolute
            value), or -inf if none.
        previous_log_sum2: running log(sum of squares) so far (or -inf if none).
        count: number of points processed so far.
        new_log_value: log(|x_new|), where x_new is the new value.
        new_sign: sign of the new value (should be +1, 0, or -1).
        unbiased: if True, compute the unbiased estimator of the variance.

    Returns:
        new_mean: running mean in the linear domain.
        new_variance: running variance in the linear domain.
        new_log_sum_pos: updated running log(sum of positive contributions).
        new_log_sum_neg: updated running log(sum of negative contributions).
        new_log_sum2: updated running log(sum of squares).
        new_count: updated count.
    """

    if count == 0:
        if new_sign >= 0:
            new_log_sum_pos = new_log_value
            new_log_sum_neg = -np.inf  # No negative contribution yet.
        else:
            new_log_sum_pos = -np.inf
            new_log_sum_neg = new_log_value
        new_log_sum2 = 2 * new_log_value
    else:
        if new_sign >= 0:
            new_log_sum_pos = logsumexp_two(previous_log_sum_pos, new_log_value)
            new_log_sum_neg = previous_log_sum_neg
        else:
            new_log_sum_neg = logsumexp_two(previous_log_sum_neg, new_log_value)
            new_log_sum_pos = previous_log_sum_pos
        new_log_sum2 = logsumexp_two(previous_log_sum2, 2 * new_log_value)
    new_count = count + 1

    # Compute 1st and 2nd moments in the linear domain.
    pos_sum = np.exp(new_log_sum_pos) if new_log_sum_pos != -np.inf else 0.0
    neg_sum = np.exp(new_log_sum_neg) if new_log_sum_neg != -np.inf else 0.0
    new_mean = (pos_sum - neg_sum) / new_count

    second_moment = np.exp(new_log_sum2 - np.log(new_count))

    # Compute variance using either the population or unbiased estimator.
    if unbiased:
        if new_count > 1:
            new_variance = new_count / (new_count - 1) * (second_moment - new_mean**2)
        else:
            new_variance = 0.0
    else:
        new_variance = second_moment - new_mean**2

    return new_mean, new_variance, new_log_sum_pos, new_log_sum_neg, new_log_sum2
