import math
from functools import lru_cache
from typing import Callable, Optional

import numpy as np
from numpy._typing import NDArray

from pydvl.utils import Utility, get_running_avg_variance, maybe_progress
from pydvl.value import ValuationResult, ValuationStatus
from pydvl.value.sampler import OwenSampler, PermutationSampler, Sampler, UniformSampler


SemiValue = Callable[[Utility, Sampler, ...], ValuationResult]
Coefficient = Callable[[int, int], int]
StoppingCriterion = Callable[
    [int, NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]], ValuationStatus
]


def _semivalues(
    sampler: Sampler,
    u: Utility,
    coefficient: Optional[Coefficient] = None,
    stop: Optional[StoppingCriterion] = None,
    *,
    progress: bool = False,
    job_id: int = 1,
) -> ValuationResult:
    """Helper function sent to workers for all computations of semi-values.

    :param u: Utility object with model, data, and scoring function.
    :param sampler: The subset sampler to use for utility computations.
    :param progress: Whether to display progress bars for each job.
    :param job_id: id to use for reporting progress.
    :return: Object with valuation results.
    """
    n = len(u.data)
    values = np.zeros(n, dtype=np.float_)
    variances = np.zeros(n, dtype=np.float_)
    counts = np.zeros(n, dtype=np.int_)
    if coefficient is None:
        coefficient = lambda _, __: 1.0  # noqa
        coefficient.__name__ = "none"
    status = ValuationStatus.Pending
    for idx, s in maybe_progress(sampler, progress, position=job_id):
        marginal = (u({idx}.union(s)) - u(s)) * coefficient(n, len(s))
        values[idx], variances[idx] = get_running_avg_variance(
            values[idx], variances[idx], marginal, counts[idx]
        )
        counts[idx] += 1
        if stop is not None:
            status = stop(values, variances, counts)
            if status != ValuationStatus.Pending:
                break

    if stop is None:
        status = ValuationStatus.MaxIterations
    return ValuationResult(
        algorithm=f"semivalue_{coefficient.__name__}_{sampler.__class__.__name__}",
        status=status,
        values=values,
        stderr=np.sqrt(variances / np.maximum(1, counts)),
    )


def variance_criterion(
    values: NDArray[np.float_],
    variances: NDArray[np.float_],
    counts: NDArray[np.int_],
    eps: float,
    values_ratio: float,
) -> ValuationStatus:
    r"""Checks that the variance of the values is below a threshold.

    A value $v_i$ is considered to be below the threshold if the associated
    standard deviation $s_i = \sqrt{\var{v_ii}}$ fulfills:

    $$ s_i \lt | \eps v_i | . $$

    In other words, the computation of the value for sample $x_i$ is considered
    complete once the square root of its variance is within a fraction $\eps$ of
    the value.

    .. fixme::
       This ad-hoc criterion might fail under some circumstances. Complement it
       or find something better.

    :param values: The array of data values.
    :param variances: The array of variances for each value.
    :param counts: The array counting the number of updates performed so far
        to each value.
    :param eps: Threshold multiplier for the values
    :param values_ratio: Amount of values that must fulfill the criterion
    :return: :attr:`~pydvl.value.results.ValuationStatus.Converged` if at least
        a fraction of `values_ratio` of the values has standard error below the
        threshold.

    """
    if len(values) == 0:
        raise ValueError("Empty values array")
    if len(values) != len(variances) or len(values) != len(counts):
        raise ValueError("Mismatching array lengths")
    if (
        not isinstance(values, np.ndarray)
        or not isinstance(variances, np.ndarray)
        or not isinstance(counts, np.ndarray)
    ):
        raise TypeError("Input must be NDArray")

    sat_ratio = np.count_nonzero(variances <= np.abs(eps * values)) / len(values)
    if sat_ratio >= values_ratio:
        return ValuationStatus.Converged
    return ValuationStatus.Pending


@lru_cache
def combinatorial_coefficient(n: int, ns: int) -> int:
    # Correction coming from Monte Carlo integration so that the mean of the
    # marginals converges to the value: the uniform distribution over the
    # powerset of a set with n-1 elements has mass 2^{n-1} over each subset. The
    # additional factor n corresponds to the one in the Shapley definition
    return 2 ** (n - 1) / (n * math.comb(n - 1, ns))


@lru_cache
def banzhaf_coefficient(n: int, _: int):
    return 1.0 / 2 ** (n - 1)


def beta_coefficient(alpha: int, beta: int):
    @lru_cache
    def coefficient_w_tilde(n: int, ns: int):
        j = ns + 1
        p1 = (beta + np.arange(0, j - 1)).prod() * (alpha + np.arange(0, n - j)).prod()
        p2 = (alpha + beta + np.arange(0, n - 1)).prod()
        return n * math.comb(n - 1, ns) * p1 / p2

    return coefficient_w_tilde


def shapley(u: Utility, max_subsets: int):
    sampler = UniformSampler(u.data.indices, max_subsets=max_subsets)
    return _semivalues(sampler, u, combinatorial_coefficient)


def permutation_shapley(u: Utility, max_subsets: int):
    sampler = PermutationSampler(u.data.indices, max_subsets=max_subsets)
    return _semivalues(sampler, u)


def beta_shapley(
    u: Utility, alpha: int, beta: int, max_subsets: int
) -> ValuationResult:
    sampler = PermutationSampler(u.data.indices, max_subsets=max_subsets)
    return _semivalues(sampler, u, beta_coefficient(alpha, beta))


def beta_shapley_paper(
    u: Utility, alpha: int, beta: int, max_subsets: int
) -> ValuationResult:
    sampler = UniformSampler(u.data.indices, max_subsets=max_subsets)
    return _semivalues(sampler, u, beta_coefficient(alpha, beta))


def banzhaf_index(u: Utility, max_subsets: int):
    sampler = PermutationSampler(u.data.indices, max_subsets=max_subsets)
    return _semivalues(sampler, u)


def owen_shapley(u: Utility, max_subsets: int, max_q: int):
    sampler = OwenSampler(u.data.indices, max_subsets=max_subsets)
    raise NotImplementedError("to do")
