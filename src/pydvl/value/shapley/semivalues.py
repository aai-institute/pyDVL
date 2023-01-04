import math
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Protocol

import numpy as np

from pydvl.utils import Utility, get_running_avg_variance, maybe_progress
from pydvl.value import ValuationResult, ValuationStatus
from pydvl.value.sampler import OwenSampler, PermutationSampler, Sampler, UniformSampler

if TYPE_CHECKING:
    from numpy._typing import NDArray


Coefficient = Callable[[int, int], float]
StoppingCriterion = Callable[
    [int, "NDArray[np.float_]", "NDArray[np.float_]", "NDArray[np.int_]"],
    ValuationStatus,
]


class SemiValue(Protocol):
    def __call__(
        self, u: Utility, stop: StoppingCriterion, *args, **kwargs
    ) -> ValuationResult:
        ...


def _semivalues(
    u: Utility,
    sampler: Sampler,
    coefficient: Coefficient,
    stop: StoppingCriterion,
    *,
    progress: bool = False,
    job_id: int = 1,
) -> ValuationResult:
    """Helper function used for all computations of semi-values.

    :param u: Utility object with model, data, and scoring function.
    :param sampler: The subset sampler to use for utility computations.
    :param coefficient: The semivalue coefficient

    :param progress: Whether to display progress bars for each job.
    :param job_id: id to use for reporting progress.
    :return: Object with valuation results.
    """
    n = len(u.data)
    values = np.zeros(n, dtype=np.float_)
    variances = np.zeros(n, dtype=np.float_)
    counts = np.zeros(n, dtype=np.int_)
    status = ValuationStatus.Pending

    for step, (idx, s) in maybe_progress(enumerate(sampler), progress, position=job_id):
        marginal = (u({idx}.union(s)) - u(s)) * coefficient(n, len(s))
        values[idx], variances[idx] = get_running_avg_variance(
            values[idx], variances[idx], marginal, counts[idx]
        )
        counts[idx] += 1
        status = stop(step, values, variances, counts)
        if status != ValuationStatus.Pending:
            break

    return ValuationResult(
        algorithm=f"semivalue_{coefficient.__name__}_{sampler.__class__.__name__}",
        status=status,
        values=values,
        steps=counts.sum(),
        stderr=np.sqrt(variances / np.maximum(1, counts)),
    )


def max_samples_criterion(max_samples: np.int_) -> StoppingCriterion:
    def check_max_samples(
        step: int,
        values: "NDArray[np.float_]",
        variances: "NDArray[np.float_]",
        counts: "NDArray[np.int_]",
    ) -> ValuationStatus:
        if step >= max_samples:
            return ValuationStatus.MaxIterations
        return ValuationStatus.Pending

    return check_max_samples


def max_updates_criterion(max_updates: int, values_ratio: float) -> StoppingCriterion:
    """Checks whether a given fraction of all values has been updated at
    least ``maxiter`` times.
    :param max_updates: Maximal amount of updates for each value
    :param values_ratio: Amount of values that must fulfill the criterion
    :return: :attr:`~pydvl.value.results.ValuationStatus.Converged` if at least
        a fraction of `values_ratio` of the values has standard error below the
        threshold.

    """

    def check_max_updates(
        step: int,
        values: "NDArray[np.float_]",
        variances: "NDArray[np.float_]",
        counts: "NDArray[np.int_]",
    ) -> ValuationStatus:
        if np.count_nonzero(counts >= max_updates) / len(counts) >= values_ratio:
            return ValuationStatus.MaxIterations
        return ValuationStatus.Pending

    return check_max_updates


def stderr_criterion(eps: float, values_ratio: float) -> StoppingCriterion:
    r"""Checks that the standard error of the values is below a threshold.

    A value $v_i$ is considered to be below the threshold if the associated
    standard error $s_i = \sqrt{\var{v_ii}/n}$ fulfills:

    $$ s_i \lt | \eps v_i | . $$

    In other words, the computation of the value for sample $x_i$ is considered
    complete once the estimator for the standard error is within a fraction
    $\eps$ of the value.

    .. fixme::
       This ad-hoc will fail if the distribution of utilities for an index has
       high variance. We need something better, taking 1st or maybe 2nd order
       info into account.

    :param eps: Threshold multiplier for the values
    :param values_ratio: Amount of values that must fulfill the criterion
    :return: A convergence criterion returning
        :attr:`~pydvl.value.results.ValuationStatus.Converged` if at least a
        fraction of `values_ratio` of the values has standard error below the
        threshold.
    """

    def check_stderr(
        step: int,
        values: "NDArray[np.float_]",
        variances: "NDArray[np.float_]",
        counts: "NDArray[np.int_]",
    ) -> ValuationStatus:
        if len(values) == 0:
            raise ValueError("Empty values array")
        if len(values) != len(variances) or len(values) != len(counts):
            raise ValueError("Mismatching array lengths")

        if np.any(counts == 0):
            return ValuationStatus.Pending

        passing_ratio = np.count_nonzero(
            np.sqrt(variances / counts) <= np.abs(eps * values)
        ) / len(values)
        if passing_ratio >= values_ratio:
            return ValuationStatus.Converged
        return ValuationStatus.Pending

    return check_stderr


@lru_cache
def combinatorial_coefficient(n: int, ns: int) -> float:
    # Correction coming from Monte Carlo integration so that the mean of the
    # marginals converges to the value: the uniform distribution over the
    # powerset of a set with n-1 elements has mass 2^{n-1} over each subset.
    # The factor 1 / n corresponds to the one in the Shapley definition
    return 2 ** (n - 1) / (n * math.comb(n - 1, ns))


@lru_cache
def banzhaf_coefficient(n: int, _: int) -> float:
    return n / 2 ** (n - 1)


def permutation_coefficient(_: int, __: int) -> float:
    return 1.0


def beta_coefficient(alpha: int, beta: int):
    @lru_cache
    def coefficient_w_tilde(n: int, ns: int) -> float:
        j = ns + 1
        p1 = (beta + np.arange(0, j - 1)).prod() * (alpha + np.arange(0, n - j)).prod()
        p2 = (alpha + beta + np.arange(0, n - 1)).prod()
        return n * math.comb(n - 1, ns) * p1 / p2

    return coefficient_w_tilde


def shapley(u: Utility, criterion: StoppingCriterion):
    sampler = UniformSampler(u.data.indices)
    return _semivalues(u, sampler, combinatorial_coefficient, criterion)


def permutation_shapley(u: Utility, criterion: StoppingCriterion):
    sampler = PermutationSampler(u.data.indices)
    return _semivalues(u, sampler, permutation_coefficient, criterion)


def beta_shapley(
    u: Utility, criterion: StoppingCriterion, alpha: int, beta: int
) -> ValuationResult:
    sampler = PermutationSampler(u.data.indices)
    return _semivalues(u, sampler, beta_coefficient(alpha, beta), criterion)


def beta_shapley_paper(
    u: Utility, criterion: StoppingCriterion, alpha: int, beta: int
) -> ValuationResult:
    sampler = UniformSampler(u.data.indices)
    return _semivalues(u, sampler, beta_coefficient(alpha, beta), criterion)


def banzhaf_index(u: Utility, criterion: StoppingCriterion):
    sampler = PermutationSampler(u.data.indices)
    return _semivalues(u, sampler, banzhaf_coefficient, criterion)


def owen_shapley(u: Utility, criterion: StoppingCriterion, max_q: int):
    sampler = OwenSampler(
        u.data.indices, method=OwenSampler.Algorithm.Standard, n_steps=200
    )
    raise NotImplementedError("to do")
