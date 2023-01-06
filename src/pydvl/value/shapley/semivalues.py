r""" Computation of generic semi-values

A semi-value is a valuation function with generic form:

$$\v_\text{semi}(i) = \frac{1}{n}
      \sum_{i=1}^n w(k) \sum_{S \subset D_{-i}^{(k)}} [U(S_{+i})-U(S)],$$

where the coefficients $w(k)$ satisfy the property:

$$ \sum_{k=1}^n w(k) = n.$$

This module provides the core functionality for the computation of any such
semi-value. The interface is :class:`SemiValue`. The coefficients implement
:class:`Coefficient`.

.. todo::
   Finish documenting this.

"""

import math
import numpy as np
import scipy as sp

from functools import lru_cache
from typing import Protocol

from pydvl.utils import Utility, get_running_avg_variance, maybe_progress
from pydvl.value import ValuationResult, ValuationStatus
from pydvl.value.sampler import OwenSampler, PermutationSampler, Sampler, UniformSampler

from pydvl.value.shapley.stopping import StoppingCriterion, StoppingCriterionCallable


class SVCoefficient(Protocol):
    """A coefficient for the computation of a :class:`SemiValue`."""

    def __call__(self, n: int, k: int) -> float:
        """
        :param n: Number of data points
        :param k: Size of the subset for which the coefficient is being computed
        """
        ...


class SemiValue(Protocol):
    def __call__(
        self, u: Utility, stop: StoppingCriterionCallable, *args, **kwargs
    ) -> ValuationResult:
        ...


def _semivalues(
    u: Utility,
    sampler: Sampler,
    coefficient: SVCoefficient,
    stop: StoppingCriterion,
    *,
    progress: bool = False,
    job_id: int = 1,
) -> ValuationResult:
    r"""Helper function used for all computations of semi-values.

    The exact semi-value is given by:

    $$\v_\text{semi}(i) = \frac{1}{n}
      \sum_{i=1}^n w(k) \sum_{S \subset D_{-i}^{(k)}} [U(S_{+i})-U(S)]$$


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
        marginal = (
            (u({idx}.union(s)) - u(s)) * coefficient(n, len(s)) * sampler.weight(s)
        )
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


@lru_cache
def combinatorial_coefficient(n: int, k: int) -> float:
    return 1 / math.comb(n - 1, k)


@lru_cache
def banzhaf_coefficient(n: int, _: int) -> float:
    return n / 2 ** (n - 1)


def permutation_coefficient(_: int, __: int) -> float:
    return 1.0


def beta_coefficient(alpha: float, beta: float) -> SVCoefficient:

    # Leave this here in case we want to avoid depending on scipy
    # n = n or 1024
    # @lru_cache(maxsize=int(n * (n - 1) / 2))
    # def B(a: int, b: int) -> float:
    #     """Γ(a) * Γ(b) / Γ(a+b)"""
    #     ii = np.arange(1, b)
    #     return (ii / (a + ii)).prod() / a

    B = sp.special.beta
    const = B(alpha, beta)

    def beta_coefficient_w(n: int, k: int) -> float:
        """Beta coefficient"""
        j = k + 1
        w = n * B(j + beta - 1, n - j + alpha) / const
        # return math.comb(n - 1, j - 1) * w
        return w

    return beta_coefficient_w


def shapley(u: Utility, criterion: StoppingCriterion):
    sampler = UniformSampler(u.data.indices)
    return _semivalues(u, sampler, combinatorial_coefficient, criterion)


def permutation_shapley(u: Utility, criterion: StoppingCriterion):
    sampler = PermutationSampler(u.data.indices)
    return _semivalues(u, sampler, permutation_coefficient, criterion)


def beta_shapley(
    u: Utility, criterion: StoppingCriterion, alpha: float, beta: float
) -> ValuationResult:
    """Implements the Beta Shapley semi-value as introduced in
    :footcite:t:`kwon_beta_2022`.

    .. rubric:: References

    .. footbibliography::

    """
    sampler = PermutationSampler(u.data.indices)
    return _semivalues(u, sampler, beta_coefficient(alpha, beta), criterion)


def beta_shapley_paper(
    u: Utility, criterion: StoppingCriterion, alpha: float, beta: float
) -> ValuationResult:
    sampler = UniformSampler(u.data.indices)
    return _semivalues(u, sampler, beta_coefficient(alpha, beta), criterion)


def banzhaf_index(u: Utility, criterion: StoppingCriterion):
    """Implements the Banzhaf index semi-value as introduced in
    :footcite:t:`wang_data_2022`.

    .. rubric:: References

    .. footbibliography::

    """
    sampler = PermutationSampler(u.data.indices)
    return _semivalues(u, sampler, banzhaf_coefficient, criterion)


def owen_shapley(u: Utility, criterion: StoppingCriterion, max_q: int):
    sampler = OwenSampler(
        u.data.indices, method=OwenSampler.Algorithm.Standard, n_steps=200
    )
    raise NotImplementedError("to do")
