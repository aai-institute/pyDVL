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
from itertools import takewhile
from typing import Protocol

import scipy as sp
from tqdm import tqdm

from pydvl.utils import Utility
from pydvl.value import ValuationResult
from pydvl.value.sampler import (
    OwenSampler,
    PermutationSampler,
    PowersetSampler,
    UniformSampler,
)
from pydvl.value.stopping import StoppingCriterion, StoppingCriterionCallable


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
    sampler: PowersetSampler,
    coefficient: SVCoefficient,
    done: StoppingCriterion,
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
    :param done: Stopping criterion.
    :param progress: Whether to display progress bars for each job.
    :param job_id: id to use for reporting progress.
    :return: Object with valuation results.
    """
    n = len(u.data)
    result = ValuationResult.empty(
        algorithm=f"semivalue-{str(sampler)}-{str(coefficient)}", indices=u.data.indices
    )

    samples = takewhile(lambda _: not done(result), sampler)
    pbar = tqdm(disable=not progress, position=job_id, total=100, unit="%")
    for idx, s in samples:
        pbar.n = 100 * done.completion()
        pbar.refresh()
        marginal = (
            (u({idx}.union(s)) - u(s)) * coefficient(n, len(s)) * sampler.weight(s)
        )
        result.update(idx, marginal)

    return result


def combinatorial_coefficient(n: int, k: int) -> float:
    return 1 / math.comb(n - 1, k)


def banzhaf_coefficient(n: int, k: int) -> float:
    return n / 2 ** (n - 1)


def permutation_coefficient(n: int, k: int) -> float:
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
