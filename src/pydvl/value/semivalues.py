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
import operator
from enum import Enum
from functools import reduce
from itertools import takewhile
from typing import Protocol, Type, cast

import scipy as sp
from tqdm import tqdm

from pydvl.utils import MapReduceJob, ParallelConfig, Utility
from pydvl.value import ValuationResult
from pydvl.value.sampler import PermutationSampler, PowersetSampler
from pydvl.value.stopping import MaxUpdates, StoppingCriterion


__all__ = ["serial_semivalues", "semivalues"]


class SVCoefficient(Protocol):
    """A coefficient for the computation of semi-values."""

    __name__: str

    def __call__(self, n: int, k: int) -> float:
        """Computes the coefficient for a given subset size.

        :param n: Total number of elements in the set.
        :param k: Size of the subset for which the coefficient is being computed
        """
        ...


def serial_semivalues(
    sampler: PowersetSampler,
    u: Utility,
    coefficient: SVCoefficient,
    done: StoppingCriterion,
    *,
    progress: bool = False,
    job_id: int = 1,
) -> ValuationResult:
    r"""Helper function used for all computations of semi-values.

    The exact semi-value is given by:

    $$\v_\text{semi}(i) =
      \sum_{i=1}^n w(k) \sum_{S \subset D_{-i}^{(k)}} [U(S_{+i})-U(S)]$$


    :param u: Utility object with model, data, and scoring function.
    :param sampler: The subset sampler to use for utility computations.
    :param coefficient: The semivalue coefficient
    :param done: Stopping criterion.
    :param progress: Whether to display progress bars for each job.
    :param job_id: id to use for reporting progress.
    :return: Object with valuation results.
    """
    n = len(u.data.indices)
    result = ValuationResult.empty(
        algorithm=f"semivalue-{str(sampler)}-{coefficient.__name__}",
        indices=sampler.indices,
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


def semivalues(
    sampler: PowersetSampler,
    u: Utility,
    coefficient: SVCoefficient,
    done: StoppingCriterion,
    *,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    progress: bool = False,
) -> ValuationResult:
    map_reduce_job: MapReduceJob[PowersetSampler, ValuationResult] = MapReduceJob(
        sampler,
        map_func=serial_semivalues,
        reduce_func=lambda results: reduce(operator.add, results),
        map_kwargs=dict(u=u, coefficient=coefficient, done=done, progress=progress),
        config=config,
        n_jobs=n_jobs,
    )
    return map_reduce_job()


def shapley_coefficient(n: int, k: int) -> float:
    return float(1 / math.comb(n - 1, k) / n)


def banzhaf_coefficient(n: int, k: int) -> float:
    return float(1 / 2 ** (n - 1))


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
        return float(w / n)

    return cast(SVCoefficient, beta_coefficient_w)


class SemiValueMode(str, Enum):
    Shapley = "shapley"
    BetaShapley = "beta-shapley"
    Banzhaf = "banzhaf"


def compute_semivalues(
    u: Utility,
    *,
    done: StoppingCriterion = MaxUpdates(100),
    mode: SemiValueMode = SemiValueMode.Shapley,
    sampler_t: Type[PowersetSampler] = PermutationSampler,
    n_jobs: int = 1,
    **kwargs,
) -> ValuationResult:
    """Entry point for most common semi-value computations. All are implemented
    with permutation sampling.

    For any other sampling method, use :func:`parallel_semivalues` directly.

    See :ref:`data valuation` for an overview of valuation.

    The modes supported are:

    - :attr:`SemiValueMode.Shapley`: Shapley values.
    - :attr:`SemiValueMode.BetaShapley`: Implements the Beta Shapley semi-value
        as introduced in :footcite:t:`kwon_beta_2022`. Pass additional keyword
        arguments ``alpha`` and ``beta`` to set the parameters of the Beta
        distribution (both default to 1).
    - :attr:`SemiValueMode.Banzhaf`: Implements the Banzhaf semi-value as
        introduced in :footcite:t:`wang_data_2022`.

    :param u: Utility object with model, data, and scoring function.
    :param done: Stopping criterion.
    :param mode: The semi-value mode to use. See :class:`SemiValueMode` for a
        list.
    :param sampler_t: The sampler type to use. See :mod:`pydvl.value.sampler`
        for a list.
    :param n_jobs: Number of parallel jobs to use.
    :param kwargs: Additional keyword arguments passed to
        :func:`~pydvl.value.semivalues.semivalues`.
    """
    sampler_instance = sampler_t(u.data.indices)
    if mode == SemiValueMode.Shapley:
        coefficient = shapley_coefficient
    elif mode == SemiValueMode.BetaShapley:
        alpha = kwargs.get("alpha", 1)
        beta = kwargs.get("beta", 1)
        coefficient = beta_coefficient(alpha, beta)
    elif mode == SemiValueMode.Banzhaf:
        coefficient = banzhaf_coefficient
    else:
        raise ValueError(f"Unknown mode {mode}")
    coefficient = cast(SVCoefficient, coefficient)
    return semivalues(sampler_instance, u, coefficient, done, n_jobs=n_jobs, **kwargs)
