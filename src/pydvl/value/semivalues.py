r"""
This module provides the core functionality for the computation of generic
semi-values. A **semi-value** is any valuation function with the form:

$$v_\text{semi}(i) = \sum_{i=1}^n w(k)
\sum_{S \subset D_{-i}^{(k)}} [U(S_{+i})-U(S)],$$

where the coefficients $w(k)$ satisfy the property:

$$\sum_{k=1}^n w(k) = 1.$$

As such, the computation of a semi-value requires two components:

1. A **subset sampler** that generates subsets of the set $D$ of interest.
2. A **coefficient** $w(k)$ that assigns a weight to each subset size $k$.

Samplers can be found in :mod:`pydvl.value.sampler`, and can be classified into
two categories: powerset samplers and (one) permutation sampler. Powerset
samplers generate subsets of $D_{-i}$, while the permutation sampler generates
permutations of $D$. The former conform to the above definition of semi-values,
while the latter reformulates it as:

$$
v_u(x_i) = \frac{1}{n!} \sum_{\sigma \in \Pi(n)}
\tilde{w}( | \sigma_{:i} | )[u(\sigma_{:i} \cup \{i\}) − u(\sigma_{:i})],
$$

where $\sigma_{:i}$ denotes the set of indices in permutation sigma before the
position where $i$ appears (see :ref:`data valuation` for details), and
$\tilde{w}(k) = n \choose{n-1}{k} w(k)$ is the weight correction due to the
reformulation.


There are several pre-defined coefficients, including the Shapley value
of :footcite:t:`ghorbani_data_2019`, the Banzhaf index of
:footcite:t:`wang_data_2022`, and the Beta coefficient of
:footcite:t:`kwon_beta_2022`.

.. note::
   For implementation consistency, we slightly depart from the common definition
   of semi-values, which includes a factor $1/n$ in the sum over subsets.
   Instead, we subsume this factor into the coefficient $w(k)$.

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

__all__ = [
    "banzhaf_coefficient",
    "beta_coefficient",
    "shapley_coefficient",
    "semivalues",
    "SemiValueMode",
]


class SVCoefficient(Protocol):
    """A coefficient for the computation of semi-values."""

    __name__: str

    def __call__(self, n: int, k: int) -> float:
        """Computes the coefficient for a given subset size.

        :param n: Total number of elements in the set.
        :param k: Size of the subset for which the coefficient is being computed
        """
        ...


def _semivalues(
    sampler: PowersetSampler,
    u: Utility,
    coefficient: SVCoefficient,
    done: StoppingCriterion,
    *,
    progress: bool = False,
    job_id: int = 1,
) -> ValuationResult:
    r"""Serial computation of semi-values. This is a helper function for
    :func:`semivalues`.

    :param sampler: The subset sampler to use for utility computations.
    :param u: Utility object with model, data, and scoring function.
    :param coefficient: The semivalue coefficient
    :param done: Stopping criterion.
    :param progress: Whether to display progress bars for each job.
    :param job_id: id to use for reporting progress.
    :return: Object with the results.
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
    """
    Computes semi-values for a given utility function and subset sampler.

    :param sampler: The subset sampler to use for utility computations.
    :param u: Utility object with model, data, and scoring function.
    :param coefficient: The semi-value coefficient
    :param done: Stopping criterion.
    :param n_jobs: Number of parallel jobs to use.
    :param config: Object configuring parallel computation, with cluster
        address, number of cpus, etc.
    :param progress: Whether to display progress bars for each job.
    :return: Object with the results.

    """
    map_reduce_job: MapReduceJob[PowersetSampler, ValuationResult] = MapReduceJob(
        sampler,
        map_func=_semivalues,
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
        w = B(j + beta - 1, n - j + alpha) / const
        # return math.comb(n - 1, j - 1) * w * n
        return float(w)

    return cast(SVCoefficient, beta_coefficient_w)


class SemiValueMode(str, Enum):
    Shapley = "shapley"
    BetaShapley = "beta_shapley"
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
