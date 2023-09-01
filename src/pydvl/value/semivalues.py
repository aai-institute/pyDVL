r"""
This module provides the core functionality for the computation of generic
semi-values. A **semi-value** is any valuation function with the form:

$$v_\text{semi}(i) = \sum_{i=1}^n w(k)
\sum_{S \subset D_{-i}^{(k)}} [U(S_{+i})-U(S)],$$

where the coefficients $w(k)$ satisfy the property:

$$\sum_{k=1}^n w(k) = 1.$$

.. note::
   For implementation consistency, we slightly depart from the common definition
   of semi-values, which includes a factor $1/n$ in the sum over subsets.
   Instead, we subsume this factor into the coefficient $w(k)$.

As such, the computation of a semi-value requires two components:

1. A **subset sampler** that generates subsets of the set $D$ of interest.
2. A **coefficient** $w(k)$ that assigns a weight to each subset size $k$.

Samplers can be found in :mod:`pydvl.value.sampler`, and can be classified into
two categories: powerset samplers and permutation samplers. Powerset
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

.. warning::
   Both :class:`~pydvl.value.sampler.PermutationSampler` and
   :class:`~pydvl.value.sampler.DeterministicPermutationSampler`
   require caching to be enabled or computation will be doubled wrt. a 'direct'
   implementation of permutation MC.

There are several pre-defined coefficients, including the Shapley value
of :footcite:t:`ghorbani_data_2019`, the Banzhaf index of
:footcite:t:`wang_data_2022`, and the Beta coefficient of
:footcite:t:`kwon_beta_2022`. For each of these methods, there is a convenience
wrapper function. Respectively, these are: :func:`compute_shapley_semivalues`,
:func:`compute_banzhaf_semivalues`, and :func:`compute_beta_shapley_semivalues`.

"""
from __future__ import annotations

import logging
import math
from enum import Enum
from typing import Optional, Protocol, Tuple, Type, TypeVar, cast

import numpy as np
import scipy as sp
from deprecate import deprecated
from tqdm import tqdm

from pydvl.utils import ParallelConfig, Utility
from pydvl.utils.parallel.backlog import Backlog
from pydvl.utils.types import Seed
from pydvl.value import ValuationResult
from pydvl.value.sampler import (
    PermutationSampler,
    PowersetSampler,
    SampleT,
    StochasticSampler,
)
from pydvl.value.stopping import MaxUpdates, StoppingCriterion

__all__ = [
    "compute_banzhaf_semivalues",
    "compute_beta_shapley_semivalues",
    "compute_shapley_semivalues",
    "beta_coefficient",
    "banzhaf_coefficient",
    "shapley_coefficient",
    "semivalues",
    "compute_semivalues",
    "SemiValueMode",
]

log = logging.getLogger(__name__)


class SVCoefficient(Protocol):
    """A coefficient for the computation of semi-values."""

    def __call__(self, n: int, k: int) -> float:
        """Computes the coefficient for a given subset size.

        :param n: Total number of elements in the set.
        :param k: Size of the subset for which the coefficient is being computed
        """
        ...


IndexT = TypeVar("IndexT", bound=np.generic)
MarginalT = Tuple[IndexT, float]


def _marginal(u: Utility, coefficient: SVCoefficient, sample: SampleT) -> MarginalT:
    """Computation of marginal utility. This is a helper function for
    :func:`semivalues`.

    :param u: Utility object with model, data, and scoring function.
    :param coefficient: The semi-value coefficient and sampler weight
    :param sample: A tuple of index and subset of indices to compute a marginal
        utility.

    :return: tuple with index and its marginal utility.
    """
    n = len(u.data)
    idx, s = sample
    marginal = (u({idx}.union(s)) - u(s)) * coefficient(n, len(s))
    return idx, marginal


# @deprecated(
#     target=compute_semivalues,  # TODO: rename this to compute_semivalues
#     deprecated_in="0.8.0",
#     remove_in="0.9.0",
# )
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
    """Computes semi-values for a given utility function and subset sampler.

    :param sampler: The subset sampler to use for utility computations.
    :param u: Utility object with model, data, and scoring function.
    :param coefficient: The semi-value coefficient
    :param done: Stopping criterion.
    :param n_jobs: Number of parallel jobs to use.
    :param config: Object configuring parallel computation, with cluster
        address, number of cpus, etc.
    :param progress: Whether to display a progress bar.

    :return: Object with the results.
    """
    from concurrent.futures import FIRST_COMPLETED, Future, wait

    from pydvl.utils import effective_n_jobs, init_executor, init_parallel_backend

    if isinstance(sampler, PermutationSampler) and not u.enable_cache:
        log.warning(
            "PermutationSampler requires caching to be enabled or computation "
            "will be doubled wrt. a 'direct' implementation of permutation MC"
        )

    result = ValuationResult.zeros(
        algorithm=f"semivalue-{str(sampler)}-{coefficient.__name__}",  # type: ignore
        indices=u.data.indices,
        data_names=u.data.data_names,
    )

    parallel_backend = init_parallel_backend(config)
    u = parallel_backend.put(u)
    correction = parallel_backend.put(
        lambda n, k: coefficient(n, k) * sampler.weight(n, k)
    )

    max_workers = effective_n_jobs(n_jobs, config)
    n_submitted_jobs = 2 * max_workers  # number of jobs in the queue

    sampler_it = iter(sampler)
    pbar = tqdm(disable=not progress, total=100, unit="%")
    backlog = Backlog[Tuple[int, float]]()

    with init_executor(
        max_workers=max_workers, config=config, cancel_futures=True
    ) as executor:
        pending: set[Future] = set()
        while True:
            pbar.n = 100 * done.completion()
            pbar.refresh()

            completed, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in completed:
                backlog.add(future.result())

            for idx, marginal in backlog.get():
                result.update(idx, marginal)
                if done(result):
                    return result

            # Ensure that we always have n_submitted_jobs running
            try:
                for _ in range(n_submitted_jobs - len(pending)):
                    pending.add(
                        executor.submit(
                            backlog.wrap(_marginal),
                            u=u,
                            coefficient=correction,
                            sample=next(sampler_it),
                        )
                    )
            except StopIteration:
                return result


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


def compute_shapley_semivalues(
    u: Utility,
    *,
    done: StoppingCriterion = MaxUpdates(100),
    sampler_t: Type[StochasticSampler] = PermutationSampler,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    progress: bool = False,
    seed: Optional[Seed] = None,
) -> ValuationResult:
    """Computes Shapley values for a given utility function.

    This is a convenience wrapper for :func:`semivalues` with the Shapley
    coefficient. Use :func:`~pydvl.value.shapley.common.compute_shapley_values`
    for a more flexible interface and additional methods, including Truncated
    Monte Carlo.

    :param u: Utility object with model, data, and scoring function.
    :param done: Stopping criterion.
    :param sampler_t: The sampler type to use. See :mod:`pydvl.value.sampler`
        for a list.
    :param n_jobs: Number of parallel jobs to use.
    :param seed: Either an instance of a numpy random number generator or a seed for it.
    :param config: Object configuring parallel computation, with cluster
        address, number of cpus, etc.
    :param progress: Whether to display a progress bar.

    :return: Object with the results.
    """
    return semivalues(
        sampler_t(u.data.indices, seed=seed),
        u,
        shapley_coefficient,
        done,
        n_jobs=n_jobs,
        config=config,
        progress=progress,
    )


def compute_banzhaf_semivalues(
    u: Utility,
    *,
    done: StoppingCriterion = MaxUpdates(100),
    sampler_t: Type[StochasticSampler] = PermutationSampler,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    progress: bool = False,
    seed: Optional[Seed] = None,
) -> ValuationResult:
    """Computes Banzhaf values for a given utility function.

    This is a convenience wrapper for :func:`semivalues` with the Banzhaf
    coefficient.

    :param u: Utility object with model, data, and scoring function.
    :param done: Stopping criterion.
    :param sampler_t: The sampler type to use. See :mod:`pydvl.value.sampler`
        for a list.
    :param n_jobs: Number of parallel jobs to use.
    :param seed: Either an instance of a numpy random number generator or a seed for it.
    :param config: Object configuring parallel computation, with cluster
        address, number of cpus, etc.
    :param progress: Whether to display a progress bar.

    :return: Object with the results.
    """
    return semivalues(
        sampler_t(u.data.indices, seed=seed),
        u,
        banzhaf_coefficient,
        done,
        n_jobs=n_jobs,
        config=config,
        progress=progress,
    )


def compute_beta_shapley_semivalues(
    u: Utility,
    *,
    alpha: float = 1,
    beta: float = 1,
    done: StoppingCriterion = MaxUpdates(100),
    sampler_t: Type[StochasticSampler] = PermutationSampler,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    progress: bool = False,
    seed: Optional[Seed] = None,
) -> ValuationResult:
    """Computes Beta Shapley values for a given utility function.

    This is a convenience wrapper for :func:`semivalues` with the Beta Shapley
    coefficient.

    :param u: Utility object with model, data, and scoring function.
    :param alpha: Alpha parameter of the Beta distribution.
    :param beta: Beta parameter of the Beta distribution.
    :param done: Stopping criterion.
    :param sampler_t: The sampler type to use. See :mod:`pydvl.value.sampler`
        for a list.
    :param n_jobs: Number of parallel jobs to use.
    :param seed: Either an instance of a numpy random number generator or a seed for it.
    :param config: Object configuring parallel computation, with cluster
        address, number of cpus, etc.
    :param progress: Whether to display a progress bar.

    :return: Object with the results.
    """
    return semivalues(
        sampler_t(u.data.indices, seed=seed),
        u,
        beta_coefficient(alpha, beta),
        done,
        n_jobs=n_jobs,
        config=config,
        progress=progress,
    )


@deprecated(
    target=True,
    deprecated_in="0.7.0",
    remove_in="0.8.0",
)
class SemiValueMode(str, Enum):
    Shapley = "shapley"
    BetaShapley = "beta_shapley"
    Banzhaf = "banzhaf"


@deprecated(target=True, deprecated_in="0.7.0", remove_in="0.8.0")
def compute_semivalues(
    u: Utility,
    *,
    done: StoppingCriterion = MaxUpdates(100),
    mode: SemiValueMode = SemiValueMode.Shapley,
    sampler_t: Type[StochasticSampler] = PermutationSampler,
    n_jobs: int = 1,
    seed: Optional[Seed] = None,
    **kwargs,
) -> ValuationResult:
    """Convenience entry point for most common semi-value computations.

    .. warning::
       This method is deprecated and will be removed in 0.8.0. Use
       :func:`~pydvl.value.semivalues.compute_shapley_semivalues`,
       :func:`~pydvl.value.semivalues.compute_banzhaf_semivalues`, or
       :func:`~pydvl.value.semivalues.compute_beta_shapley_semivalues` instead

    The modes supported with this interface are the following. For greater
    flexibility use :func:`~pydvl.value.semivalues.semivalues` directly.

    - :attr:`SemiValueMode.Shapley`: Shapley values.
    - :attr:`SemiValueMode.BetaShapley`: Implements the Beta Shapley semi-value
      as introduced in :footcite:t:`kwon_beta_2022`. Pass additional keyword
      arguments ``alpha`` and ``beta`` to set the parameters of the Beta
      distribution (both default to 1).
    - :attr:`SemiValueMode.Banzhaf`: Implements the Banzhaf semi-value as
      introduced in :footcite:t:`wang_data_2022`.

    See :ref:`data valuation` for an overview of data valuation.

    :param u: Utility object with model, data, and scoring function.
    :param done: Stopping criterion.
    :param mode: The semi-value mode to use. See :class:`SemiValueMode` for a
        list.
    :param sampler_t: The sampler type to use. See :mod:`pydvl.value.sampler`
        for a list.
    :param n_jobs: Number of parallel jobs to use.
    :param seed: Either an instance of a numpy random number generator or a seed for it.
    :param kwargs: Additional keyword arguments passed to
        :func:`~pydvl.value.semivalues.semivalues`.

    :return: Object with the results.
    """
    if mode == SemiValueMode.Shapley:
        coefficient = shapley_coefficient
    elif mode == SemiValueMode.BetaShapley:
        alpha = kwargs.pop("alpha", 1)
        beta = kwargs.pop("beta", 1)
        coefficient = beta_coefficient(alpha, beta)
    elif mode == SemiValueMode.Banzhaf:
        coefficient = banzhaf_coefficient
    else:
        raise ValueError(f"Unknown mode {mode}")
    coefficient = cast(SVCoefficient, coefficient)
    return semivalues(
        sampler_t(u.data.indices, seed=seed),
        u,
        coefficient,
        done,
        n_jobs=n_jobs,
        **kwargs,
    )
