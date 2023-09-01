r"""
This module provides the core functionality for the computation of generic
semi-values. A **semi-value** is any valuation function with the form:

$$v_\text{semi}(i) = \sum_{i=1}^n w(k)
\sum_{S \subset D_{-i}^{(k)}} [U(S_{+i})-U(S)],$$

where the coefficients $w(k)$ satisfy the property:

$$\sum_{k=1}^n w(k) = 1.$$

!!! Note
    For implementation consistency, we slightly depart from the common definition
    of semi-values, which includes a factor $1/n$ in the sum over subsets.
    Instead, we subsume this factor into the coefficient $w(k)$.

As such, the computation of a semi-value requires two components:

1. A **subset sampler** that generates subsets of the set $D$ of interest.
2. A **coefficient** $w(k)$ that assigns a weight to each subset size $k$.

Samplers can be found in [sampler][pydvl.value.sampler], and can be classified
into two categories: powerset samplers and permutation samplers. Powerset
samplers generate subsets of $D_{-i}$, while the permutation sampler generates
permutations of $D$. The former conform to the above definition of semi-values,
while the latter reformulates it as:

$$
v(i) = \frac{1}{n!} \sum_{\sigma \in \Pi(n)}
\tilde{w}( | \sigma_{:i} | )[U(\sigma_{:i} \cup \{i\}) − U(\sigma_{:i})],
$$

where $\sigma_{:i}$ denotes the set of indices in permutation sigma before the
position where $i$ appears (see [Data valuation][computing-data-values] for
details), and

$$ \tilde{w} (k) = n \binom{n - 1}{k} w (k) $$

is the weight correction due to the reformulation.

!!! Warning
    Both [PermutationSampler][pydvl.value.sampler.PermutationSampler] and
    [DeterministicPermutationSampler][pydvl.value.sampler.DeterministicPermutationSampler]
    require caching to be enabled or computation will be doubled wrt. a 'direct'
    implementation of permutation MC.

There are several pre-defined coefficients, including the Shapley value of
(Ghorbani and Zou, 2019)[^1], the Banzhaf index of (Wang and Jia)[^3], and the Beta
coefficient of (Kwon and Zou, 2022)[^2]. For each of these methods, there is a
convenience wrapper function. Respectively, these are:
[compute_shapley_semivalues][pydvl.value.semivalues.compute_shapley_semivalues],
[compute_banzhaf_semivalues][pydvl.value.semivalues.compute_banzhaf_semivalues],
and [compute_beta_shapley_semivalues][pydvl.value.semivalues.compute_beta_shapley_semivalues].
instead.


# References:

[^1]: <a name="ghorbani_data_2019"></a>Ghorbani, A., Zou, J., 2019.
    [Data Shapley: Equitable Valuation of Data for Machine Learning](http://proceedings.mlr.press/v97/ghorbani19c.html).
    In: Proceedings of the 36th International Conference on Machine Learning, PMLR, pp. 2242–2251.

[^2]: <a name="kwon_beta_2022"></a>Kwon, Y. and Zou, J., 2022.
    [Beta Shapley: A Unified and Noise-reduced Data Valuation Framework for Machine Learning](http://arxiv.org/abs/2110.14049).
    In: Proceedings of the 25th International Conference on Artificial Intelligence and Statistics (AISTATS) 2022, Vol. 151. PMLR, Valencia, Spain.

[^3]: <a name="wang_data_2022"></a>Wang, J.T. and Jia, R., 2022.
    [Data Banzhaf: A Robust Data Valuation Framework for Machine Learning](http://arxiv.org/abs/2205.15466).
    ArXiv preprint arXiv:2205.15466.
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
    "compute_generic_semivalues",
    "compute_semivalues",
    "SemiValueMode",
]

log = logging.getLogger(__name__)


class SVCoefficient(Protocol):
    """A coefficient for the computation of semi-values."""

    def __call__(self, n: int, k: int) -> float:
        """Computes the coefficient for a given subset size.

        Args:
            n: Total number of elements in the set.
            k: Size of the subset for which the coefficient is being computed
        """
        ...


IndexT = TypeVar("IndexT", bound=np.generic)
MarginalT = Tuple[IndexT, float]


def _marginal(u: Utility, coefficient: SVCoefficient, sample: SampleT) -> MarginalT:
    """Computation of marginal utility. This is a helper function for
    [compute_generic_semivalues][pydvl.value.semivalues.compute_generic_semivalues].

    Args:
        u: Utility object with model, data, and scoring function.
        coefficient: The semivalue coefficient and sampler weight
        sample: A tuple of index and subset of indices to compute a marginal
            utility.

    Returns:
        Tuple with index and its marginal utility.
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
def compute_generic_semivalues(
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

    Args:
        sampler: The subset sampler to use for utility computations.
        u: Utility object with model, data, and scoring function.
        coefficient: The semi-value coefficient
        done: Stopping criterion.
        n_jobs: Number of parallel jobs to use.
        config: Object configuring parallel computation, with cluster
            address, number of cpus, etc.
        progress: Whether to display a progress bar.

    Returns:
        Object with the results.
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

    This is a convenience wrapper for
    [compute_generic_semivalues][pydvl.value.semivalues.compute_generic_semivalues]
    with the Shapley coefficient. Use
    [compute_shapley_values][pydvl.value.shapley.common.compute_shapley_values]
    for a more flexible interface and additional methods, including TMCS.

    Args:
        u: Utility object with model, data, and scoring function.
        done: Stopping criterion.
        sampler_t: The sampler type to use. See :mod:`pydvl.value.sampler`
            for a list.
        n_jobs: Number of parallel jobs to use.
        config: Object configuring parallel computation, with cluster
            address, number of cpus, etc.
        seed: Either an instance of a numpy random number generator or a seed for it.
        progress: Whether to display a progress bar.

    Returns:
        Object with the results.
    """
    return compute_generic_semivalues(
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

    This is a convenience wrapper for
    [compute_generic_semivalues][pydvl.value.semivalues.compute_generic_semivalues]
    with the Banzhaf coefficient.

    Args:
        u: Utility object with model, data, and scoring function.
        done: Stopping criterion.
        sampler_t: The sampler type to use. See :mod:`pydvl.value.sampler` for a
            list.
        n_jobs: Number of parallel jobs to use.
        seed: Either an instance of a numpy random number generator or a seed for it.
        config: Object configuring parallel computation, with cluster address,
            number of cpus, etc.
        progress: Whether to display a progress bar.

    Returns:
        Object with the results.
    """
    return compute_generic_semivalues(
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

    This is a convenience wrapper for
    [compute_generic_semivalues][pydvl.value.semivalues.compute_generic_semivalues]
    with the Beta Shapley coefficient.

    Args:
        u: Utility object with model, data, and scoring function.
        alpha: Alpha parameter of the Beta distribution.
        beta: Beta parameter of the Beta distribution.
        done: Stopping criterion.
        sampler_t: The sampler type to use. See :mod:`pydvl.value.sampler` for a list.
        n_jobs: Number of parallel jobs to use.
        seed: Either an instance of a numpy random number generator or a seed for it.
        config: Object configuring parallel computation, with cluster address, number of cpus, etc.
        progress: Whether to display a progress bar.

    Returns:
        Object with the results.
    """
    return compute_generic_semivalues(
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
    """Enumeration of semi-value modes.

    !!! warning "Deprecation notice"
        This enum and the associated methods are deprecated and will be removed
        in 0.8.0.
    """

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

    !!! warning "Deprecation warning"
        This method is deprecated and will be replaced in 0.8.0 by the more
        general implementation of
        [compute_generic_semivalues][pydvl.value.semivalues.compute_generic_semivalues].
        Use
        [compute_shapley_semivalues][pydvl.value.semivalues.compute_shapley_semivalues],
        [compute_banzhaf_semivalues][pydvl.value.semivalues.compute_banzhaf_semivalues],
        or
        [compute_beta_shapley_semivalues][pydvl.value.semivalues.compute_beta_shapley_semivalues]
        instead.

    The modes supported with this interface are the following. For greater
    flexibility use
    [compute_generic_semivalues][pydvl.value.semivalues.compute_generic_semivalues]
    directly.

    - [SemiValueMode.Shapley][pydvl.value.semivalues.SemiValueMode]:
      Shapley values.
    - [SemiValueMode.BetaShapley][pydvl.value.semivalues.SemiValueMode.BetaShapley]:
      Implements the Beta Shapley semi-value as introduced in
      (Kwon and Zou, 2022)<sup><a href="#kwon_beta_2022">1</a></sup>.
      Pass additional keyword arguments `alpha` and `beta` to set the
      parameters of the Beta distribution (both default to 1).
    - [SemiValueMode.Banzhaf][SemiValueMode.Banzhaf]: Implements the Banzhaf
      semi-value as introduced in (Wang and Jia, 2022)<sup><a href="#wang_data_2022">1</a></sup>.

    See [[data-valuation]] for an overview of valuation.
    - [SemiValueMode.Banzhaf][pydvl.value.semivalues.SemiValueMode]: Implements
      the Banzhaf semi-value as introduced in [@wang_data_2022].

    Args:
        u: Utility object with model, data, and scoring function.
        done: Stopping criterion.
        mode: The semi-value mode to use. See
            [SemiValueMode][pydvl.value.semivalues.SemiValueMode] for a list.
        sampler_t: The sampler type to use. See [sampler][pydvl.value.sampler]
            for a list.
        n_jobs: Number of parallel jobs to use.
        seed: Either an instance of a numpy random number generator or a seed for it.
        kwargs: Additional keyword arguments passed to
            [compute_generic_semivalues][pydvl.value.semivalues.compute_generic_semivalues].

    Returns:
        Object with the results.
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
    return compute_generic_semivalues(
        sampler_t(u.data.indices, seed=seed),
        u,
        coefficient,
        done,
        n_jobs=n_jobs,
        **kwargs,
    )
