r"""
This module provides the core functionality for the computation of generic
semi-values. A **semi-value** is any valuation function with the form:

$$v_\text{semi}(i) = \sum_{i=1}^n w(k)
\sum_{S \subset D_{-i}^{(k)}} [U(S_{+i})-U(S)],$$

where the coefficients $w(k)$ satisfy the property:

$$\sum_{k=1}^n w(k) = 1.$$

??? Note
    For implementation consistency, we slightly depart from the common definition
    of semi-values, which includes a factor $1/n$ in the sum over subsets.
    Instead, we subsume this factor into the coefficient $w(k)$.

## Main components

The computation of a semi-value requires two components:

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

## Computing semi-values

Samplers and coefficients can be arbitrarily mixed by means of the main entry
point of this module,
[compute_generic_semivalues][pydvl.value.semivalues.compute_generic_semivalues].
There are several pre-defined coefficients, including the Shapley value of
(Ghorbani and Zou, 2019)[^1], the Banzhaf index of (Wang and Jia)[^3], and the Beta
coefficient of (Kwon and Zou, 2022)[^2]. For each of these methods, there is a
convenience wrapper function. Respectively, these are:
[compute_shapley_semivalues][pydvl.value.semivalues.compute_shapley_semivalues],
[compute_banzhaf_semivalues][pydvl.value.semivalues.compute_banzhaf_semivalues],
and [compute_beta_shapley_semivalues][pydvl.value.semivalues.compute_beta_shapley_semivalues].
instead.

!!! tip "Parallelization and batching"
    In order to ensure reproducibility and fine-grained control of
    parallelization, samples are generated in the main process and then
    distributed to worker processes for evaluation. For small sample sizes, this
    can lead to a significant overhead. To avoid this, we temporarily provide an
    additional argument `batch_size` to all methods which can improve
    performance with small models up to an order of magnitude. Note that this
    argument will be removed before version 1.0 in favour of a more general
    solution.


## References

[^1]: <a name="ghorbani_data_2019"></a>Ghorbani, A., Zou, J., 2019.
    [Data Shapley: Equitable Valuation of Data for Machine Learning](https://proceedings.mlr.press/v97/ghorbani19c.html).
    In: Proceedings of the 36th International Conference on Machine Learning, PMLR, pp. 2242–2251.

[^2]: <a name="kwon_beta_2022"></a>Kwon, Y. and Zou, J., 2022.
    [Beta Shapley: A Unified and Noise-reduced Data Valuation Framework for Machine Learning](https://arxiv.org/abs/2110.14049).
    In: Proceedings of the 25th International Conference on Artificial Intelligence and Statistics (AISTATS) 2022, Vol. 151. PMLR, Valencia, Spain.

[^3]: <a name="wang_data_2023"></a>Wang, J.T. and Jia, R., 2023.
    [Data Banzhaf: A Robust Data Valuation Framework for Machine Learning](https://proceedings.mlr.press/v206/wang23e.html).
    In: Proceedings of The 26th International Conference on Artificial Intelligence and Statistics, pp. 6388-6421.
"""

from __future__ import annotations

import logging
import math
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, Future, wait
from enum import Enum
from itertools import islice
from typing import Any, Iterable, List, Optional, Protocol, Tuple, Type, cast

import numpy as np
import scipy as sp
from deprecate import deprecated
from tqdm import tqdm

from pydvl.parallel import ParallelBackend, _maybe_init_parallel_backend
from pydvl.parallel.config import ParallelConfig
from pydvl.utils import Utility
from pydvl.utils.types import IndexT, Seed
from pydvl.value import ValuationResult
from pydvl.value.sampler import (
    MSRSampler,
    PermutationSampler,
    PowersetSampler,
    SampleT,
    StochasticSampler,
)
from pydvl.value.stopping import StoppingCriterion

__all__ = [
    "compute_banzhaf_semivalues",
    "compute_msr_banzhaf_semivalues",
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
    """The protocol that coefficients for the computation of semi-values must
    fulfill."""

    def __call__(self, n: int, k: int) -> float:
        """Computes the coefficient for a given subset size.

        Args:
            n: Total number of elements in the set.
            k: Size of the subset for which the coefficient is being computed
        """
        ...


MarginalT = Tuple[IndexT, float]


class MarginalFunction(ABC):
    @abstractmethod
    def __call__(
        self, u: Utility, coefficient: SVCoefficient, samples: Iterable[SampleT]
    ) -> Tuple[MarginalT, ...]:
        raise NotImplementedError


class DefaultMarginal(MarginalFunction):
    def __call__(
        self, u: Utility, coefficient: SVCoefficient, samples: Iterable[SampleT]
    ) -> Tuple[MarginalT, ...]:
        """Computation of marginal utility. This is a helper function for
        [compute_generic_semivalues][pydvl.value.semivalues.compute_generic_semivalues].

        Args:
            u: Utility object with model, data, and scoring function.
            coefficient: The semivalue coefficient and sampler weight
            samples: A collection of samples. Each sample is a tuple of index and subset of
                indices to compute a marginal utility.

        Returns:
            A collection of marginals. Each marginal is a tuple with index and its marginal
            utility.
        """
        n = len(u.data)
        marginals: List[MarginalT] = []
        for idx, s in samples:
            marginal = (u({idx}.union(s)) - u(s)) * coefficient(n, len(s))
            marginals.append((idx, marginal))
        return tuple(marginals)


class RawUtility(MarginalFunction):
    def __call__(
        self, u: Utility, coefficient: SVCoefficient, samples: Iterable[SampleT]
    ) -> Tuple[MarginalT, ...]:
        """Computation of raw utility without marginalization. This is a helper function for
        [compute_generic_semivalues][pydvl.value.semivalues.compute_generic_semivalues].

        Args:
            u: Utility object with model, data, and scoring function.
            coefficient: The semivalue coefficient and sampler weight
            samples: A collection of samples. Each sample is a tuple of index and subset of
                indices to compute a marginal utility.

        Returns:
            A collection of marginals. Each marginal is a tuple with index and its raw utility.
        """
        marginals: List[MarginalT] = []
        for idx, s in samples:
            marginals.append((s, u(s)))
        return tuple(marginals)


class FutureProcessor:
    """
    The FutureProcessor class used to process the results of the parallel marginal evaluations.

    The marginals are evaluated in parallel by `n_jobs` threads, but some algorithms require a central
    method to postprocess the marginal results. This can be achieved through the future processor.
    This base class does not perform any postprocessing, it is a noop used in most data valuation algorithms.
    """

    def __call__(self, future_result: Any) -> Any:
        return future_result


class MSRFutureProcessor(FutureProcessor):
    """
    This FutureProcessor processes the raw marginals in a way that MSR sampling requires.

    MSR sampling evaluates the utility once, and then updates all data semivalues based on this one evaluation.
    In order to do this, the RawUtility value needs to be postprocessed through this class.
    For more details on MSR, please refer to the paper (Wang et. al.)<sup><a href="wang_data_2023">3</a></sup>.
    This processor keeps track of the current values and computes marginals for all data points, so that
    the values in the ValuationResult can be updated properly down the line.
    """

    def __init__(self, u: Utility):
        self.n = len(u.data)
        self.all_indices = u.data.indices.copy()
        self.point_in_subset = np.zeros((self.n,))
        self.positive_sums = np.zeros((self.n,))
        self.negative_sums = np.zeros((self.n,))
        self.total_evaluations = 0

    def compute_values(self) -> np.ndarray:
        points_not_in_subset = self.total_evaluations - self.point_in_subset
        feasibility_map = np.logical_and(
            self.point_in_subset > 0, points_not_in_subset > 0
        )
        values: np.ndarray = (
            np.divide(
                1,
                self.point_in_subset,
                out=np.zeros_like(self.point_in_subset),
                where=feasibility_map,
            )
            * self.positive_sums
            - np.divide(
                1,
                points_not_in_subset,
                out=np.zeros_like(self.point_in_subset),
                where=feasibility_map,
            )
            * self.negative_sums
        )
        return values

    def __call__(
        self, future_result: List[Tuple[List[IndexT], float]]
    ) -> List[List[MarginalT]]:
        """Computation of marginal utility using Maximum Sample Reuse.

        This processor requires the Marginal Function to be set to RawUtility.
        Then, this processor computes marginals based on the utility value and the index set provided.

        The final formula that gives the Banzhaf semivalue using MSR is:
        $$\\hat{\\phi}_{MSR}(i) = \frac{1}{|\\mathbf{S}_{\ni i}|} \\sum_{S \\in \\mathbf{S}_{\ni i}} U(S)
        - \frac{1}{|\\mathbf{S}_{\not{\ni} i}|} \\sum_{S \\in \\mathbf{S}_{\not{\ni} i}} U(S)$$

        Args:
            future_result: Result of the parallel computing jobs comprised of
                a list of indices that were used to evaluate the utility, and the evaluation result (metric).

        Returns:
            A collection of marginals. Each marginal is a tuple with index and its marginal
            utility.
        """
        marginals: List[List[MarginalT]] = []
        for batch_id, (s, evaluation) in enumerate(future_result):
            previous_values = self.compute_values()
            self.total_evaluations += 1
            self.point_in_subset[s] += 1
            self.positive_sums[s] += evaluation
            not_s = np.setdiff1d(self.all_indices, s)
            self.negative_sums[not_s] += evaluation
            new_values = self.compute_values()
            # Hack to work around the update mechanic that does not work out of the box for MSR
            marginal_vals = (
                self.total_evaluations * new_values
                - (self.total_evaluations - 1) * previous_values
            )
            marginals.append([])
            for data_index in range(self.n):
                marginals[batch_id].append(
                    (data_index, float(marginal_vals[data_index]))
                )
        return marginals


# @deprecated(
#     target=compute_semivalues,  # TODO: rename this to compute_semivalues
#     deprecated_in="0.8.0",
#     remove_in="0.9.0",
# )
@deprecated(
    target=True,
    args_mapping={"config": "config"},
    deprecated_in="0.9.0",
    remove_in="0.10.0",
)
def compute_generic_semivalues(
    sampler: PowersetSampler[IndexT],
    u: Utility,
    coefficient: SVCoefficient,
    done: StoppingCriterion,
    *,
    marginal: MarginalFunction = DefaultMarginal(),
    future_processor: FutureProcessor = FutureProcessor(),
    batch_size: int = 1,
    skip_converged: bool = False,
    n_jobs: int = 1,
    parallel_backend: Optional[ParallelBackend] = None,
    config: Optional[ParallelConfig] = None,
    progress: bool = False,
) -> ValuationResult:
    """Computes semi-values for a given utility function and subset sampler.

    Args:
        sampler: The subset sampler to use for utility computations.
        u: Utility object with model, data, and scoring function.
        coefficient: The semi-value coefficient
        done: Stopping criterion.
        marginal: Marginal function to be used for computing the semivalues
        future_processor: Additional postprocessing steps required for some algorithms
        batch_size: Number of marginal evaluations per single parallel job.
        skip_converged: Whether to skip marginal evaluations for indices that
            have already converged. **CAUTION**: This is only entirely safe if
            the stopping criterion is [MaxUpdates][pydvl.value.stopping.MaxUpdates].
            For any other stopping criterion, the convergence status of indices
            may change during the computation, or they may be marked as having
            converged even though in fact the estimated values are far from the
            true values (e.g. for
            [AbsoluteStandardError][pydvl.value.stopping.AbsoluteStandardError],
            you will probably have to carefully adjust the threshold).
        n_jobs: Number of parallel jobs to use.
        parallel_backend: Parallel backend instance to use
            for parallelizing computations. If `None`,
            use [JoblibParallelBackend][pydvl.parallel.backends.JoblibParallelBackend] backend.
            See the [Parallel Backends][pydvl.parallel.backends] package
            for available options.
        config: (**DEPRECATED**) Object configuring parallel computation,
            with cluster address, number of cpus, etc.
        progress: Whether to display a progress bar.

    Returns:
        Object with the results.

    !!! warning "Deprecation notice"
        Parameter `batch_size` is for experimental use and will be removed in
        future versions.

    !!! tip "Changed in version 0.9.0"
        Deprecated `config` argument and added a `parallel_backend`
        argument to allow users to pass the Parallel Backend instance
        directly.
    """
    if isinstance(sampler, PermutationSampler) and u.cache is None:
        log.warning(
            "PermutationSampler requires caching to be enabled or computation "
            "will be doubled wrt. a 'direct' implementation of permutation MC"
        )

    if batch_size != 1:
        warnings.warn(
            "Parameter `batch_size` is for experimental use and will be"
            " removed in future versions",
            DeprecationWarning,
        )

    result = ValuationResult.zeros(
        algorithm=f"semivalue-{str(sampler)}-{coefficient.__name__}",  # type: ignore
        indices=u.data.indices,
        data_names=u.data.data_names,
    )

    parallel_backend = _maybe_init_parallel_backend(parallel_backend, config)
    u = parallel_backend.put(u)
    correction = parallel_backend.put(
        lambda n, k: coefficient(n, k) * sampler.weight(n, k)
    )

    max_workers = parallel_backend.effective_n_jobs(n_jobs)
    n_submitted_jobs = 2 * max_workers  # number of jobs in the queue

    sampler_it = iter(sampler)
    pbar = tqdm(disable=not progress, total=100, unit="%")

    with parallel_backend.executor(
        max_workers=max_workers, cancel_futures=True
    ) as executor:
        pending: set[Future] = set()
        while True:
            pbar.n = 100 * done.completion()
            pbar.refresh()

            completed, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in completed:
                processed_future = future_processor(
                    future.result()
                )  # List of tuples or
                for batch_future in processed_future:
                    if isinstance(batch_future, list):  # Case when batch size is > 1
                        for idx, marginal_val in batch_future:
                            result.update(idx, marginal_val)
                    else:  # Batch size 1
                        idx, marginal_val = batch_future
                        result.update(idx, marginal_val)
                    if done(result):
                        return result

            # Ensure that we always have n_submitted_jobs running
            try:
                while len(pending) < n_submitted_jobs:
                    samples = tuple(islice(sampler_it, batch_size))
                    if len(samples) == 0:
                        raise StopIteration

                    # Filter out samples for indices that have already converged
                    filtered_samples = samples
                    if skip_converged and np.count_nonzero(done.converged) > 0:
                        # TODO: cloudpickle can't pickle result of `filter` on python 3.8
                        filtered_samples = tuple(
                            filter(lambda t: not done.converged[t[0]], samples)
                        )

                    if filtered_samples:
                        pending.add(
                            executor.submit(
                                marginal,
                                u=u,
                                coefficient=correction,
                                samples=filtered_samples,
                            )
                        )
            except StopIteration:
                if len(pending) == 0:
                    return result


def always_one_coefficient(n: int, k: int) -> float:
    return 1.0


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


@deprecated(
    target=True,
    args_mapping={"config": "config"},
    deprecated_in="0.9.0",
    remove_in="0.10.0",
)
def compute_shapley_semivalues(
    u: Utility,
    *,
    done: StoppingCriterion,
    sampler_t: Type[StochasticSampler] = PermutationSampler,
    batch_size: int = 1,
    n_jobs: int = 1,
    parallel_backend: Optional[ParallelBackend] = None,
    config: Optional[ParallelConfig] = None,
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
        sampler_t: The sampler type to use. See the
            [sampler][pydvl.value.sampler] module for a list.
        batch_size: Number of marginal evaluations per single parallel job.
        n_jobs: Number of parallel jobs to use.
        parallel_backend: Parallel backend instance to use
            for parallelizing computations. If `None`,
            use [JoblibParallelBackend][pydvl.parallel.backends.JoblibParallelBackend] backend.
            See the [Parallel Backends][pydvl.parallel.backends] package
            for available options.
        config: (**DEPRECATED**) Object configuring parallel computation,
            with cluster address, number of cpus, etc.
        seed: Either an instance of a numpy random number generator or a seed
            for it.
        progress: Whether to display a progress bar.

    Returns:
        Object with the results.

    !!! warning "Deprecation notice"
        Parameter `batch_size` is for experimental use and will be removed in
        future versions.

    !!! tip "Changed in version 0.9.0"
        Deprecated `config` argument and added a `parallel_backend`
        argument to allow users to pass the Parallel Backend instance
        directly.
    """
    # HACK: cannot infer return type because of useless IndexT, NameT
    return compute_generic_semivalues(  # type: ignore
        sampler_t(u.data.indices, seed=seed),
        u,
        shapley_coefficient,
        done,
        batch_size=batch_size,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        config=config,
        progress=progress,
    )


@deprecated(
    target=True,
    args_mapping={"config": "config"},
    deprecated_in="0.9.0",
    remove_in="0.10.0",
)
def compute_banzhaf_semivalues(
    u: Utility,
    *,
    done: StoppingCriterion,
    sampler_t: Type[StochasticSampler] = PermutationSampler,
    batch_size: int = 1,
    n_jobs: int = 1,
    parallel_backend: Optional[ParallelBackend] = None,
    config: Optional[ParallelConfig] = None,
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
        sampler_t: The sampler type to use. See the
            [sampler][pydvl.value.sampler] module for a list.
        batch_size: Number of marginal evaluations per single parallel job.
        n_jobs: Number of parallel jobs to use.
        seed: Either an instance of a numpy random number generator or a seed
            for it.
        parallel_backend: Parallel backend instance to use
            for parallelizing computations. If `None`,
            use [JoblibParallelBackend][pydvl.parallel.backends.JoblibParallelBackend] backend.
            See the [Parallel Backends][pydvl.parallel.backends] package
            for available options.
        config: (**DEPRECATED**) Object configuring parallel computation,
            with cluster address, number of cpus, etc.
        progress: Whether to display a progress bar.

    Returns:
        Object with the results.

    !!! warning "Deprecation notice"
        Parameter `batch_size` is for experimental use and will be removed in
        future versions.

    !!! tip "Changed in version 0.9.0"
        Deprecated `config` argument and added a `parallel_backend`
        argument to allow users to pass the Parallel Backend instance
        directly.
    """
    # HACK: cannot infer return type because of useless IndexT, NameT
    return compute_generic_semivalues(  # type: ignore
        sampler_t(u.data.indices, seed=seed),
        u,
        banzhaf_coefficient,
        done,
        batch_size=batch_size,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        config=config,
        progress=progress,
    )


@deprecated(
    target=True,
    args_mapping={"config": "config"},
    deprecated_in="0.9.0",
    remove_in="0.10.0",
)
def compute_msr_banzhaf_semivalues(
    u: Utility,
    *,
    done: StoppingCriterion,
    sampler_t: Type[StochasticSampler] = MSRSampler,
    batch_size: int = 1,
    n_jobs: int = 1,
    parallel_backend: Optional[ParallelBackend] = None,
    config: Optional[ParallelConfig] = None,
    progress: bool = False,
    seed: Optional[Seed] = None,
) -> ValuationResult:
    """Computes MSR sampled Banzhaf values for a given utility function.

    This is a convenience wrapper for
    [compute_generic_semivalues][pydvl.value.semivalues.compute_generic_semivalues]
    with the Banzhaf coefficient and MSR sampling.

    This algorithm works by sampling random subsets and then evaluating the utility
    on that subset only once. Based on the evaluation and the subset indices,
    the MSRFutureProcessor then computes the marginal updates like in the paper
    (Wang et. al.)<sup><a href="wang_data_2023">3</a></sup>.
    Their approach updates the semivalues for all data points every time a new evaluation
    is computed. This increases sample efficiency compared to normal Monte Carlo updates.

    Args:
        u: Utility object with model, data, and scoring function.
        done: Stopping criterion.
        sampler_t: The sampler type to use. See the
            [sampler][pydvl.value.sampler] module for a list.
        batch_size: Number of marginal evaluations per single parallel job.
        n_jobs: Number of parallel jobs to use.
        seed: Either an instance of a numpy random number generator or a seed
            for it.
        config: Object configuring parallel computation, with cluster address,
            number of cpus, etc.
        progress: Whether to display a progress bar.

    Returns:
        Object with the results.

    !!! warning "Deprecation notice"
        Parameter `batch_size` is for experimental use and will be removed in
        future versions.
    """
    # HACK: cannot infer return type because of useless IndexT, NameT
    return compute_generic_semivalues(  # type: ignore
        sampler_t(u.data.indices, seed=seed),
        u,
        always_one_coefficient,
        done,
        marginal=RawUtility(),
        future_processor=MSRFutureProcessor(u),
        batch_size=batch_size,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        config=config,
        progress=progress,
    )


@deprecated(
    target=True,
    args_mapping={"config": "config"},
    deprecated_in="0.9.0",
    remove_in="0.10.0",
)
def compute_beta_shapley_semivalues(
    u: Utility,
    *,
    alpha: float = 1,
    beta: float = 1,
    done: StoppingCriterion,
    sampler_t: Type[StochasticSampler] = PermutationSampler,
    batch_size: int = 1,
    n_jobs: int = 1,
    parallel_backend: Optional[ParallelBackend] = None,
    config: Optional[ParallelConfig] = None,
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
        sampler_t: The sampler type to use. See the
            [sampler][pydvl.value.sampler] module for a list.
        batch_size: Number of marginal evaluations per (parallelized) task.
        n_jobs: Number of parallel jobs to use.
        seed: Either an instance of a numpy random number generator or a seed for it.
        parallel_backend: Parallel backend instance to use
            for parallelizing computations. If `None`,
            use [JoblibParallelBackend][pydvl.parallel.backends.JoblibParallelBackend] backend.
            See the [Parallel Backends][pydvl.parallel.backends] package
            for available options.
        config: (**DEPRECATED**) Object configuring parallel computation,
            with cluster address, number of cpus, etc.
        progress: Whether to display a progress bar.

    Returns:
        Object with the results.

    !!! warning "Deprecation notice"
        Parameter `batch_size` is for experimental use and will be removed in
        future versions.

    !!! tip "Changed in version 0.9.0"
        Deprecated `config` argument and added a `parallel_backend`
        argument to allow users to pass the Parallel Backend instance
        directly.
    """
    # HACK: cannot infer return type because of useless IndexT, NameT
    return compute_generic_semivalues(  # type: ignore
        sampler_t(u.data.indices, seed=seed),
        u,
        beta_coefficient(alpha, beta),
        done,
        batch_size=batch_size,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        config=config,
        progress=progress,
    )


@deprecated(target=True, deprecated_in="0.7.0", remove_in="0.8.0")
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
    done: StoppingCriterion,
    mode: SemiValueMode = SemiValueMode.Shapley,
    sampler_t: Type[StochasticSampler] = PermutationSampler,
    batch_size: int = 1,
    n_jobs: int = 1,
    seed: Optional[Seed] = None,
    **kwargs: Any,
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
    - [SemiValueMode.BetaShapley][pydvl.value.semivalues.SemiValueMode]:
      Implements the Beta Shapley semi-value as introduced in
      (Kwon and Zou, 2022)<sup><a href="#kwon_beta_2022">1</a></sup>.
      Pass additional keyword arguments `alpha` and `beta` to set the
      parameters of the Beta distribution (both default to 1).
    - [SemiValueMode.Banzhaf][pydvl.value.semivalues.SemiValueMode]: Implements
      the Banzhaf semi-value as introduced in (Wang and Jia, 2022)<sup><a
      href="#wang_data_2023">1</a></sup>.

    See [Data valuation][data-valuation-intro] for an overview of valuation.

    Args:
        u: Utility object with model, data, and scoring function.
        done: Stopping criterion.
        mode: The semi-value mode to use. See
            [SemiValueMode][pydvl.value.semivalues.SemiValueMode] for a list.
        sampler_t: The sampler type to use. See [sampler][pydvl.value.sampler]
            for a list.
        batch_size: Number of marginal evaluations per (parallelized) task.
        n_jobs: Number of parallel jobs to use.
        seed: Either an instance of a numpy random number generator or a seed for it.
        kwargs: Additional keyword arguments passed to
            [compute_generic_semivalues][pydvl.value.semivalues.compute_generic_semivalues].

    Returns:
        Object with the results.

    !!! warning "Deprecation notice"
        Parameter `batch_size` is for experimental use and will be removed in
        future versions.
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

    # HACK: cannot infer return type because of useless IndexT, NameT
    return compute_generic_semivalues(  # type: ignore
        sampler_t(u.data.indices, seed=seed),
        u,
        coefficient,
        done,
        n_jobs=n_jobs,
        batch_size=batch_size,
        **kwargs,
    )
