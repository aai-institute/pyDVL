r"""
Monte Carlo approximations to Shapley Data values.

!!! Warning
    You probably want to use the common interface provided by
    [compute_shapley_values()][pydvl.value.shapley.compute_shapley_values]
    instead of directly using the functions in this module.

Because exact computation of Shapley values requires $\mathcal{O}(2^n)$
re-trainings of the model, several Monte Carlo approximations are available. The
first two sample from the powerset of the training data directly:
[combinatorial_montecarlo_shapley()][pydvl.value.shapley.montecarlo.combinatorial_montecarlo_shapley]
and [owen_sampling_shapley()][pydvl.value.shapley.owen.owen_sampling_shapley].
The latter uses a reformulation in terms of a continuous extension of the
utility.

Alternatively, employing another reformulation of the expression above as a sum
over permutations, one has the implementation in
[permutation_montecarlo_shapley()][pydvl.value.shapley.montecarlo.permutation_montecarlo_shapley]
with the option to pass an early stopping strategy to reduce computation
as done in Truncated MonteCarlo Shapley (TMCS).

!!! info "Also see"
    It is also possible to use [group_testing_shapley()][pydvl.value.shapley.gt.group_testing_shapley]
    to reduce the number of evaluations of the utility. The method is however
    typically outperformed by others in this module.

!!! info "Also see"
    Additionally, you can consider grouping your data points using
    [GroupedDataset][pydvl.utils.dataset.GroupedDataset] and computing the values
    of the groups instead. This is not to be confused with "group testing" as
    implemented in [group_testing_shapley()][pydvl.value.shapley.gt.group_testing_shapley]: any of
    the algorithms mentioned above, including Group Testing, can work to valuate
    groups of samples as units.

## References

[^1]: <a name="ghorbani_data_2019"></a>Ghorbani, A., Zou, J., 2019.
    [Data Shapley: Equitable Valuation of Data for Machine Learning](https://proceedings.mlr.press/v97/ghorbani19c.html).
    In: Proceedings of the 36th International Conference on Machine Learning, PMLR, pp. 2242–2251.

"""

from __future__ import annotations

import logging
import math
import operator
from concurrent.futures import FIRST_COMPLETED, Future, wait
from functools import reduce
from typing import Optional, Sequence, Union

import numpy as np
from deprecate import deprecated
from numpy.random import SeedSequence
from numpy.typing import NDArray
from tqdm.auto import tqdm

from pydvl.parallel import (
    CancellationPolicy,
    MapReduceJob,
    ParallelBackend,
    ParallelConfig,
    _maybe_init_parallel_backend,
)
from pydvl.utils.numeric import random_powerset
from pydvl.utils.progress import repeat_indices
from pydvl.utils.types import Seed, ensure_seed_sequence
from pydvl.utils.utility import Utility
from pydvl.value.result import ValuationResult
from pydvl.value.shapley.truncated import NoTruncation, TruncationPolicy
from pydvl.value.stopping import StoppingCriterion

logger = logging.getLogger(__name__)

__all__ = ["permutation_montecarlo_shapley", "combinatorial_montecarlo_shapley"]


def _permutation_montecarlo_one_step(
    u: Utility,
    truncation: TruncationPolicy,
    algorithm_name: str,
    seed: Optional[Union[Seed, SeedSequence]] = None,
) -> ValuationResult:
    """Helper function for
    [permutation_montecarlo_shapley()][pydvl.value.shapley.montecarlo.permutation_montecarlo_shapley].

    Computes marginal utilities of each training sample in a randomly sampled
    permutation.

    Args:
        u: Utility object with model, data, and scoring function
        truncation: A callable which decides whether to interrupt
            processing a permutation and set all subsequent marginals to zero.
        algorithm_name: For the results object. Used internally by different
            variants of Shapley using this subroutine
        seed: Either an instance of a numpy random number generator or a seed
            for it.

    Returns:
        An object with the results
    """

    result = ValuationResult.zeros(
        algorithm=algorithm_name, indices=u.data.indices, data_names=u.data.data_names
    )
    prev_score = 0.0
    permutation = np.random.default_rng(seed).permutation(u.data.indices)
    permutation_done = False
    truncation.reset()
    for i, idx in enumerate(permutation):
        if permutation_done:
            score = prev_score
        else:
            score = u(permutation[: i + 1])
        marginal = score - prev_score
        result.update(idx, marginal)
        prev_score = score
        if not permutation_done and truncation(i, score):
            permutation_done = True
    nans = np.isnan(result.values).sum()
    if nans > 0:
        logger.warning(
            f"{nans} NaN values in current permutation, ignoring. "
            "Consider setting a default value for the Scorer"
        )
        result = ValuationResult.empty(algorithm=algorithm_name)
    return result


@deprecated(
    target=True,
    args_mapping={"config": "config"},
    deprecated_in="0.9.0",
    remove_in="0.10.0",
)
def permutation_montecarlo_shapley(
    u: Utility,
    done: StoppingCriterion,
    *,
    truncation: TruncationPolicy = NoTruncation(),
    n_jobs: int = 1,
    parallel_backend: Optional[ParallelBackend] = None,
    config: Optional[ParallelConfig] = None,
    progress: bool = False,
    seed: Optional[Seed] = None,
) -> ValuationResult:
    r"""Computes an approximate Shapley value by sampling independent
    permutations of the index set, approximating the sum:

    $$
    v_u(x_i) = \frac{1}{n!} \sum_{\sigma \in \Pi(n)}
    \tilde{w}( | \sigma_{:i} | )[u(\sigma_{:i} \cup \{i\}) − u(\sigma_{:i})],
    $$

    where $\sigma_{:i}$ denotes the set of indices in permutation sigma before
    the position where $i$ appears (see [[data-valuation-intro]] for details).

    This implements the method described in (Ghorbani and Zou, 2019)<sup><a
    href="#ghorbani_data_2019">1</a></sup> with a double stopping criterion.

    !!! Todo
        Think of how to add Robin-Gelman or some other more principled stopping
        criterion.

    Instead of naively implementing the expectation, we sequentially add points
    to coalitions from a permutation and incrementally compute marginal utilities.
    We stop computing marginals for a given permutation based on a
    [TruncationPolicy][pydvl.value.shapley.truncated.TruncationPolicy].
    (Ghorbani and Zou, 2019)<sup><a href="#ghorbani_data_2019">1</a></sup>
    mention two policies: one that stops after a certain
    fraction of marginals are computed, implemented in
    [FixedTruncation][pydvl.value.shapley.truncated.FixedTruncation],
    and one that stops if the last computed utility ("score") is close to the
    total utility using the standard deviation of the utility as a measure of
    proximity, implemented in
    [BootstrapTruncation][pydvl.value.shapley.truncated.BootstrapTruncation].

    We keep sampling permutations and updating all shapley values
    until the [StoppingCriterion][pydvl.value.stopping.StoppingCriterion] returns
    `True`.

    Args:
        u: Utility object with model, data, and scoring function.
        done: function checking whether computation must stop.
        truncation: An optional callable which decides whether to interrupt
            processing a permutation and set all subsequent marginals to
            zero. Typically used to stop computation when the marginal is small.
        n_jobs: number of jobs across which to distribute the computation.
        parallel_backend: Parallel backend instance to use
            for parallelizing computations. If `None`,
            use [JoblibParallelBackend][pydvl.parallel.backends.JoblibParallelBackend] backend.
            See the [Parallel Backends][pydvl.parallel.backends] package
            for available options.
        config: (**DEPRECATED**) Object configuring parallel computation,
            with cluster address, number of cpus, etc.
        progress: Whether to display a progress bar.
        seed: Either an instance of a numpy random number generator or a seed for it.

    Returns:
        Object with the data values.

    !!! tip "Changed in version 0.9.0"
        Deprecated `config` argument and added a `parallel_backend`
        argument to allow users to pass the Parallel Backend instance
        directly.
    """
    algorithm = "permutation_montecarlo_shapley"

    parallel_backend = _maybe_init_parallel_backend(parallel_backend, config)
    u = parallel_backend.put(u)
    max_workers = parallel_backend.effective_n_jobs(n_jobs)
    n_submitted_jobs = 2 * max_workers  # number of jobs in the executor's queue

    seed_sequence = ensure_seed_sequence(seed)
    result = ValuationResult.zeros(
        algorithm=algorithm, indices=u.data.indices, data_names=u.data.data_names
    )

    pbar = tqdm(disable=not progress, total=100, unit="%")

    with parallel_backend.executor(
        max_workers=max_workers, cancel_futures=CancellationPolicy.ALL
    ) as executor:
        pending: set[Future] = set()
        while True:
            pbar.n = 100 * done.completion()
            pbar.refresh()

            completed, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            for future in completed:
                result += future.result()
                # we could check outside the loop, but that means more
                # submissions if the stopping criterion is unstable
                if done(result):
                    return result

            # Ensure that we always have n_submitted_jobs in the queue or running
            n_remaining_slots = n_submitted_jobs - len(pending)
            seeds = seed_sequence.spawn(n_remaining_slots)
            for i in range(n_remaining_slots):
                future = executor.submit(
                    _permutation_montecarlo_one_step,
                    u,
                    truncation,
                    algorithm,
                    seed=seeds[i],
                )
                pending.add(future)


def _combinatorial_montecarlo_shapley(
    indices: Sequence[int],
    u: Utility,
    done: StoppingCriterion,
    *,
    progress: bool = False,
    job_id: int = 1,
    seed: Optional[Seed] = None,
) -> ValuationResult:
    """Helper function for
    [combinatorial_montecarlo_shapley][pydvl.value.shapley.montecarlo.combinatorial_montecarlo_shapley].

    This is the code that is sent to workers to compute values using the
    combinatorial definition.

    Args:
        indices: Indices of the samples to compute values for.
        u: Utility object with model, data, and scoring function
        done: Check on the results which decides when to stop sampling
            subsets for an index.
        progress: Whether to display progress bars for each job.
        seed: Either an instance of a numpy random number generator or a seed
            for it.
        job_id: id to use for reporting progress

    Returns:
        The results for the indices.
    """
    n = len(u.data)

    # Correction coming from Monte Carlo integration so that the mean of the
    # marginals converges to the value: the uniform distribution over the
    # powerset of a set with n-1 elements has mass 2^{n-1} over each subset. The
    # additional factor n corresponds to the one in the Shapley definition
    correction = 2 ** (n - 1) / n
    result = ValuationResult.zeros(
        algorithm="combinatorial_montecarlo_shapley",
        indices=np.array(indices, dtype=np.int_),
        data_names=[u.data.data_names[i] for i in indices],
    )

    rng = np.random.default_rng(seed)

    for idx in repeat_indices(
        indices,
        result=result,  # type:ignore
        done=done,  # type:ignore
        disable=not progress,
        position=job_id,
    ):
        # Randomly sample subsets of full dataset without idx
        subset = np.setxor1d(u.data.indices, [idx], assume_unique=True)
        s = next(random_powerset(subset, n_samples=1, seed=rng))
        marginal = (u({idx}.union(s)) - u(s)) / math.comb(n - 1, len(s))
        result.update(idx, correction * marginal)

    return result


@deprecated(
    target=True,
    args_mapping={"config": "config"},
    deprecated_in="0.9.0",
    remove_in="0.10.0",
)
def combinatorial_montecarlo_shapley(
    u: Utility,
    done: StoppingCriterion,
    *,
    n_jobs: int = 1,
    parallel_backend: Optional[ParallelBackend] = None,
    config: Optional[ParallelConfig] = None,
    progress: bool = False,
    seed: Optional[Seed] = None,
) -> ValuationResult:
    r"""Computes an approximate Shapley value using the combinatorial
    definition:

    $$v_u(i) = \frac{1}{n} \sum_{S \subseteq N \setminus \{i\}}
    \binom{n-1}{ | S | }^{-1} [u(S \cup \{i\}) − u(S)]$$

    This consists of randomly sampling subsets of the power set of the training
    indices in [u.data][pydvl.utils.utility.Utility], and computing their
    marginal utilities. See [Data valuation][data-valuation-intro] for details.

    Note that because sampling is done with replacement, the approximation is
    poor even for $2^{m}$ subsets with $m>n$, even though there are $2^{n-1}$
    subsets for each $i$. Prefer
    [permutation_montecarlo_shapley()][pydvl.value.shapley.montecarlo.permutation_montecarlo_shapley].

    Parallelization is done by splitting the set of indices across processes and
    computing the sum over subsets $S \subseteq N \setminus \{i\}$ separately.

    Args:
        u: Utility object with model, data, and scoring function
        done: Stopping criterion for the computation.
        n_jobs: number of parallel jobs across which to distribute the
            computation. Each worker receives a chunk of
            [indices][pydvl.utils.dataset.Dataset.indices]
        parallel_backend: Parallel backend instance to use
            for parallelizing computations. If `None`,
            use [JoblibParallelBackend][pydvl.parallel.backends.JoblibParallelBackend] backend.
            See the [Parallel Backends][pydvl.parallel.backends] package
            for available options.
        config: (**DEPRECATED**) Object configuring parallel computation,
            with cluster address, number of cpus, etc.
        progress: Whether to display progress bars for each job.
        seed: Either an instance of a numpy random number generator or a seed for it.

    Returns:
        Object with the data values.

    !!! tip "Changed in version 0.9.0"
        Deprecated `config` argument and added a `parallel_backend`
        argument to allow users to pass the Parallel Backend instance
        directly.
    """
    parallel_backend = _maybe_init_parallel_backend(parallel_backend, config)

    map_reduce_job: MapReduceJob[NDArray, ValuationResult] = MapReduceJob(
        u.data.indices,
        map_func=_combinatorial_montecarlo_shapley,
        reduce_func=lambda results: reduce(operator.add, results),
        map_kwargs=dict(u=u, done=done, progress=progress),
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
    )
    return map_reduce_job(seed=seed)
