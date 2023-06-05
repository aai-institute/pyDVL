r"""
Monte Carlo approximations to Shapley Data values.

!!! Warning
   You probably want to use the common interface provided by
   [compute_shapley_values()][pydvl.value.shapley.compute_shapley_values] instead of directly using
   the functions in this module.

Because exact computation of Shapley values requires $\mathcal{O}(2^n)$
re-trainings of the model, several Monte Carlo approximations are available.
The first two sample from the powerset of the training data directly:
:func:`combinatorial_montecarlo_shapley` and :func:`owen_sampling_shapley`. The
latter uses a reformulation in terms of a continuous extension of the utility.

Alternatively, employing another reformulation of the expression above as a sum
over permutations, one has the implementation in
:func:`permutation_montecarlo_shapley`, or using an early stopping strategy to
reduce computation :func:`truncated_montecarlo_shapley`.

.. seealso::
   It is also possible to use [group_testing_shapley()][pydvl.value.shapley.gt.group_testing_shapley]
   to reduce the number of evaluations of the utility. The method is however
   typically outperformed by others in this module.

.. seealso::
   Additionally, you can consider grouping your data points using
   :class:`~pydvl.utils.dataset.GroupedDataset` and computing the values of the
   groups instead. This is not to be confused with "group testing" as
   implemented in [group_testing_shapley()][pydvl.value.shapley.gt.group_testing_shapley]: any of
   the algorithms mentioned above, including Group Testing, can work to valuate
   groups of samples as units.
"""
import logging
import math
import operator
from functools import reduce
from itertools import cycle, takewhile
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from pydvl.utils.config import ParallelConfig
from pydvl.utils.numeric import random_powerset
from pydvl.utils.parallel import MapReduceJob
from pydvl.utils.utility import Utility
from pydvl.value.result import ValuationResult
from pydvl.value.shapley.truncated import NoTruncation, TruncationPolicy
from pydvl.value.stopping import StoppingCriterion

logger = logging.getLogger(__name__)

__all__ = ["permutation_montecarlo_shapley", "combinatorial_montecarlo_shapley"]


def _permutation_montecarlo_shapley(
    u: Utility,
    *,
    done: StoppingCriterion,
    truncation: TruncationPolicy,
    algorithm_name: str = "permutation_montecarlo_shapley",
    progress: bool = False,
    job_id: int = 1,
) -> ValuationResult:
    """Helper function for :func:`permutation_montecarlo_shapley`.

    Computes marginal utilities of each training sample in
    :obj:`pydvl.utils.utility.Utility.data` by iterating through randomly
    sampled permutations.

        u: Utility object with model, data, and scoring function
        done: Check on the results which decides when to stop
        truncation: A callable which decides whether to interrupt
        processing a permutation and set all subsequent marginals to zero.
        algorithm_name: For the results object. Used internally by different
        variants of Shapley using this subroutine
        progress: Whether to display progress bars for each job.
        job_id: id to use for reporting progress (e.g. to place progres bars)
    Returns:
        An object with the results
    """
    result = ValuationResult.zeros(
        algorithm=algorithm_name, indices=u.data.indices, data_names=u.data.data_names
    )

    pbar = tqdm(disable=not progress, position=job_id, total=100, unit="%")
    while not done(result):
        pbar.n = 100 * done.completion()
        pbar.refresh()
        prev_score = 0.0
        permutation = np.random.permutation(u.data.indices)
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
    return result


def permutation_montecarlo_shapley(
    u: Utility,
    done: StoppingCriterion,
    *,
    truncation: TruncationPolicy = NoTruncation(),
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    progress: bool = False,
) -> ValuationResult:
    r"""Computes an approximate Shapley value by sampling independent index
    permutations to approximate the sum:

    $$
    v_u(x_i) = \frac{1}{n!} \sum_{\sigma \in \Pi(n)}
    \tilde{w}( | \sigma_{:i} | )[u(\sigma_{:i} \cup \{i\}) − u(\sigma_{:i})],
    $$

    where $\sigma_{:i}$ denotes the set of indices in permutation sigma before the
    position where $i$ appears (see [Data valuation][computing-data-values] for details).

        u: Utility object with model, data, and scoring function.
        done: function checking whether computation must stop.
        truncation: An optional callable which decides whether to
        interrupt processing a permutation and set all subsequent marginals to
        zero. Typically used to stop computation when the marginal is small.
        n_jobs: number of jobs across which to distribute the computation.
        config: Object configuring parallel computation, with cluster
        address, number of cpus, etc.
        progress: Whether to display progress bars for each job.
    Returns:
        Object with the data values.
    """

    map_reduce_job: MapReduceJob[Utility, ValuationResult] = MapReduceJob(
        u,
        map_func=_permutation_montecarlo_shapley,
        reduce_func=lambda results: reduce(operator.add, results),
        map_kwargs=dict(
            algorithm_name="permutation_montecarlo_shapley",
            done=done,
            truncation=truncation,
            progress=progress,
        ),
        config=config,
        n_jobs=n_jobs,
    )
    return map_reduce_job()


def _combinatorial_montecarlo_shapley(
    indices: Sequence[int],
    u: Utility,
    done: StoppingCriterion,
    *,
    progress: bool = False,
    job_id: int = 1,
) -> ValuationResult:
    """Helper function for :func:`combinatorial_montecarlo_shapley`.

    This is the code that is sent to workers to compute values using the
    combinatorial definition.

        indices: Indices of the samples to compute values for.
        u: Utility object with model, data, and scoring function
        done: Check on the results which decides when to stop sampling
        subsets for an index.
        progress: Whether to display progress bars for each job.
        job_id: id to use for reporting progress
    Returns:
        A tuple of ndarrays with estimated values and standard errors
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

    repeat_indices = takewhile(lambda _: not done(result), cycle(indices))
    pbar = tqdm(disable=not progress, position=job_id, total=100, unit="%")
    for idx in repeat_indices:
        pbar.n = 100 * done.completion()
        pbar.refresh()
        # Randomly sample subsets of full dataset without idx
        subset = np.setxor1d(u.data.indices, [idx], assume_unique=True)
        s = next(random_powerset(subset, n_samples=1))
        marginal = (u({idx}.union(s)) - u(s)) / math.comb(n - 1, len(s))
        result.update(idx, correction * marginal)

    return result


def combinatorial_montecarlo_shapley(
    u: Utility,
    done: StoppingCriterion,
    *,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    progress: bool = False,
) -> ValuationResult:
    r"""Computes an approximate Shapley value using the combinatorial
    definition:

    $$v_u(i) = \frac{1}{n} \sum_{S \subseteq N \setminus \{i\}}
    \binom{n-1}{ | S | }^{-1} [u(S \cup \{i\}) − u(S)]$$

    This consists of randomly sampling subsets of the power set of the training
    indices in [data][pydvl.utils.utility.Utility.data], and computing their
    marginal utilities. See [Data valuation][computing-data-values] for details.

    Note that because sampling is done with replacement, the approximation is
    poor even for $2^{m}$ subsets with $m>n$, even though there are $2^{n-1}$
    subsets for each $i$. Prefer
    [permutation_montecarlo_shapley()][pydvl.value.shapley.montecarlo.permutation_montecarlo_shapley].

    Parallelization is done by splitting the set of indices across processes and
    computing the sum over subsets $S \subseteq N \setminus \{i\}$ separately.

        u: Utility object with model, data, and scoring function
        done: Stopping criterion for the computation.
        n_jobs: number of parallel jobs across which to distribute the
        computation. Each worker receives a chunk of
        [indices][pydvl.utils.dataset.Dataset.indices]
        config: Object configuring parallel computation, with cluster
        address, number of cpus, etc.
        progress: Whether to display progress bars for each job.
    Returns:
        Object with the data values.
    """

    map_reduce_job: MapReduceJob[NDArray, ValuationResult] = MapReduceJob(
        u.data.indices,
        map_func=_combinatorial_montecarlo_shapley,
        reduce_func=lambda results: reduce(operator.add, results),
        map_kwargs=dict(u=u, done=done, progress=progress),
        n_jobs=n_jobs,
        config=config,
    )
    return map_reduce_job()
