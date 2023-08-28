r"""
Monte Carlo approximations to Shapley Data values.

.. warning::
   You probably want to use the common interface provided by
   :func:`~pydvl.value.shapley.compute_shapley_values` instead of directly using
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
   It is also possible to use :func:`~pydvl.value.shapley.gt.group_testing_shapley`
   to reduce the number of evaluations of the utility. The method is however
   typically outperformed by others in this module.

.. seealso::
   Additionally, you can consider grouping your data points using
   :class:`~pydvl.utils.dataset.GroupedDataset` and computing the values of the
   groups instead. This is not to be confused with "group testing" as
   implemented in :func:`~pydvl.value.shapley.gt.group_testing_shapley`: any of
   the algorithms mentioned above, including Group Testing, can work to valuate
   groups of samples as units.
"""
from __future__ import annotations

import logging
import math
import operator
from concurrent.futures import FIRST_COMPLETED, Future, wait
from functools import reduce
from itertools import cycle, takewhile
from typing import Optional, Sequence

import numpy as np
from deprecate import deprecated
from numpy.typing import NDArray
from tqdm import tqdm

from pydvl.utils import Dataset, effective_n_jobs, init_executor, init_parallel_backend
from pydvl.utils.config import ParallelConfig
from pydvl.utils.numeric import random_powerset
from pydvl.utils.parallel import CancellationPolicy, MapReduceJob
from pydvl.utils.utility import Utility
from pydvl.value.result import ValuationResult
from pydvl.value.shapley.truncated import NoTruncation, TruncationPolicy
from pydvl.value.stopping import MaxChecks, StoppingCriterion

logger = logging.getLogger(__name__)

__all__ = [
    "permutation_montecarlo_shapley",
    "permutation_montecarlo_shapley_rollout",
    "combinatorial_montecarlo_shapley",
]


def permutation_montecarlo_shapley_rollout(
    u: Utility,
    permutation: NDArray[np.int_],
    truncation: TruncationPolicy,
    algorithm_name: str,
    additional_indices: Optional[NDArray[np.int_]] = None,
    use_default_scorer_value: bool = True,
) -> ValuationResult:
    """
    A truncated version of a permutation-based MC estimator.
    values. It generates a permutation p[i] of the class label indices and iterates over
    all subsets starting from the empty set to the full set of indices.

    :param u: Utility object containing model, data, and scoring function.
    :param permutation: Permutation of indices to be considered.
    :param truncation: Callable which decides whether to interrupt processing a
        permutation and set all subsequent marginals to zero.
    :param additional_indices: Set of additional indices for data points which should be
        always considered.
    :param use_default_scorer_value: Use default scorer value even if additional_indices
        is not None.
    :return: ValuationResult object containing computed data values.
    """
    if (
        additional_indices is not None
        and len(np.intersect1d(permutation, additional_indices)) > 0
    ):
        raise ValueError(
            "The class label set and the complement set have to be disjoint."
        )

    result = ValuationResult.zeros(
        algorithm=algorithm_name,
        indices=u.data.indices,
        data_names=u.data.data_names,
    )

    prev_score = (
        u.default_score
        if (
            use_default_scorer_value
            or additional_indices is None
            or additional_indices is not None
            and len(additional_indices) == 0
        )
        else u(additional_indices)
    )

    truncation_u = u
    if additional_indices is not None:
        # hack to calculate the correct value in reset.
        truncation_indices = np.sort(np.concatenate((permutation, additional_indices)))
        truncation_u = Utility(
            u.model,
            Dataset(
                u.data.x_train[truncation_indices],
                u.data.y_train[truncation_indices],
                u.data.x_test,
                u.data.y_test,
            ),
            u.scorer,
        )
    truncation.reset(truncation_u)

    is_terminated = False
    for i, idx in enumerate(permutation):
        if is_terminated or (is_terminated := truncation(i, prev_score)):
            score = prev_score
        else:
            score = u(
                np.concatenate((permutation[: i + 1], additional_indices))
                if additional_indices is not None and len(additional_indices) > 0
                else permutation[: i + 1]
            )

        marginal = score - prev_score
        result.update(idx, marginal)
        prev_score = score

    return result


@deprecated(
    target=True,
    deprecated_in="0.7.0",
    remove_in="0.8.0",
    args_mapping=dict(
        coordinator_update_period=None, worker_update_period=None, progress=None
    ),
)
def permutation_montecarlo_shapley(
    u: Utility,
    done: StoppingCriterion,
    *,
    truncation: TruncationPolicy = NoTruncation(),
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    progress: bool = False,
) -> ValuationResult:
    r"""Computes an approximate Shapley value by sampling independent
    permutations of the index set, approximating the sum:

    $$
    v_u(x_i) = \frac{1}{n!} \sum_{\sigma \in \Pi(n)}
    \tilde{w}( | \sigma_{:i} | )[u(\sigma_{:i} \cup \{i\}) − u(\sigma_{:i})],
    $$

    where $\sigma_{:i}$ denotes the set of indices in permutation sigma before
    the position where $i$ appears (see :ref:`data valuation` for details).

    This implements the method described in :footcite:t:`ghorbani_data_2019`
    with a double stopping criterion.

    .. todo::
       Think of how to add Robin-Gelman or some other more principled stopping
       criterion.

    Instead of naively implementing the expectation, we sequentially add points
    to coalitions from a permutation and incrementally compute marginal utilities.
    We stop computing marginals for a given permutation based on a
    :class:`TruncationPolicy`. :footcite:t:`ghorbani_data_2019` mention two
    policies: one that stops after a certain fraction of marginals are computed,
    implemented in :class:`FixedTruncation`, and one that stops if the last
    computed utility ("score") is close to the total utility using the standard
    deviation of the utility as a measure of proximity, implemented in
    :class:`BootstrapTruncation`.

    We keep sampling permutations and updating all shapley values
    until the :class:`StoppingCriterion` returns ``True``.

    :param u: Utility object with model, data, and scoring function.
    :param done: function checking whether computation must stop.
    :param truncation: An optional callable which decides whether to
        interrupt processing a permutation and set all subsequent marginals to
        zero. Typically used to stop computation when the marginal is small.
    :param n_jobs: number of jobs across which to distribute the computation.
    :param config: Object configuring parallel computation, with cluster
        address, number of cpus, etc.
    :param progress: Whether to display progress bars for each job.
    :return: Object with the data values.
    """
    algorithm = "permutation_montecarlo_shapley"

    parallel_backend = init_parallel_backend(config)
    u = parallel_backend.put(u)
    max_workers = effective_n_jobs(n_jobs, config)
    n_submitted_jobs = 2 * max_workers  # number of jobs in the executor's queue

    result = ValuationResult.zeros(algorithm=algorithm)

    pbar = tqdm(disable=not progress, total=100, unit="%")

    with init_executor(
        max_workers=max_workers, config=config, cancel_futures=CancellationPolicy.ALL
    ) as executor:
        pending: set[Future] = set()
        while True:
            pbar.n = 100 * done.completion()
            pbar.refresh()

            completed, pending = wait(
                pending, timeout=config.wait_timeout, return_when=FIRST_COMPLETED
            )

            for future in completed:
                result += future.result()
                # we could check outside the loop, but that means more
                # submissions if the stopping criterion is unstable
                if done(result):
                    return result

            # Ensure that we always have n_submitted_jobs in the queue or running
            for _ in range(n_submitted_jobs - len(pending)):
                p = np.random.permutation(u.data.indices)
                future = executor.submit(
                    permutation_montecarlo_shapley_rollout, u, p, truncation, algorithm
                )
                pending.add(future)


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

    :param indices: Indices of the samples to compute values for.
    :param u: Utility object with model, data, and scoring function
    :param done: Check on the results which decides when to stop sampling
        subsets for an index.
    :param progress: Whether to display progress bars for each job.
    :param job_id: id to use for reporting progress
    :return: A tuple of ndarrays with estimated values and standard errors
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
    indices in :attr:`~pydvl.utils.utility.Utility.data`, and computing their
    marginal utilities. See :ref:`data valuation` for details.

    Note that because sampling is done with replacement, the approximation is
    poor even for $2^{m}$ subsets with $m>n$, even though there are $2^{n-1}$
    subsets for each $i$. Prefer
    :func:`~pydvl.shapley.montecarlo.permutation_montecarlo_shapley`.

    Parallelization is done by splitting the set of indices across processes and
    computing the sum over subsets $S \subseteq N \setminus \{i\}$ separately.

    :param u: Utility object with model, data, and scoring function
    :param done: Stopping criterion for the computation.
    :param n_jobs: number of parallel jobs across which to distribute the
        computation. Each worker receives a chunk of
        :attr:`~pydvl.utils.dataset.Dataset.indices`
    :param config: Object configuring parallel computation, with cluster
        address, number of cpus, etc.
    :param progress: Whether to display progress bars for each job.
    :return: Object with the data values.
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
