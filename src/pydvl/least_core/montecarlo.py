"""
.. versionadded:: 0.2.0
"""
import logging
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, Iterable, Tuple

import numpy as np
import scipy

from ..reporting.scores import sort_values
from ..utils import Utility, maybe_progress
from ..utils.config import ParallelConfig
from ..utils.numeric import PowerSetDistribution, random_powerset
from ..utils.parallel import MapReduceJob, init_parallel_backend

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


__all__ = ["montecarlo_least_core"]


def _montecarlo_least_core(
    u: Utility,
    max_iterations: int,
    dist: PowerSetDistribution,
    *,
    progress: bool = False,
    job_id: int = 1,
    **kwargs,
) -> Tuple["NDArray", "NDArray"]:
    """It calculates the difference between the score of a model with and without
    each training datapoint. This is repeated a number max_iterations of times and
    with different random combinations.

    :param u: Utility object with model, data, and scoring function
    :param max_iterations: total number of iterations (permutations) to use
    :param progress: true to plot progress bar
    :param job_id: id to use for reporting progress
    :return: a matrix with each row being a different permutation
        and each column being the score of a different data point
    """
    n = len(u.data)

    utility_values = np.zeros(max_iterations)

    # Randomly sample subsets of full dataset
    power_set = random_powerset(
        u.data.indices,
        dist=dist,
        max_subsets=max_iterations,
    )

    A_ub = np.zeros((max_iterations, n + 1), dtype=np.int32)
    A_ub[:, -1] = -1

    for i, subset in enumerate(
        maybe_progress(
            power_set,
            progress,
            total=max_iterations,
            position=job_id,
        )
    ):
        indices = np.zeros(n + 1, dtype=np.bool)
        indices[list(subset)] = True
        A_ub[i, indices] = -1
        utility_values[i] = u(subset)

    return utility_values, A_ub


def _reduce_func(
    results: Iterable[Tuple["NDArray", "NDArray"]]
) -> Tuple["NDArray", "NDArray"]:
    utility_values_list, A_ub_list = zip(*results)
    utility_values = np.concatenate(utility_values_list)
    A_ub = np.concatenate(A_ub_list)
    return utility_values, A_ub


def montecarlo_least_core(
    u: Utility,
    max_iterations: int,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    *,
    dist: PowerSetDistribution = PowerSetDistribution.WEIGHTED,
    progress: bool = False,
) -> "OrderedDict[str, float]":
    """Computes an approximate Shapley value using the combinatorial definition.

    :param u: utility
    :param max_iterations: total number of iterations (permutations) to use
    :param n_jobs: number of jobs across which to distribute the computation
    :param progress: true to plot progress bar
    :return: Tuple, with the first element being an ordered
        Dict of approximated Shapley values for the indices, the second being the
        montecarlo error related to each of them.
    """
    n = len(u.data)

    max_iterations = min(max_iterations, 2**n)

    parallel_backend = init_parallel_backend(config)

    u_id = parallel_backend.put(u)

    iterations_per_job = max_iterations // n_jobs

    map_reduce_job: MapReduceJob["Utility", Tuple["NDArray", "NDArray"]] = MapReduceJob(
        map_func=_montecarlo_least_core,
        reduce_func=_reduce_func,
        map_kwargs=dict(
            dist=dist,
            max_iterations=iterations_per_job,
            progress=progress,
        ),
        config=config,
    )
    utility_values, A_ub = map_reduce_job(u_id, chunkify_inputs=False, n_jobs=n_jobs)[0]

    if np.any(np.isnan(utility_values)):
        warnings.warn(
            f"Calculation returned {np.sum(np.isnan(utility_values))} nan values out of {utility_values.size}",
            RuntimeWarning,
        )

    c = np.zeros(n + 1, dtype=np.int32)
    c[-1] = 1
    A_eq = np.ones((1, n + 1), dtype=np.int32)
    A_eq[:, -1] = 0
    b_ub = -utility_values
    b_eq = np.array([u(u.data.indices)])

    result = scipy.optimize.linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        method="highs",
        bounds=(None, None),
    )

    if not result.success:
        warnings.warn("Could not find optimal solution", RuntimeWarning)
        return {}

    values = result.x[:-1]
    sorted_values = sort_values({u.data.data_names[i]: v for i, v in enumerate(values)})
    return sorted_values
