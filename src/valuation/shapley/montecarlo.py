import logging
import math
from collections import OrderedDict
from time import sleep, time
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np

from valuation.reporting.scores import sort_values
from valuation.utils import Utility
from valuation.utils.config import ParallelConfig
from valuation.utils.numeric import PowerSetDistribution, random_powerset
from valuation.utils.parallel import MapReduceJob, init_parallel_backend
from valuation.utils.progress import maybe_progress

from .actor import get_shapley_coordinator, get_shapley_worker

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = logging.getLogger(__name__)

__all__ = [
    "truncated_montecarlo_shapley",
    "permutation_montecarlo_shapley",
    "combinatorial_montecarlo_shapley",
]


def truncated_montecarlo_shapley(
    u: Utility,
    score_tolerance: Optional[float] = None,
    max_iterations: Optional[int] = None,
    n_jobs: Optional[int] = None,
    config: ParallelConfig = ParallelConfig(),
    *,
    progress: bool = False,
    coordinator_update_frequency: int = 10,
    worker_update_frequency: int = 5,
) -> Tuple[OrderedDict, Dict]:
    """MonteCarlo approximation to the Shapley value of data points.

    Instead of naively implementing the expectation, we sequentially add points
    to a dataset from a permutation. We keep sampling permutations and updating
    all shapley values until the std/value score in
    the moving average falls below a given threshold (score_tolerance) or
    or when the number of iterations exceeds a certain number (max_iterations).

    :param u: Utility object with model, data, and scoring function
    :param score_tolerance: During calculation of shapley values, the
        coordinator will check if the median standard deviation over average
        score for each point's has dropped below score_tolerance.
        If so, the computation will be terminated.
    :param max_iterations: a sum of the total number of permutation is calculated
        If the current number of permutations has exceeded max_iterations, computation
        will stop.
    :param n_jobs: number of jobs processing permutations. If None, it will be set
        to available_cpus().
    :param address: if None, shapley calculation will run only on local machine.
        If "auto", it will use a local cluster if already started.
        If "ray://{ip address}", it will run the process on the cluster
        found at the IP address passed, e.g. "ray://123.45.67.89:10001" will run the process
        on the cluster at 123.45.67.89:10001.
    :param progress: set to True to use tqdm progress bars.
    :return: Tuple, with the first element being an ordered
        Dict of approximated Shapley values for the indices, the second being the
        montecarlo error related to each dvl value.
    """
    parallel_backend = init_parallel_backend(config)

    n_jobs = parallel_backend.effective_n_jobs(n_jobs)

    u_id = parallel_backend.put(u)

    coordinator = get_shapley_coordinator(  # type: ignore
        score_tolerance, max_iterations, progress, config=config
    )
    workers = [
        get_shapley_worker(  # type: ignore
            u=u_id,
            coordinator=coordinator,
            worker_id=worker_id,
            progress=progress,
            update_frequency=worker_update_frequency,
            config=config,
        )
        for worker_id in range(n_jobs)
    ]
    for worker_id in range(n_jobs):
        workers[worker_id].run(block=False)
    last_update_time = time()
    is_done = False
    while not is_done:
        sleep(0.01)
        if time() - last_update_time > coordinator_update_frequency:
            is_done = coordinator.check_status()
            last_update_time = time()
    dvl_values, dvl_std = coordinator.get_results()
    sorted_shapley_values = sort_values(
        {u.data.data_names[i]: v for i, v in enumerate(dvl_values)}
    )
    montecarlo_error = {u.data.data_names[i]: v for i, v in enumerate(dvl_std)}
    return sorted_shapley_values, montecarlo_error


def _permutation_montecarlo_shapley(
    u: Utility, max_permutations: int, progress: bool = False, job_id: int = 1, **kwargs
) -> "NDArray":
    """It calculates the difference between the score of a model with and without
    each training datapoint. This is repeated a number max_permutations of times and
    with different permutations.

    :param u: Utility object with model, data, and scoring function
    :param max_iterations: total number of iterations (permutations) to use
    :param progress: true to plot progress bar
    :param job_id: id to use for reporting progress
    :return: a matrix with each row being a different permutation
        and each column being the score of a different data point
    """
    n = len(u.data)
    values = np.zeros(shape=(max_permutations, n))
    pbar = maybe_progress(max_permutations, progress, position=job_id)
    for iter_idx, _ in enumerate(pbar):
        prev_score = 0.0
        permutation = np.random.permutation(u.data.indices)
        marginals = np.zeros(shape=(n))
        for i, idx in enumerate(permutation):
            score = u(permutation[: i + 1])
            marginals[idx] = score - prev_score
            prev_score = score
        values[iter_idx] = marginals
    return values


def permutation_montecarlo_shapley(
    u: Utility,
    max_iterations: int,
    n_jobs: int,
    config: ParallelConfig = ParallelConfig(),
    *,
    progress: bool = False,
) -> Tuple[OrderedDict, Dict]:
    """Computes an approximate Shapley value using independent permutations of the indices.

    :param u: Utility object with model, data, and scoring function
    :param max_iterations: total number of iterations (permutations) to use
    :param n_jobs: number of jobs across which to distribute the computation
    :param progress: true to plot progress bar
    :return: Tuple, with the first element being an ordered
        Dict of approximated Shapley values for the indices, the second being the
        montecarlo error related to each of them.
    """
    parallel_backend = init_parallel_backend(config)

    u_id = parallel_backend.put(u)

    iterations_per_job = max_iterations // n_jobs

    map_reduce_job: MapReduceJob["Utility", "NDArray"] = MapReduceJob(
        map_func=_permutation_montecarlo_shapley,
        reduce_func=np.concatenate,
        map_kwargs=dict(max_permutations=iterations_per_job, progress=progress),
        reduce_kwargs=dict(axis=0),
        config=config,
    )
    full_results = map_reduce_job(u_id, chunkify_inputs=False, n_jobs=n_jobs)[0]

    # Careful: for some models there might be nans, e.g. for i=0 or i=1!
    if np.any(np.isnan(full_results)):
        log.warning(
            f"Calculation returned {np.sum(np.isnan(full_results))} nan values out of {full_results.size}"
        )
    acc = np.nanmean(full_results, axis=0)
    acc_std = np.nanstd(full_results, axis=0) / np.sqrt(full_results.shape[0])
    sorted_shapley_values = sort_values(
        {u.data.data_names[i]: v for i, v in enumerate(acc)}
    )
    montecarlo_error = {u.data.data_names[i]: v for i, v in enumerate(acc_std)}
    return sorted_shapley_values, montecarlo_error


def _combinatorial_montecarlo_shapley(
    u: Utility,
    max_iterations: int,
    dist: PowerSetDistribution,
    *,
    progress: bool = False,
    job_id: int = 1,
    **kwargs,
):
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
    correction = 2 ** (n - 1) / n
    values = np.zeros(shape=(max_iterations, n))
    pbar = maybe_progress(u.data.indices, progress, position=job_id)
    for idx, _ in enumerate(pbar):
        # Randomly sample subsets of full dataset without idx
        subset = np.setxor1d(u.data.indices, [idx], assume_unique=True)
        power_set = random_powerset(
            subset,
            dist=dist,
            max_subsets=max_iterations,
        )
        # Normalization accounts for a uniform dist. on powerset (i.e. not
        # weighted by set size) and the montecarlo sampling
        for s_idx, s in enumerate(
            maybe_progress(
                power_set,
                progress,
                desc=f"Index {idx}",
                total=max_iterations,
                position=job_id,
            )
        ):
            values[s_idx, idx] = (u({idx}.union(s)) - u(s)) / math.comb(n - 1, len(s))

    return correction * values


def combinatorial_montecarlo_shapley(
    u: Utility,
    max_iterations: int,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    *,
    dist: PowerSetDistribution = PowerSetDistribution.WEIGHTED,
    progress: bool = False,
) -> Tuple[OrderedDict, Dict]:
    """Computes an approximate Shapley value using the combinatorial definition.

    :param u: utility
    :param max_iterations: total number of iterations (permutations) to use
    :param n_jobs: number of jobs across which to distribute the computation
    :param progress: true to plot progress bar
    :return: Tuple, with the first element being an ordered
        Dict of approximated Shapley values for the indices, the second being the
        montecarlo error related to each of them.
    """
    parallel_backend = init_parallel_backend(config)

    u_id = parallel_backend.put(u)

    iterations_per_job = max_iterations // n_jobs

    map_reduce_job: MapReduceJob["Utility", "NDArray"] = MapReduceJob(
        map_func=_combinatorial_montecarlo_shapley,
        reduce_func=np.concatenate,
        map_kwargs=dict(
            dist=dist,
            max_iterations=iterations_per_job,
            progress=progress,
        ),
        reduce_kwargs=dict(axis=0),
        config=config,
    )
    full_results = map_reduce_job(u_id, chunkify_inputs=False, n_jobs=n_jobs)[0]

    if np.any(np.isnan(full_results)):
        log.warning(
            f"Calculation returned {np.sum(np.isnan(full_results))} nan values out of {full_results.size}"
        )
    acc = np.nanmean(full_results, axis=0)
    acc_std = np.nanstd(full_results, axis=0) / np.sqrt(full_results.shape[0])
    sorted_shapley_values = sort_values(
        {u.data.data_names[i]: v for i, v in enumerate(acc)}
    )
    montecarlo_error = {u.data.data_names[i]: v for i, v in enumerate(acc_std)}
    return sorted_shapley_values, montecarlo_error
