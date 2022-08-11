"""
Simple implementation of DataShapley [1].

TODO:
 * don't copy data to all workers foolishly
 * use ray / whatever to distribute jobs to multiple machines
 * provide a single interface "montecarlo_shapley" for all methods with the
   parallelization backend as an argument ("multiprocessing", "ray", "serial")
 * shapley values for groups of samples
"""
import logging
import warnings
from collections import OrderedDict
from functools import partial
from time import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import ray
from joblib import Parallel, delayed

from valuation.reporting.scores import sort_values, sort_values_array
from valuation.utils import Utility
from valuation.utils.numeric import (
    PowerSetDistribution,
    get_running_avg_variance,
    random_powerset,
)
from valuation.utils.parallel import (
    MapReduceJob,
    available_cpus,
    make_nested_backend,
    map_reduce,
)
from valuation.utils.progress import maybe_progress

log = logging.getLogger(__name__)

__all__ = [
    "truncated_montecarlo_shapley",
    "permutation_montecarlo_shapley",
    "combinatorial_montecarlo_shapley",
]


@ray.remote
class ShapleyCoordinator:
    def __init__(
        self,
        score_tolerance: float,
        max_iterations: Optional[int] = None,
        progress: Optional[bool] = True,
    ):
        """
        :param score_tolerance: For every permutation, computation stops
            after the mean increase in performance is within score_tolerance of
            global_score
        :param progress: set to True to display progress bars
        """
        self.score_tolerance = score_tolerance
        self.progress = progress
        self.max_iterations = max_iterations
        self.workers_status = {}
        self._is_done = False
        self.total_iterations = 0

    def add_status(self, worker_id, shapley_stats):
        self.workers_status[worker_id] = shapley_stats

    @property
    def is_done(self):
        return self._is_done

    def get_results(self):
        dvl_values = []
        dvl_std = []
        for _, worker_data in self.workers_status.iteritems():
            dvl_values.append(worker_data["dvl_values"])
            dvl_std.append(worker_data["dvl_std"])
            self.total_iterations += worker_data["num_iter"]

        if np.any(dvl_std == 0):
            log.warning(
                "Found std=0 for some workers. Weighted average will return nans."
            )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            num_workers = len(dvl_values)
            dvl_value = np.average(dvl_values, axis=0, weights=1 / dvl_std)
            dvl_std = np.average(dvl_std, axis=0, weights=1 / dvl_std) / np.sqrt(
                num_workers
            )
        return dvl_value, dvl_std

    def check_status(self):
        dvl_value, dvl_std = self.get_results()
        std_to_val_ratio = dvl_std / dvl_value
        if np.all(std_to_val_ratio < self.score_tolerance):
            self._is_done = True
        if (
            self.max_iterations is not None
            and self.total_iterations > self.max_iterations
        ):
            self._is_done = True


@ray.remote
class ShapleyWorker:
    """A worker. It should work."""

    def __init__(
        self,
        u: Utility,
        coordinator: ShapleyCoordinator,
        worker_id: int,
        progress: bool,
        update_frequency: int = 30,
    ):
        """
        :param u: Utility object with model, data, and scoring function
        :param global_score: Score of the model on an independent test set
        :param score_tolerance: For every permutation, computation stops
         after the mean increase in performance is within score_tolerance of
         global_score
        :param min_scores: Use so many of the last samples for a permutation
         in order to compute the moving average of scores.
        :param progress: set to True to display progress bars

        """

        self.u = u
        self.num_samples = len(self.u.data)
        self.worker_id
        self.pbar = maybe_progress(
            self.num_samples,
            progress,
            position=worker_id,
            desc=f"Worker {worker_id}",
        )
        self.coordinator = coordinator
        self.update_frequency = update_frequency
        self.avg_dvl: np.ndarray = None
        self.var_dvl: np.ndarray = None
        self.permutation_count = 0

    def run(self):
        """ """
        while not self.coordinator.is_done:
            start_time = time()
            elapsed_time = 0
            while elapsed_time < self.update_frequency:
                values = _permutation_montecarlo_shapley(self.u, max_iterations=1)[0]
                if self.avg_dvl is None:
                    self.avg_dvl = values
                    self.std_dvl = np.array([0] * len(self.avg_dvl))
                    self.worker_count = 1
                else:
                    self.avg_dvl, self.var_dvl = get_running_avg_variance(
                        self.avg_dvl, self.var_dvl, values, self.permutation_count
                    )
                    self.permutation_count += 1
                elapsed_time = time() - start_time
            self.coordinator.add_status(
                self.worker_id,
                {
                    "dvl_values": self.avg_dvl,
                    "dvl_std": np.sqrt(self.var_dvl),
                    "num_iter": self.permutation_count,
                },
            )


def truncated_montecarlo_shapley(
    u: Utility,
    score_tolerance: float,
    coordinator_update_frequency: int = 60,
    worker_update_frequency: int = 30,
    max_iterations: Optional[int] = None,
    num_workers: Optional[int] = None,
    progress: bool = False,
) -> Tuple[OrderedDict, List[int]]:
    """MonteCarlo approximation to the Shapley value of data points.

    Instead of naively implementing the expectation, we sequentially add points
    to a dataset from a permutation. We compute scores and stop when model
    performance doesn't increase beyond a threshold.

    We keep sampling permutations and updating all values until the change in
    the moving average for all values falls below another threshold.

    :param u: Utility object with model, data, and scoring function
    :param bootstrap_iterations: Repeat the computation of `global_score`
        this many times to estimate its variance.
    :param min_scores: Use so many of the last scores in order to compute
        the moving average.
    :param score_tolerance: For every permutation, computation stops
        after the mean increase in performance is within score_tolerance*stddev
        of the bootstrapped score over the **test** set (i.e. we bootstrap to
        compute variance of scores, then use that to stop)
    :param min_values: complete at least these many value computations for
        every index and use so many of the last values for each sample index
        in order to compute the moving averages of values.
    :param value_tolerance: Stop sampling permutations after the first
        derivative of the means of the last min_steps values are within
        eps close to 0
    :param max_iterations: never run more than this many iterations (in
        total, across all workers: if num_workers = 100 and max_iterations =
        100, then each worker will run at most one job)
    :param num_workers: number of workers processing permutations. Typically set
        to available_cpus().
    :param run_id: for display purposes (location of progress bar)
    :param progress: set to True to use tqdm progress bars.
    :return: Dict of approximated Shapley values for the indices
    """
    if num_workers is None:
        num_workers = available_cpus()

    ray.init(num_cpus=num_workers)
    u_id = ray.put(u)
    try:
        coordinator = ShapleyCoordinator.remote(
            score_tolerance, max_iterations, progress
        )
        workers = [
            ShapleyWorker.remote(
                u=u_id,
                coordinator=coordinator,
                worker_id=worker_id,
                progress=progress,
                update_frequency=worker_update_frequency,
            )
            for worker_id in range(num_workers)
        ]
        for worker_id in range(num_workers):
            workers[worker_id].run.remote()
        start_time = time()
        is_done = False
        while not is_done:
            if (time() - start_time) % coordinator_update_frequency == 0:
                coordinator.check_status.remote()
                is_done = ray.get(coordinator.is_done.remote())
        dvl_values, dvl_std = ray.get(coordinator.get_results.remote())
        ray.shutdown()
        sorted_shapley_values = sort_values(
            {u.data.data_names[i]: v for i, v in enumerate(dvl_values)}
        )
        montecarlo_error = {u.data.data_names[i]: v for i, v in enumerate(dvl_std)}
        return sorted_shapley_values, montecarlo_error

    except KeyboardInterrupt as e:
        ray.shutdown()
        raise e


def _permutation_montecarlo_shapley(
    u: Utility, max_permutations: int, progress: bool = False, job_id: int = 1
):
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
    u: Utility, max_iterations: int, num_jobs: int = 1, progress: bool = False
) -> Tuple[OrderedDict, None]:

    fun = partial(_permutation_montecarlo_shapley, u, max_iterations, progress)
    # TODO move to map_reduce as soon as it is fixed
    backend = make_nested_backend("loky")()
    results = Parallel(n_jobs=num_jobs, backend=backend)(
        delayed(fun)(job_id=j + 1) for j in range(num_jobs)
    )
    full_results = np.concatenate(results, axis=0)
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


def combinatorial_montecarlo_shapley(
    u: Utility, max_iterations: int, num_jobs: int = 1, progress: bool = False
) -> Tuple[OrderedDict, None]:
    """Computes an approximate Shapley value using the combinatorial
    definition and MonteCarlo samples.
    """
    n = len(u.data)

    dist = PowerSetDistribution.WEIGHTED
    correction = 2 ** (n - 1) / n

    def fun(indices: np.ndarray, job_id: int) -> np.ndarray:
        """Given indices and job id, this funcion calculates random
        powersets of the training data and trains the model with them.
        Used for parallelisation, as argument for MapReduceJob.
        """
        values = np.zeros(shape=(len(indices), max_iterations))
        for idx in indices:
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
                values[idx, s_idx] = (u({idx}.union(s)) - u(s)) / np.math.comb(
                    n - 1, len(s)
                )

        return correction * values

    job = MapReduceJob.from_fun(fun, np.concatenate)
    full_results = map_reduce(job, u.data.indices, num_jobs=num_jobs)[0]
    if np.any(np.isnan(full_results)):
        log.warning(
            f"Calculation returned {np.sum(np.isnan(full_results))} nan values out of {full_results.size}"
        )
    acc = np.nanmean(full_results, axis=1)
    acc_std = np.nanstd(full_results, axis=1) / np.sqrt(full_results.shape[1])
    sorted_shapley_values = sort_values(
        {u.data.data_names[i]: v for i, v in enumerate(acc)}
    )
    montecarlo_error = {u.data.data_names[i]: v for i, v in enumerate(acc_std)}
    return sorted_shapley_values, montecarlo_error
