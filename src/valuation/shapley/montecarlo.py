import logging
import math
import warnings
from collections import OrderedDict
from functools import partial
from time import time
from typing import Dict, Optional, Tuple

import numpy as np
import ray
from joblib import Parallel, delayed

from valuation.reporting.scores import sort_values
from valuation.utils import Utility
from valuation.utils.numeric import (
    PowerSetDistribution,
    get_running_avg_variance,
    random_powerset,
)
from valuation.utils.parallel import available_cpus
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
        score_tolerance: Optional[float] = None,
        max_iterations: Optional[int] = None,
        progress: Optional[bool] = True,
    ):
        """
         The coordinator has two main tasks: aggregating the results of the workers
         and terminating the process once a certain accuracy or total number of
         iterations is reached.

        :param score_tolerance: During calculation of shapley values, the
             coordinator will check if the median standard deviation over average
             score for each point's has dropped below score_tolerance.
             If so, the computation will be terminated.
         :param max_iterations: a sum of the total number of permutation is calculated
             If the current number of permutations has exceeded max_iterations, computation
             will stop.
         :param progress: True to plot progress, False otherwise.
        """
        if score_tolerance is None and max_iterations is None:
            raise ValueError(
                "At least one between score_tolerance and max_iterations must be passed,"
                "or the process cannot be stopped."
            )
        self.score_tolerance = score_tolerance
        self.progress = progress
        self.max_iterations = max_iterations
        self.workers_status: Dict[int, Dict[str, float]] = {}
        self._is_done = False
        self.total_iterations = 0

    def add_status(self, worker_id: int, shapley_stats: Dict):
        """
        Used by workers to report their status. It puts the results
        directly into the worker_status dictionary.

        :param worker_id: id of the worker
        :param shapley_stats: results of worker calculations
        """
        self.workers_status[worker_id] = shapley_stats

    # this should be a @property, but with it ray.get messes up
    def is_done(self):
        """
        Used by workers to check whether to terminate the process.
        """
        return self._is_done

    def get_results(self):
        """
        It aggregates the results of the different workers and returns
        the average and std of the values. If no worker has reported yet,
        it returns two empty arrays
        """
        dvl_values = []
        dvl_stds = []
        dvl_iterations = []
        self.total_iterations = 0
        if len(self.workers_status) == 0:
            return np.array([]), np.array([])
        for _, worker_data in self.workers_status.items():
            dvl_values.append(worker_data["dvl_values"])
            dvl_stds.append(worker_data["dvl_std"])
            dvl_iterations.append(worker_data["num_iter"])
            self.total_iterations += worker_data["num_iter"]

        num_workers = len(dvl_values)
        if num_workers > 1:
            dvl_value = np.average(dvl_values, axis=0, weights=dvl_iterations)
            dvl_std = np.sqrt(
                np.average(
                    (dvl_values - dvl_value) ** 2, axis=0, weights=dvl_iterations
                )
            ) / (num_workers - 1) ** (1 / 2)
        else:
            dvl_value = dvl_values[0]
            dvl_std = dvl_stds[0]
        return dvl_value, dvl_std

    def check_status(self):
        """
        It checks whether the accuracy of the calculation or the total number of iterations have crossed
        the set thresholds.
        If so, it sets the is_done label as True.
        """
        if len(self.workers_status) == 0:
            log.info("No worker has updated its status yet.")
        else:
            dvl_value, dvl_std = self.get_results()
            std_to_val_ratio = np.median(dvl_std) / np.median(dvl_value)
            if (
                self.score_tolerance is not None
                and std_to_val_ratio < self.score_tolerance
            ):
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
        The workers calculate the Shapley values using the permutation
        definition and report the results to the coordinator.

        :param u: Utility object with model, data, and scoring function
        :param coordinator: worker results will be pushed to this coordinator
        :param worker_id: id used for reporting through maybe_progress
        :param progress: set to True to report progress, else False
        :param update_frequency: interval in seconds among different updates to
            and from the coordinator
        """
        self.u = u
        self.num_samples = len(self.u.data)
        self.worker_id = worker_id
        self.pbar = maybe_progress(
            self.num_samples,
            progress,
            position=worker_id,
            desc=f"Worker {worker_id}",
        )
        self.coordinator = coordinator
        self.update_frequency = update_frequency
        self.avg_dvl: Optional[np.ndarray] = None
        self.var_dvl: Optional[np.ndarray] = None
        self.permutation_count = 0

    def run(self):
        """Runs the worker.
        It calls _permutation_montecarlo_shapley a certain number of times and calculates
        Shapley values on different permutations of the indices.
        After a number of seconds equal to update_frequency has passed, it reports the results
        to the coordinator. Before starting the next iteration, it checks the is_done flag, and if true
        terminates.
        """
        is_done = ray.get(self.coordinator.is_done.remote())
        while not is_done:
            start_time = time()
            elapsed_time = 0
            while elapsed_time < self.update_frequency:
                values = _permutation_montecarlo_shapley(self.u, max_permutations=1)[0]
                if np.any(np.isnan(values)):
                    log.warning(
                        "Nan values found in model scoring. Ignoring current permutation."
                    )
                    continue
                if self.avg_dvl is None:
                    self.avg_dvl = values
                    self.var_dvl = np.array([0] * len(self.avg_dvl))
                    self.permutation_count = 1
                else:
                    self.avg_dvl, self.var_dvl = get_running_avg_variance(
                        self.avg_dvl, self.var_dvl, values, self.permutation_count
                    )
                    self.permutation_count += 1
                elapsed_time = time() - start_time
            self.coordinator.add_status.remote(
                self.worker_id,
                {
                    "dvl_values": self.avg_dvl,
                    "dvl_std": np.sqrt(self.var_dvl)
                    / self.permutation_count ** (1 / 2),
                    "num_iter": self.permutation_count,
                },
            )
            is_done = ray.get(self.coordinator.is_done.remote())


def truncated_montecarlo_shapley(
    u: Utility,
    score_tolerance: Optional[float] = None,
    max_iterations: Optional[int] = None,
    num_workers: Optional[int] = None,
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
    :param num_workers: number of workers processing permutations. If None, it will be set
        to available_cpus().
    :param progress: set to True to use tqdm progress bars.
    :return: Tuple, with the first element being an ordered
        Dict of approximated Shapley values for the indices, the second being the
        montecarlo error related to each dvl value.
    """
    if num_workers is None:
        num_workers = available_cpus()

    ray.init(num_cpus=num_workers)
    u_id = ray.put(u)
    try:
        coordinator = ShapleyCoordinator.remote(  # type: ignore
            score_tolerance, max_iterations, progress
        )
        workers = [
            ShapleyWorker.remote(  # type: ignore
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
        last_update_time = time()
        is_done = False
        while not is_done:
            if time() - last_update_time > coordinator_update_frequency:
                coordinator.check_status.remote()
                is_done = ray.get(coordinator.is_done.remote())
                last_update_time = time()
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
    u: Utility, max_iterations: int, num_workers: int = 1, progress: bool = False
) -> Tuple[OrderedDict, Dict]:
    """Computes an approximate Shapley value using independent permutations of the indices.

    :param u: Utility object with model, data, and scoring function
    :param max_iterations: total number of iterations (permutations) to use
    :param num_workers: number of workers to distribute the jobs
    :param progress: true to plot progress bar
    :return: Tuple, with the first element being an ordered
        Dict of approximated Shapley values for the indices, the second being the
        montecarlo error related to each of them.
    """
    iterations_per_job = max_iterations // num_workers

    fun = partial(_permutation_montecarlo_shapley, u, iterations_per_job, progress)
    results = Parallel(n_jobs=num_workers)(
        delayed(fun)(job_id=j + 1) for j in range(num_workers)
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
    u: Utility, max_iterations: int, num_workers: int = 1, progress: bool = False
) -> Tuple[OrderedDict, Dict]:
    """Computes an approximate Shapley value using the combinatorial definition.

    :param u: utility
    :param max_iterations: total number of iterations (permutations) to use
    :param num_workers: number of workers to distribute the jobs
    :param progress: true to plot progress bar
    :return: Tuple, with the first element being an ordered
        Dict of approximated Shapley values for the indices, the second being the
        montecarlo error related to each of them.
    """
    n = len(u.data)

    dist = PowerSetDistribution.WEIGHTED
    correction = 2 ** (n - 1) / n
    iterations_per_job = max_iterations // num_workers

    def fun(indices: np.ndarray, job_id: int):
        """Given indices and job id, this funcion calculates random
        powerset of the indices and for each it trains and evaluates the model.
        Then, it returns the shapley values for each of the indices.
        """
        values = np.zeros(shape=(len(indices), iterations_per_job))
        for idx in indices:
            # Randomly sample subsets of full dataset without idx
            subset = np.setxor1d(u.data.indices, [idx], assume_unique=True)
            power_set = random_powerset(
                subset,
                dist=dist,
                max_subsets=iterations_per_job,
            )
            # Normalization accounts for a uniform dist. on powerset (i.e. not
            # weighted by set size) and the montecarlo sampling
            for s_idx, s in enumerate(
                maybe_progress(
                    power_set,
                    progress,
                    desc=f"Index {idx}",
                    total=iterations_per_job,
                    position=job_id,
                )
            ):
                values[idx, s_idx] = (u({idx}.union(s)) - u(s)) / math.comb(
                    n - 1, len(s)
                )

        return correction * values

    results = Parallel(n_jobs=num_workers)(
        delayed(fun)(u.data.indices, job_id=j + 1) for j in range(num_workers)
    )
    full_results = np.concatenate(results, axis=1)
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
