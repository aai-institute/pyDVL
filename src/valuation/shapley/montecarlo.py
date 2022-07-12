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
import os
from collections import OrderedDict
from time import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from valuation.reporting.scores import sort_values, sort_values_array
from valuation.utils import (
    Dataset,
    GroupedDataset,
    MemcachedConfig,
    Scorer,
    SupervisedModel,
    Utility,
    bootstrap_test_score,
    vanishing_derivatives,
)
from valuation.utils.numeric import PowerSetDistribution, random_powerset
from valuation.utils.parallel import (
    Coordinator,
    InterruptibleWorker,
    MapReduceJob,
    make_nested_backend,
    map_reduce,
)
from valuation.utils.progress import maybe_progress

log = logging.getLogger(__name__)

__all__ = [
    "truncated_montecarlo_shapley",
    "serial_truncated_montecarlo_shapley",
    "permutation_montecarlo_shapley",
    "combinatorial_montecarlo_shapley",
]


class ShapleyWorker(InterruptibleWorker):
    """A worker. It should work."""

    def __init__(
        self,
        u: Utility,
        global_score: float,
        score_tolerance: float,
        min_scores: int,
        progress: bool,
        **kwargs,
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
        super().__init__(**kwargs)

        self.u = u
        self.global_score = global_score
        self.min_scores = min_scores
        self.score_tolerance = score_tolerance
        self.num_samples = len(self.u.data)
        self.progress = progress
        self.pbar = maybe_progress(
            self.num_samples,
            self.progress,
            position=self.id,
            desc=f"Worker {self.id:02d}",
        )

    def _run(self, permutation: np.ndarray) -> Tuple[np.ndarray, Optional[int]]:
        """ """
        n = len(permutation)
        scores = np.zeros(n)

        self.pbar.reset()
        early_stop = None
        prev_score = 0.0
        last_scores = -np.inf * np.ones(self.min_scores)
        for i, idx in enumerate(permutation):
            if self.aborted():
                break
            # Stop if last min_scores have an average mean below threshold:
            mean_score = np.nanmean(last_scores)
            if np.isclose(mean_score, self.global_score, atol=self.score_tolerance):
                early_stop = i
                break
            score = self.u(permutation[: i + 1])
            last_scores[i % self.min_scores] = score  # order doesn't matter
            scores[idx] = score - prev_score
            prev_score = score
            self.pbar.set_postfix_str(
                f"last {self.min_scores} scores: " f"{mean_score:.2e}"
            )
            self.pbar.update()
        # self.pbar.close()
        return scores, early_stop


def truncated_montecarlo_shapley(
    u: Utility,
    bootstrap_iterations: int,
    min_scores: int,
    score_tolerance: float,
    min_values: int,
    value_tolerance: float,
    max_iterations: int,
    num_workers: int,
    run_id: int = 0,
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
    :param num_workers: number of workers processing permutations. Set e.g.
        to available_cpus()/t if each worker runs on t threads.
    :param run_id: for display purposes (location of progress bar)
    :param progress: set to True to use tqdm progress bars.
    :return: Dict of approximated Shapley values for the indices
    """
    n = len(u.data)
    values = np.zeros(n).reshape((-1, 1))
    converged_history = []

    mean, std = bootstrap_test_score(u, bootstrap_iterations)
    global_score = mean
    score_tolerance *= std

    def process_result(result: Tuple):
        nonlocal values
        scores, _ = result
        values = np.concatenate([values, scores.reshape((-1, 1))], axis=1)

    worker_params = {
        "u": u,
        "global_score": global_score,
        "score_tolerance": score_tolerance,
        "min_scores": min_scores,
        "progress": progress,
    }
    boss = Coordinator(processor=process_result)
    boss.instantiate(num_workers, ShapleyWorker, **worker_params)

    # Fill the queue before starting the workers or they will immediately quit
    for _ in range(2 * num_workers):
        boss.put(np.random.permutation(u.data.indices))

    boss.start()

    pbar = maybe_progress(n, progress, position=0, desc=f"Run {run_id}. Converged")
    converged = iteration = 0
    while converged < n and iteration <= max_iterations:
        boss.get_and_process()
        boss.put(np.random.permutation(u.data.indices))
        iteration += 1

        converged = vanishing_derivatives(
            values, min_values=min_values, atol=value_tolerance
        )
        converged_history.append(converged)
        # converged can decrease. reset() clears times, but if just call
        # update(), the bar collapses. This is some hackery to fix that:
        pbar.n = converged
        pbar.last_print_n, pbar.last_print_t = converged, time()
        pbar.refresh()

    boss.end(pbar)
    pbar.close()
    return sort_values_array(values), converged_history


# @checkpoint(["converged_history", "values"])
def serial_truncated_montecarlo_shapley(
    u: Utility,
    bootstrap_iterations: int,
    score_tolerance: float,
    min_steps: int,
    value_tolerance: float,
    max_iterations: int,
    progress: bool = False,
) -> Tuple[OrderedDict, List[int]]:
    """Truncated MonteCarlo method to compute Shapley values of data points
    using only one CPU.

    Instead of naively implementing the expectation, we sequentially add points
    to a dataset from a permutation. We compute scores and stop when model
    performance doesn't increase beyond a threshold.

    We keep sampling permutations and updating all values until the change in
    the moving average for all values falls below another threshold.

    :param u: Utility object with model, data, and scoring function
    :param bootstrap_iterations: Repeat global_score computation this many
        times to estimate variance.
    :param score_tolerance: For every permutation, computation stops
        after the mean increase in performance is within score_tolerance*stddev
        of the bootstrapped score over the **test** set (i.e. we bootstrap to
        compute variance of scores, then use that to stop)
    :param min_steps: use so many of the last values for each sample index
        in order to compute the moving averages of values.
    :param value_tolerance: Stop sampling permutations after the first
        derivative of the means of the last min_steps values are within
        eps close to 0
    :param max_iterations: never run more than these many iterations
    :param progress: whether to display progress bars
    :return: Dict of approximated Shapley values for the indices
    """
    n = len(u.data)
    all_marginals = np.zeros(n).reshape((-1, 1))

    converged_history = []

    m, s = bootstrap_test_score(u, bootstrap_iterations)
    global_score, eps = m, score_tolerance * s

    iteration = 1
    pbar = maybe_progress(
        range(max_iterations), progress, position=0, desc="Iterations"
    )
    pbar2 = maybe_progress(range(n), progress, position=1, desc="Converged")
    while iteration < max_iterations:
        permutation = np.random.permutation(u.data.indices)
        marginals = np.zeros(n)
        prev_score = 0.0
        last_scores = -np.inf * np.ones(min_steps)
        pbar2.reset(total=n)
        for i, j in enumerate(permutation):
            if np.isclose(np.nanmean(last_scores), global_score, atol=eps):
                continue
            # Careful: for some models there might be nans, e.g. for i=0 or i=1!
            score = u(permutation[: i + 1])
            last_scores[i % min_steps] = score  # order doesn't matter
            marginals[j] = score - prev_score
            prev_score = score

        all_marginals = np.concatenate(
            [all_marginals, marginals.reshape((-1, 1))], axis=1
        )

        converged = vanishing_derivatives(
            all_marginals, min_values=min_steps, atol=value_tolerance
        )
        converged_history.append(converged)
        pbar2.update(converged)
        if converged >= n:
            break
        iteration += 1
        pbar.update()

    pbar.close()
    values = np.nanmean(all_marginals, axis=1)
    values = {i: v for i, v in enumerate(values)}
    return sort_values(values), converged_history


def permutation_montecarlo_shapley(
    u: Utility, max_iterations: int, num_jobs: int = 1, progress: bool = False
) -> Tuple[OrderedDict, None]:
    def fun(job_id: int):
        n = len(u.data)
        values = np.zeros(shape=(max_iterations, n))
        pbar = maybe_progress(max_iterations, progress, position=job_id)
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


def create_utility(
    model: SupervisedModel,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    scoring: Optional[Scorer],
    data_groups: List = None,
    enable_cache: bool = True,
    cache_options: MemcachedConfig = None,
):
    if data_groups is None:
        dataset = Dataset(x_train, y_train, x_test, y_test)
    else:
        dataset = GroupedDataset(x_train, y_train, x_test, y_test, data_groups)
    return Utility(
        model, dataset, scoring, enable_cache=enable_cache, cache_options=cache_options
    )


def shapley_dval(
    u: Utility,
    iterations_per_job: int,
    num_jobs: int = 1,
    use_combinatorial=False,
):
    """Facade for montecarlo shapley methods. By default, it uses permutation_montecarlo_shapley"""
    if num_jobs == 1:
        progress = True
    else:
        progress = False
    if use_combinatorial:
        dval, dval_std = combinatorial_montecarlo_shapley(
            u, iterations_per_job, num_jobs, progress
        )
    else:
        dval, dval_std = permutation_montecarlo_shapley(
            u, iterations_per_job, num_jobs, progress
        )
    return pd.DataFrame(
        list(zip(dval.keys(), dval.values(), dval_std.values())),
        columns=["data_key", "shapley_dval", "dval_std"],
    )
