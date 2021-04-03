"""
Simple implementation of DataShapley [1].

TODO:
 * don't copy data to all workers foolishly
 * compute shapley values for groups of samples
 * use ray / whatever to distribute jobs to multiple machines
 * ...
"""

import queue
from time import time

import numpy as np
import pandas as pd
import multiprocessing as mp

from typing import Callable, Dict, Generator, List, Tuple
from collections import OrderedDict
from joblib import Parallel
from tqdm.auto import tqdm, trange
from valuation import _logger, Regressor


class Worker(mp.Process):
    """ A simple consumer worker using two queues.

     TODO: use shared memory to avoid copying data
     """

    def __init__(self,
                 worker_id: int,
                 tasks: mp.Queue,
                 results: mp.Queue,
                 converged: mp.Value,
                 model: Regressor,
                 global_score: float,
                 x_train: pd.DataFrame,
                 y_train: pd.DataFrame,
                 x_test: pd.DataFrame,
                 y_test: pd.DataFrame,
                 min_samples: int,
                 score_tolerance: float):
        # Mark as daemon, so we are killed when the parent exits (e.g. Ctrl+C)
        super().__init__(daemon=True)

        self.id = worker_id
        self.tasks = tasks
        self.results = results
        self.converged = converged
        self.early_stops = []  # TODO: use mp.Value and report
        self.model = model
        self.global_score = global_score
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.min_samples = min_samples
        self.score_tolerance = score_tolerance

        self.num_samples = len(x_train)

        # https://docs.python.org/3/library/multiprocessing.html
        # #multiprocessing.Queue.cancel_join_thread
        # By default if a process is not the creator of the queue then on exit
        # it will attempt to join the queue’s background thread: without this
        # call, running processes cannot be joined until all the data they have
        # put in the queue has been processed. Any Worker posting results after
        # the main loop in montecarlo_shapley has exited, will block upon join()
        # FIXME: this doesn't remove the need for timeout in worker.join()
        self.results.cancel_join_thread()

    def run(self):
        while True:
            try:
                # FIXME: I expected get() on a close()d queue to raise, but it
                #  doesn't. Instead we exit if no tasks are waiting, but this
                #  could fail if we don't have time in the main process before
                #  to put new ones in the queue.
                task = self.tasks.get(timeout=0.1)
            except queue.Empty:
                return
            result = self._run(task)
            self.results.put(result)

    def _run(self, permutation: List[int]) -> Dict[int, float]:
        """ """
        # scores[0] is the value of training on the empty set.
        scores = np.zeros(len(permutation) + 1)
        pbar = tqdm(total=self.num_samples, position=self.id,
                    desc=f"{self.name}", leave=False)

        pbar.reset()
        early_stop = None
        for j, index in enumerate(permutation, start=1):
            if self.converged.value >= self.num_samples:
                break

            mean_last_score = scores[max(j - self.min_samples, 0):j].mean()
            pbar.set_postfix_str(
                    f"last {self.min_samples} scores: {mean_last_score:.2e}")

            # Stop if last min_samples have an average mean below threshold:
            if abs(self.global_score - mean_last_score) < self.score_tolerance:
                if early_stop is None:
                    early_stop = j
                scores[j] = scores[j - 1]
            else:
                x = self.x_train[self.x_train.index.isin(permutation[:j + 1])]
                y = self.y_train[self.y_train.index.isin(permutation[:j + 1])]
                try:
                    self.model.fit(x, y.values.ravel())
                    scores[j] = self.model.score(self.x_test,
                                                 self.y_test.values.ravel())
                except:
                    scores[j] = np.nan
            pbar.update()
        pbar.close()
        if early_stop is not None:
            n = len(permutation)
            self.early_stops.append((n - early_stop) / n)
        # TODO: return self.early_stops
        return {k: v for k, v in zip(permutation, scores[1:])}


def montecarlo_shapley(model: Regressor,
                       x_train: pd.DataFrame,
                       y_train: pd.DataFrame,
                       x_test: pd.DataFrame,
                       y_test: pd.DataFrame,
                       bootstrap_iterations: int,
                       min_samples: int,
                       score_tolerance: float,
                       min_values: int,
                       value_tolerance: float,
                       max_permutations: int,
                       num_workers: int,
                       run_id: int = 0) \
        -> Tuple[Dict[int, float], List[int]]:
    """ MonteCarlo approximation to the Shapley value of data points.

    Instead of naively implementing the expectation, we sequentially add points
    to a dataset from a permutation. We compute scores and stop when model
    performance doesn't increase beyond a threshold.

    We keep sampling permutations and updating all values until the change in
    the moving average for all values falls below another threshold.

        :param model: sklearn model / pipeline
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :param bootstrap_iterations: Repeat global_score computation this many
         times to estimate variance.
        :param min_samples: Use so many of the last samples for a permutation
         in order to compute the moving average of scores.
        :param score_tolerance: For every permutation, computation stops
         after the mean increase in performance is within score_tolerance*stddev
         of the bootstrapped score over the **test** set (i.e. we bootstrap to
         compute variance of scores, then use that to stop)
        :param min_values: complete at least these many value computations for
         every index and use so many of the last values for each sample index
         in order to compute the moving averages of values.
        :param value_tolerance: Stop sampling permutations after the first
         derivative of the means of the last min_steps values are within
         value_tolerance close to 0
        :param max_permutations: never run more than this many iterations (in
         total, across all workers: if num_workers = 100 and max_iterations =
         100
         then each worker will run at most one job)
        :return: Dict of approximated Shapley values for the indices

    """
    # if values is None:
    values = {i: [0.0] for i in x_train.index}
    # if converged_history is None:
    converged_history = []

    model.fit(x_train, y_train.values.ravel())
    _scores = []
    for _ in trange(bootstrap_iterations, desc="Bootstrapping"):
        sample = np.random.choice(x_test.index, len(x_test.index), replace=True)
        _scores.append(model.score(x_test.loc[sample],
                                   y_test.loc[sample].values.ravel()))
    global_score = float(np.mean(_scores))
    score_tolerance *= np.std(_scores)
    # import matplotlib.pyplot as plt
    # plt.hist(_scores, bins=40)
    # plt.savefig("bootstrap.png")

    iteration = 0
    num_samples = len(x_train)

    tasks_q = mp.Queue()
    results_q = mp.Queue()
    converged = mp.Value('i', 0)

    workers = [Worker(i + 1,
                      tasks_q, results_q, converged, model, global_score,
                      x_train, y_train, x_test, y_test,
                      min_samples, score_tolerance)
               for i in range(num_workers)]

    # Fill the queue before starting the workers or they will immediately quit
    for _ in range(2 * num_workers):
        tasks_q.put(np.random.permutation(x_train.index))

    for w in workers:
        w.start()

    pbar = tqdm(total=num_samples, position=0, desc=f"Run {run_id}. Converged")
    while converged.value < num_samples and iteration < max_permutations:
        res = results_q.get()

        for k, v in res.items():
            values[k].append(v)

        # Check empirical convergence of means (1st derivative ~= 0)
        last_values = np.array(list(values.values()))[:, - (min_values + 1):]
        d = np.diff(last_values, axis=1)
        converged.value = iteration >= min_values \
                          and np.isclose(d, 0.0, atol=value_tolerance).sum()
        converged_history.append(converged.value)

        # converged.value can decrease. reset() clears times, but if just update
        # the bar collapses. This is some hackery to fix that:
        pbar.n = converged.value
        pbar.last_print_n = converged.value
        pbar.last_print_t = time()
        # pbar.update(converged.value)
        pbar.refresh()
        permutation = np.random.permutation(x_train.index)
        tasks_q.put(permutation)
        iteration += 1

    tasks_q.close()  # Workers will eventually stop their computation loops
    pbar.set_description_str("Gathering pending results")
    pbar.total = results_q.qsize()
    pbar.reset()

    n = 0
    while True:
        try:
            res = results_q.get(block=False, timeout=0.1)
        except queue.Empty:
            break
        else:
            for k, v in res.items():
                values[k].append(v)
            n += 1
            pbar.update(n)
    pbar.close()

    for w in tqdm(workers, desc="Joining"):
        w.join()

    return OrderedDict(sorted(values.items(), key=lambda item: item[1])), \
           converged_history


################################################################################
# TODO: Legacy implementations. Remove after thoroughly testing the one above.


def serial_montecarlo_shapley(model: Regressor,
                              x_train: pd.DataFrame,
                              y_train: pd.DataFrame,
                              x_test: pd.DataFrame,
                              y_test: pd.DataFrame,
                              bootstrap_iterations: int,
                              min_samples: int,
                              score_tolerance: float,
                              min_steps: int,
                              value_tolerance: float,
                              max_iterations: int,
                              values: Dict[int, List[float]] = None,
                              converged_history: List[int] = None,
                              run_id: int = 0) \
        -> Tuple[Dict[int, float], List[int]]:
    """ MonteCarlo approximation to the Shapley value of data points using
    only one CPU.

    Instead of naively implementing the expectation, we sequentially add points
    to a dataset from a permutation. We compute scores and stop when model
    performance doesn't increase beyond a threshold.

    We keep sampling permutations and updating all values until the change in
    the moving average for all values falls below another threshold.

        :param model: sklearn model / pipeline
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :param bootstrap_iterations: Repeat global_score computation this many
         times to estimate variance.
        :param min_samples: Use so many of the last samples for a permutation
         in order to compute the moving average of scores.
        :param score_tolerance: For every permutation, computation stops
         after the mean increase in performance is within score_tolerance*stddev
         of the bootstrapped score over the **test** set (i.e. we bootstrap to
         compute variance of scores, then use that to stop)
        :param min_steps: complete at least these many value computations for
         every index and use so many of the last values for each sample index
         in order to compute the moving averages of values.
        :param value_tolerance: Stop sampling permutations after the first
         derivative of the means of the last min_steps values are within
         value_tolerance close to 0
        :param max_iterations: never run more than these many iterations
        :return: Dict of approximated Shapley values for the indices

    """
    if values is None:
        values = {i: [0.0] for i in x_train.index}

    if converged_history is None:
        converged_history = []

    model.fit(x_train, y_train.values.ravel())
    _scores = []
    for _ in trange(bootstrap_iterations, desc="Bootstrapping"):
        sample = np.random.choice(x_test.index, len(x_test.index), replace=True)
        _scores.append(
                model.score(x_test.loc[sample],
                            y_test.loc[sample].values.ravel()))
    global_score = np.mean(_scores)
    score_tolerance *= np.std(_scores)

    iteration = 0
    converged = 0
    num_samples = len(x_train)
    pbar = tqdm(total=num_samples, position=0, desc=f"Run {run_id}")
    while iteration < min_steps \
            or (converged < num_samples and iteration < max_iterations):
        permutation = np.random.permutation(x_train.index)

        # FIXME: need score for model fitted on empty dataset
        scores = np.zeros(len(permutation) + 1)
        pbar.reset()
        for j, index in enumerate(permutation):
            pbar.set_description_str(
                    f"Iteration {iteration}, converged {converged}: ")

            # Stop if last min_samples have an average mean below threshold:
            last_scores = scores[
                          max(j - min_samples, 0):j].mean() if j > 0 else 0.0
            pbar.set_postfix_str(
                    f"last {min_samples} scores: {last_scores:.2e}")
            pbar.update()
            if abs(global_score - last_scores) < score_tolerance:
                scores[j + 1] = scores[j]
            else:
                x = x_train[x_train.index.isin(permutation[:j + 1])]
                y = y_train[y_train.index.isin(permutation[:j + 1])]
                model.fit(x, y.values.ravel())
                scores[j + 1] = model.score(x_test, y_test.values.ravel())
            # Update mean value: mean_n = mean_{n-1} + (new_val - mean_{n-1})/n
            values[permutation[j]].append(
                    values[permutation[j]][-1]
                    + (scores[j + 1] - values[permutation[j]][-1]) / (j + 1))

        # Check empirical convergence of means (1st derivative ~= 0)
        last_values = np.array(list(values.values()))[:, - (min_steps + 1):]
        d = np.diff(last_values, axis=1)
        converged = np.isclose(d, 0.0, atol=value_tolerance).sum()
        converged_history.append(converged)

        iteration += 1
        pbar.refresh()

    pbar.close()
    # return OrderedDict(sorted({k: v[-1] for k, v in values.items()}.items(
    # ), key=lambda x: x[1]))

    return OrderedDict(sorted(values.items(), key=lambda item: item[1][-1])), \
           converged_history


def naive_montecarlo_shapley(model: Regressor,
                             utility: Callable[[np.ndarray, np.ndarray], float],
                             x_train: pd.DataFrame,
                             y_train: pd.DataFrame,
                             x_test: pd.DataFrame,
                             y_test: pd.DataFrame,
                             indices: List[int],
                             max_iterations: int,
                             tolerance: float = None,
                             job_id: int = 0) \
        -> Tuple[Dict[int, float], List[int]]:
    """ MonteCarlo approximation to the Shapley value of data points.

    This is a direct translation of the formula:

        Φ_i = E_π[ V(S^i_π ∪ {i}) - V(S^i_π) ],

    where π~Π is a draw from all possible n-permutations and S^i_π is the set
    of all indices before i in permutation π.

    As such it cannot take advantage of diminishing returns as an early stopping
    mechanism, hence the prefix "naive".

    Usage example
    =============
    n = len(x_train)
    chunk_size = 1 + int(n / num_jobs)
    indices = [[x_train.index[j] for j in range(i, min(i + chunk_size, n))]
               for i in range(0, n, chunk_size)]
    # NOTE: max_iterations should be a fraction of the number of permutations
    max_iterations = int(iterations_ratio * n)  # x% of training set size

    fun = partial(naive_montecarlo_shapley, model, model.score,
                  x_train, y_train, x_test, y_test,
                  max_iterations=max_iterations, tolerance=None)
    delayed_fun = list(delayed(fun)(indices=ii, job_id=i)
                       for i, ii in enumerate(indices))
    all_values = run_and_gather(delayed_fun, num_jobs, num_runs)

    Arguments
    =========
        :param model: sklearn model / pipeline
        :param utility: utility function (e.g. any score function to maximise)
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :param indices: List of indices in the dataset
        :param max_iterations: Set to e.g. len(x_train)/2: at 50% truncation the
            paper reports ~95% rank correlation with shapley values without
            truncation. (FIXME: dubious (by Hoeffding). Check the statement)
        :param tolerance: NOT IMPLEMENTED stop drawing permutations after delta
            in scores falls below this threshold
        :param job_id: for progress bar positioning
        :return: Dict of approximated Shapley values for the indices and dummy
                 list to conform to the generic interface in valuation.parallel
    """
    if tolerance is not None:
        raise NotImplementedError("Tolerance not implemented")

    values = {i: 0.0 for i in indices}

    for i in indices:
        pbar = trange(max_iterations, position=job_id, desc=f"Index {i}")
        mean_score = 0.0
        scores = []
        for _ in pbar:
            mean_score = np.nanmean(scores) if scores else 0.0
            pbar.set_postfix_str(f"mean: {mean_score:.2f}")
            permutation = np.random.permutation(x_train.index)
            # yuk... does not stop after match
            loc = np.where(permutation == i)[0][0]

            # HACK: some steps in a preprocessing pipeline might fail when there
            #   are too few rows. We set those as failures.

            try:
                model.fit(x_train.loc[permutation[:loc + 1]],
                          y_train.loc[permutation[:loc + 1]].values.ravel())
                score_with = utility(x_test, y_test.values.ravel())
                model.fit(x_train.loc[permutation[:loc]],
                          y_train.loc[permutation[:loc]].values.ravel())
                score_without = utility(x_test, y_test.values.ravel())
                scores.append(score_with - score_without)
            except:
                scores.append(np.nan)
        values[i] = mean_score
    return values, []

