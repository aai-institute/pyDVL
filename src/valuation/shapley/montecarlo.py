"""
Simple implementation of DataShapley [1].

TODO:
 * don't copy data to all workers foolishly
 * compute shapley values for groups of samples
 * use ray / whatever to distribute jobs to multiple machines
 * ...
"""

import numpy as np

from time import time
from typing import Dict, List, Tuple
from collections import OrderedDict
from tqdm.auto import tqdm, trange
from unittest.mock import Mock
from valuation.utils.parallel import Coordinator, InterruptibleWorker
from valuation.reporting.scores import sort_values, sort_values_history
from valuation.utils import Dataset, SupervisedModel,\
    vanishing_derivatives, utility


__all__ = ['parallel_montecarlo_shapley',
           'serial_montecarlo_shapley',
           'naive_montecarlo_shapley']


class ShapleyWorker(InterruptibleWorker):

    def __init__(self,
                 model: SupervisedModel,
                 data: Dataset,
                 global_score: float,
                 score_tolerance: float,
                 min_samples: int,
                 progress: bool,
                 **kwargs,
                 ):
        """
        :param model: sklearn model / pipeline
        :param data: split dataset
        :param global_score:
        :param score_tolerance: For every permutation, computation stops
         after the mean increase in performance is within score_tolerance*stddev
         of the bootstrapped score over the **test** set (i.e. we bootstrap to
         compute variance of scores, then use that to stop)
        :param min_samples: Use so many of the last samples for a permutation
         in order to compute the moving average of scores.
        :param progress: set to True to display progress bars

        """
        super().__init__(**kwargs)

        self.model = model
        self.global_score = global_score
        self.data = data
        self.min_samples = min_samples
        self.score_tolerance = score_tolerance
        self.num_samples = len(self.data)
        self.progress = progress

    def _run(self, permutation: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, int]:
        """ """
        # scores[0] is the value of training on the empty set.
        n = len(permutation)
        scores = np.zeros(n + 1)
        if self.progress:
            pbar = tqdm(total=self.num_samples, position=self.id,
                        desc=f"{self.name}", leave=False)
        else:
            pbar = Mock()  # HACK, sue me.
        pbar.reset()
        early_stop = None
        for j, index in enumerate(permutation, start=1):
            if self.aborted():
                break

            mean_last_score = scores[max(j - self.min_samples, 0):j].mean()

            # Stop if last min_samples have an average mean below threshold:
            if abs(self.global_score - mean_last_score) < \
                    self.score_tolerance:
                if early_stop is None:
                    early_stop = j
                scores[j] = scores[j - 1]
            else:
                scores[j] = utility(self.model, self.data,
                                    permutation[:j + 1])
            pbar.set_postfix_str(
                    f"last {self.min_samples} scores: "
                    f"{mean_last_score:.2e}")
            pbar.update()
        pbar.close()
        # FIXME: sending the permutation back is wasteful
        return permutation, scores, early_stop


def parallel_montecarlo_shapley(model: SupervisedModel,
                                data: Dataset,
                                bootstrap_iterations: int,
                                min_samples: int,
                                score_tolerance: float,
                                min_values: int,
                                value_tolerance: float,
                                max_permutations: int,
                                num_workers: int,
                                run_id: int = 0,
                                worker_progress: bool = False) \
        -> Tuple[OrderedDict, List[int]]:
    """ MonteCarlo approximation to the Shapley value of data points.

    Instead of naively implementing the expectation, we sequentially add points
    to a dataset from a permutation. We compute scores and stop when model
    performance doesn't increase beyond a threshold.

    We keep sampling permutations and updating all values until the change in
    the moving average for all values falls below another threshold.

        :param model: sklearn model / pipeline
        :param data: split dataset
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
    values = {i: [0.0] for i in data.ilocs}
    # if converged_history is None:
    converged_history = []

    model.fit(data.x_train, data.y_train.values.ravel())
    _scores = []
    for _ in trange(bootstrap_iterations, desc="Bootstrapping"):
        sample = np.random.choice(data.x_test.index, len(data.x_test.index),
                                  replace=True)
        _scores.append(model.score(data.x_test.loc[sample],
                                   data.y_test.loc[sample].values.ravel()))
    global_score = float(np.mean(_scores))
    score_tolerance *= np.std(_scores)
    # import matplotlib.pyplot as plt
    # plt.hist(_scores, bins=40)
    # plt.savefig("bootstrap.png")

    num_samples = len(data)
    converged = 0
    iteration = 1

    def process_result(result: Tuple):
        nonlocal iteration
        permutation, scores, early_stop = result
        for j, s, p in zip(permutation, scores[1:], scores[:-1]):
            values[j].append((iteration - 1) / iteration * values[j][-1]
                             + (s - p) / iteration)
        iteration += 1

    boss = Coordinator(processer=process_result)
    boss.instantiate(num_workers, ShapleyWorker,
                     model=model,
                     data=data,
                     global_score=global_score,
                     score_tolerance=score_tolerance,
                     min_samples=min_samples,
                     progress=worker_progress)

    # Fill the queue before starting the workers or they will immediately quit
    for _ in range(2 * num_workers):
        boss.put(np.random.permutation(data.ilocs))

    boss.start_shift()

    pbar = tqdm(total=num_samples, position=0, desc=f"Run {run_id}. Converged")
    while converged < num_samples and iteration <= max_permutations:
        boss.get_and_process()
        boss.put(np.random.permutation(data.ilocs))
        if iteration > min_values:
            converged = vanishing_derivatives(np.array(list(values.values())),
                                              min_values=min_values,
                                              value_tolerance=value_tolerance)
            converged_history.append(converged)

        # converged can decrease. reset() clears times, but if just call
        # update(), the bar collapses. This is some hackery to fix that:
        pbar.n = converged
        pbar.last_print_n, pbar.last_print_t = converged, time()
        pbar.refresh()

    boss.end_shift(pbar)
    pbar.close()

    return sort_values_history(values), converged_history


################################################################################
# TODO: Legacy implementations. Remove after thoroughly testing the one above.


def serial_montecarlo_shapley(model: SupervisedModel,
                              data: Dataset,
                              bootstrap_iterations: int,
                              min_samples: int,
                              score_tolerance: float,
                              min_steps: int,
                              value_tolerance: float,
                              max_iterations: int,
                              values: Dict[int, List[float]] = None,
                              converged_history: List[int] = None) \
        -> Tuple[OrderedDict, List[int]]:
    """ MonteCarlo approximation to the Shapley value of data points using
    only one CPU.

    Instead of naively implementing the expectation, we sequentially add points
    to a dataset from a permutation. We compute scores and stop when model
    performance doesn't increase beyond a threshold.

    We keep sampling permutations and updating all values until the change in
    the moving average for all values falls below another threshold.

        :param model: sklearn model / pipeline
        :param data: split Dataset
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
        :param values: Use these as starting point (retake computation)
        :param converged_history: Append to this history
        :return: Dict of approximated Shapley values for the indices

    """
    if values is None:
        values = {i: [0.0] for i in data.ilocs}

    if converged_history is None:
        converged_history = []

    model.fit(data.x_train, data.y_train.values.ravel())
    _scores = []
    for _ in trange(bootstrap_iterations, desc="Bootstrapping"):
        sample = np.random.choice(data.x_test.ilocs, len(data.x_test.ilocs),
                                  replace=True)
        _scores.append(
                model.score(data.x_test.iloc[sample],
                            data.y_test.iloc[sample].values.ravel()))
    global_score = np.mean(_scores)
    score_tolerance *= np.std(_scores)

    iteration = 0
    converged = 0
    num_samples = len(data)
    pbar = tqdm(total=num_samples, position=1, desc=f"MCShapley")
    while iteration < min_steps \
            or (converged < num_samples and iteration < max_iterations):
        permutation = np.random.permutation(data.ilocs)

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
                scores[j + 1] = utility(model, data, permutation[:j+1])
            # Update mean value: mean_n = mean_{n-1} + (new_val - mean_{n-1})/n
            values[permutation[j]].append(
                    values[permutation[j]][-1]
                    + (scores[j + 1] - values[permutation[j]][-1]) / (j + 1))

        # Check empirical convergence of means (1st derivative ~= 0)
        # last_values = np.array(list(values.values()))[:, - (min_steps + 1):]
        # d = np.diff(last_values, axis=1)
        # converged = np.isclose(d, 0.0, atol=value_tolerance).sum()
        converged = \
            vanishing_derivatives(np.array(list(values.values())),
                                  min_values=min_steps,
                                  value_tolerance=value_tolerance)
        converged_history.append(converged)
        iteration += 1
        pbar.refresh()

    pbar.close()

    return sort_values_history(values), converged_history


def naive_montecarlo_shapley(model: SupervisedModel,
                             data: Dataset,
                             indices: List[int],
                             max_iterations: int,
                             tolerance: float = None,
                             job_id: int = 0,
                             progress: bool = False) \
        -> Tuple[OrderedDict, List[int]]:
    """ MonteCarlo approximation to the Shapley value of data points.

    This is a direct translation of the formula:

        Φ_i = E_π[ V(S^i_π ∪ {i}) - V(S^i_π) ],

    where π~Π is a draw from all possible n-permutations and S^i_π is the set
    of all indices before i in permutation π.

    As such it cannot take advantage of diminishing returns as an early stopping
    mechanism, hence the prefix "naive".

    Usage example
    =============

    indices = list(range(len(data)))
    fun = partial(naive_montecarlo_shapley, model, data
                  max_iterations=max_iterations, tolerance=None)
    wrapped = parallel_wrap(fun, ("indices", indices), num_jobs=160)
    vals, hist = run_and_gather(wrapped, num_runs=10, progress_bar=True)

    Arguments
    =========
        :param model: sklearn model / pipeline
        :param data: split Dataset
        :param indices: subset of data.index to work on (useful for parallel
            computation with `parallel_wrap`)
        :param max_iterations: run these many. Compute (eps,delta) lower bound
            with `lower_bound_hoeffding`
        :param tolerance: NOT IMPLEMENTED stop drawing permutations after delta
            in scores falls below this threshold
        :param job_id: for progress bar positioning
        :param progress: whether to display a progress bar
        :return: Dict of approximated Shapley values for the indices and dummy
                 list to conform to the generic interface in valuation.parallel
    """
    if tolerance is not None:
        raise NotImplementedError("Tolerance not implemented")

    values = {i: 0.0 for i in indices}

    # FIXME: exchange the loops to avoid searching with where
    if progress:
        pbar = tqdm(indices, position=job_id, total=len(indices))
    else:
        pbar = indices
    for i in pbar:
        if progress:
            pbar.set_description(f"Index {i}")
        mean_score = 0.0
        scores = []
        for _ in range(max_iterations):
            # FIXME: compute mean incrementally mean_n+1=mean_n+1+(val-mean_n)/n
            mean_score = np.nanmean(scores) if scores else 0.0
            if progress:
                pbar.set_postfix_str(f"mean: {mean_score:.2f}")
            permutation = np.random.permutation(data.ilocs)
            # yuk... does not stop after match
            loc = np.where(permutation == i)[0][0]
            scores.append(utility(model, data, tuple(permutation[:loc + 1]))
                          - utility(model, data, tuple(permutation[:loc])))
        values[i] = mean_score
    return sort_values(values), []

