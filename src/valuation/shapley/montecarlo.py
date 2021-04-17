"""
Simple implementation of DataShapley [1].

TODO:
 * don't copy data to all workers foolishly
 * use ray / whatever to distribute jobs to multiple machines
 * provide a single interface "montecarlo_shapley" for all methods with the
   parallelization backend as an argument ("multiprocessing", "ray", "serial")
 * shapley values for groups of samples
"""
import numpy as np

from collections import OrderedDict
from sklearn.metrics import check_scoring
from time import time
from typing import List, Optional, Tuple
from valuation.reporting.scores import sort_values, sort_values_array
from valuation.utils.numeric import random_powerset
from valuation.utils.parallel import Coordinator, InterruptibleWorker
from valuation.utils.progress import maybe_progress
from valuation.utils import Dataset, SupervisedModel, vanishing_derivatives, \
    utility
from valuation.utils.types import Scorer

__all__ = ['truncated_montecarlo_shapley',
           'serial_truncated_montecarlo_shapley',
           'permutation_montecarlo_shapley']


def bootstrap_test_score(model: SupervisedModel,
                         data: Dataset,
                         scoring: Optional[Scorer],
                         bootstrap_iterations: int,
                         progress: bool = False) \
        -> Tuple[float, float]:
    """ That. Here for lack of a better place. """
    scorer = check_scoring(model, scoring)
    _scores = []
    model.fit(data.x_train, data.y_train)
    n_test = len(data.x_test)
    for _ in maybe_progress(range(bootstrap_iterations), progress,
                            desc="Bootstrapping"):
        sample = np.random.randint(low=0, high=n_test, size=n_test)
        score = scorer(model, data.x_test[sample], data.y_test[sample])
        _scores.append(score)

    return np.mean(_scores), np.std(_scores)


class ShapleyWorker(InterruptibleWorker):
    """ A worker. It should work. """

    def __init__(self,
                 model: SupervisedModel,
                 data: Dataset,
                 scoring: Optional[Scorer],
                 global_score: float,
                 score_tolerance: float,
                 min_scores: int,
                 progress: bool,
                 **kwargs,
                 ):
        """
        :param model: sklearn model / pipeline
        :param data: split dataset
        :param scoring:
        :param global_score:
        :param score_tolerance: For every permutation, computation stops
         after the mean increase in performance is within score_tolerance*stddev
         of the bootstrapped score over the **test** set (i.e. we bootstrap to
         compute variance of scores, then use that to stop)
        :param min_scores: Use so many of the last samples for a permutation
         in order to compute the moving average of scores.
        :param progress: set to True to display progress bars

        """
        super().__init__(**kwargs)

        self.model = model
        self.scorer = check_scoring(model, scoring)
        self.global_score = global_score
        self.data = data
        self.min_scores = min_scores
        self.score_tolerance = score_tolerance
        self.num_samples = len(self.data)
        self.progress = progress
        self.pbar = maybe_progress(range(len(data)), self.progress,
                                   total=(len(data)), position=self.id,
                                   desc=f"{self.name}")

    def _run(self, permutation: np.ndarray) -> Tuple[np.ndarray, Optional[int]]:
        """ """
        n = len(permutation)
        u = lambda x: utility(self.model, self.data, frozenset(x))
        scores = np.zeros(n)

        self.pbar.reset()
        early_stop = None
        prev_score = 0.0
        for j, index in enumerate(permutation):
            if self.aborted():
                break

            # Stop if last min_scores have an average mean below threshold:
            mean_last_score = scores[max(j - self.min_scores, 0):j].mean()
            if abs(self.global_score - mean_last_score) < self.score_tolerance:
                early_stop = j
                break
            score = u(permutation[:j + 1])
            scores[index] = score - prev_score
            prev_score = score
            self.pbar.set_postfix_str(f"last {self.min_scores} scores: "
                                 f"{mean_last_score:.2e}")
            self.pbar.update()
        # self.pbar.close()
        return scores, early_stop


def truncated_montecarlo_shapley(model: SupervisedModel,
                                 data: Dataset,
                                 scoring: Optional[Scorer],
                                 bootstrap_iterations: int,
                                 min_scores: int,
                                 score_tolerance: float,
                                 min_values: int,
                                 value_tolerance: float,
                                 max_iterations: int,
                                 num_workers: int,
                                 run_id: int = 0,
                                 progress: bool = False) \
        -> Tuple[OrderedDict, List[int]]:
    """ MonteCarlo approximation to the Shapley value of data points.

    Instead of naively implementing the expectation, we sequentially add points
    to a dataset from a permutation. We compute scores and stop when model
    performance doesn't increase beyond a threshold.

    We keep sampling permutations and updating all values until the change in
    the moving average for all values falls below another threshold.

        :param model: sklearn model / pipeline
        :param data: split dataset
        :param scoring: Scorer callable or string as in sklearn
        :param bootstrap_iterations: Repeat global_score computation this many
         times to estimate variance.
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
    n = len(data)
    values = np.zeros(n).reshape((-1, 1))
    converged_history = []

    mean, std = bootstrap_test_score(model, data, scoring, bootstrap_iterations)
    global_score = mean
    score_tolerance *= std

    def process_result(result: Tuple):
        nonlocal values
        scores, _ = result
        values = np.concatenate([values, scores.reshape((-1, 1))], axis=1)

    worker_params = {'model': model,
                     'data': data,
                     'scoring': scoring,
                     'global_score': global_score,
                     'score_tolerance': score_tolerance,
                     'min_scores': min_scores, 'progress': progress}
    boss = Coordinator(processor=process_result)
    boss.instantiate(num_workers, ShapleyWorker, **worker_params)

    # Fill the queue before starting the workers or they will immediately quit
    for _ in range(2 * num_workers):
        boss.put(np.random.permutation(data.indices))

    boss.start()

    pbar = maybe_progress(range(n), progress, total=n,
                          position=0, desc=f"Run {run_id}. Converged")
    converged = iteration = 0
    while converged < n and iteration <= max_iterations:
        boss.get_and_process()
        boss.put(np.random.permutation(data.indices))
        iteration += 1

        converged = vanishing_derivatives(values, min_values=min_values,
                                          eps=value_tolerance)
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
def serial_truncated_montecarlo_shapley(model: SupervisedModel,
                                        data: Dataset,
                                        bootstrap_iterations: int,
                                        score_tolerance: float,
                                        min_steps: int,
                                        value_tolerance: float,
                                        max_iterations: int,
                                        scoring: Scorer = None,
                                        progress: bool = False) \
        -> Tuple[OrderedDict, List[int]]:
    """ Truncated MonteCarlo method to compute Shapley values of data points
     using only one CPU.

    Instead of naively implementing the expectation, we sequentially add points
    to a dataset from a permutation. We compute scores and stop when model
    performance doesn't increase beyond a threshold.

    We keep sampling permutations and updating all values until the change in
    the moving average for all values falls below another threshold.

        :param model: sklearn model / pipeline
        :param data: split Dataset
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
        # :param values: Use these as starting point (retake computation)
        # :param converged_history: Append to this history
        :return: Dict of approximated Shapley values for the indices

    """
    n = len(data)
    all_marginals = np.zeros(n).reshape((-1, 1))
    u = lambda x: utility(model, data, frozenset(x), scoring=scoring)

    converged_history = []

    m, s = bootstrap_test_score(model, data, scoring, bootstrap_iterations)
    global_score, eps = m, score_tolerance * s

    iteration = 1
    pbar = maybe_progress(range(max_iterations), progress,
                          position=0, desc="Iterations")
    pbar2 = maybe_progress(range(n), progress,
                           position=1, desc="Converged")
    while iteration < max_iterations:
        permutation = np.random.permutation(data.indices)
        marginals = np.zeros(n)
        prev_score = 0.0
        last_scores = -np.inf*np.ones(min_steps)
        pbar2.reset(total=n)
        for i, j in enumerate(permutation):
            if np.isclose(np.nanmean(last_scores), global_score, atol=eps):
                continue
            # Careful: for some models there might be nans, e.g. for i=0 or i=1!
            score = u(permutation[:i+1])
            last_scores[i % min_steps] = score  # order doesn't matter
            marginals[j] = score - prev_score
            prev_score = score

        all_marginals = \
            np.concatenate([all_marginals, marginals.reshape((-1, 1))], axis=1)

        converged = \
            vanishing_derivatives(all_marginals, min_values=min_steps,
                                  eps=value_tolerance)
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


def permutation_montecarlo_shapley(model: SupervisedModel,
                                   data: Dataset,
                                   max_iterations: int,
                                   scoring: Scorer = None,
                                   job_id: int = 0,
                                   progress: bool = False) \
        -> Tuple[OrderedDict, List[int]]:
    """
    FIXME: the sum of values tends to cancel out, even though the ranking is ok
    """
    n = len(data)
    values = np.zeros(n).reshape((-1, 1))
    u = lambda x: utility(model, data, frozenset(x), scoring=scoring)
    prev_score = 0.0
    pbar = maybe_progress(max_iterations, progress, position=job_id)
    for _ in pbar:
        permutation = np.random.permutation(data.indices)
        marginals = np.zeros((n, 1))
        for i, idx in enumerate(permutation):
            score = u(permutation[:i + 1])
            marginals[idx] = score - prev_score
            prev_score = score
        values = np.concatenate([values, marginals], axis=1)
    # Careful: for some models there might be nans, e.g. for i=0 or i=1!
    values = np.nanmean(values, axis=1)
    return sort_values({i: v for i, v in enumerate(values)}), []


def combinatorial_montecarlo_shapley(model: SupervisedModel,
                                     data: Dataset,
                                     max_iterations: int,
                                     scoring: Scorer = None,
                                     indices: List[int] = None,
                                     job_id: int = 0,
                                     progress: bool = False) \
        -> Tuple[OrderedDict, None]:
    """ Computes an approximate Shapley value using the combinatorial
    definition and MonteCarlo samples.

    FIXME: this seems to have very high variance! It might help to implement
     importance sampling with more weight for smaller set sizes, where value
      contributions are likely to be higher.
    """
    n = len(data)
    if not indices:
        indices = data.indices
    values = np.zeros(len(indices))
    u = lambda x: utility(model, data, frozenset(x), scoring=scoring)
    for i, idx in enumerate(indices):
        # Randomly sample subsets of full dataset without idx
        subset = np.setxor1d(data.indices, [idx], assume_unique=True)
        power_set = \
            enumerate(random_powerset(subset, max_subsets=max_iterations))
        ut = 0.0
        for j, s in maybe_progress(power_set, progress, desc=f"Index {idx}",
                                   total=max_iterations, position=job_id):
            ut += (u({i}.union(s)) - u(s)) / np.math.comb(n - 1, len(s))
        # Normalization accounts for uniform dist. on powerset and montecarlo
        values[i] = 2**(n-1) * ut / max_iterations
    values /= n  # careful to use the right factor!
    return sort_values({i: v for i, v in zip(indices, values)}), None
