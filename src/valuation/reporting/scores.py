import numpy as np

from collections import OrderedDict
from functools import partial
from itertools import chain
from joblib import Parallel, delayed
from typing import List, Mapping, Sequence
from tqdm import tqdm, trange

from valuation.utils import Dataset, SupervisedModel


def sort_values_array(values: np.ndarray) -> OrderedDict:
    vals = np.mean(values, axis=1)
    return OrderedDict(sorted(enumerate(vals), key=lambda x: x[1]))


def sort_values_history(values: Mapping[int, Sequence[float]]) -> OrderedDict:
    """ Sorts a dict of sample_id: [values] by the last item in each list. """
    return OrderedDict(sorted(values.items(), key=lambda x: x[1][-1]))


def sort_values(values: Mapping[int, float]) -> OrderedDict:
    """ Sorts a dict of sample_id: value_float by value. """
    return OrderedDict(sorted(values.items(), key=lambda x: x[1]))


def backward_elimination(model: SupervisedModel,
                         data: Dataset,
                         indices: List[int],
                         job_id: int = 0) -> List[float]:
    """ Computes model score (on a test set) after incrementally removing
    points from the training data.

    :param model: duh
    :param data: split Dataset
    :param indices: data points to remove in sequence. Retraining happens
                    after each removal.
    :param job_id: for progress bar positioning in parallel execution
    :return: List of scores
    """
    scores = []
    x, y = data.x_train, data.y_train
    for i in tqdm(indices[:-1], position=job_id,
                  desc=f"Backward elimination. Job {job_id}"):
        x = x[data.indices != i]
        y = y[data.indices != i]
        try:
            model.fit(x, y)
            scores.append(model.score(data.x_test, data.y_test))
        except:
            scores.append(np.nan)
    return scores


def forward_selection(model: SupervisedModel,
                      data: Dataset,
                      indices: List[int],
                      job_id: int = 0) -> List[float]:
    """ Computes model score (on a test set) incrementally adding
    points from training data.

    :param model: duh
    :param data: split Dataset
    :param indices: data points to add in sequence. Retraining happens
                    after each addition
    :param job_id: for progress bar positioning in parallel execution
    :return: List of scores
    """
    scores = []
    for i in trange(len(indices), position=job_id,
                    desc=f"Forward selection. Job {job_id}"):
        # FIXME: always train on at least a fraction of the indices
        x = data.x_train[indices[:i + 1]]
        y = data.y_train[indices[:i + 1]]
        try:
            model.fit(x, y)
            scores.append(model.score(data.x_test, data.y_test))
        except:
            scores.append(np.nan)
    return scores


def compute_fb_scores(model: SupervisedModel,
                      data: Dataset,
                      values: List[OrderedDict]) -> dict:
    """ Compute scores during forward selection and backward elimination of
     points, in parallel.

     :param values: OrderedDict of Shapley values, with keys sorted by
                    increasing value of the last item of the lists
     :param model: sklearn model implementing fit()
     :param data: split Dataset
    """
    num_runs = len(values)
    # TODO: report number of early stoppings
    bfun = partial(backward_elimination, model, data)
    backward_scores_delayed = chain(
            (delayed(bfun)(indices=list(v.keys()), job_id=i)
             for i, v in enumerate(values)),
            (delayed(bfun)(indices=list(reversed(v.keys())), job_id=i)
             for i, v in enumerate(values, start=num_runs)),
            (delayed(bfun)(
                    indices=np.random.permutation(
                            list(values[i % num_runs].keys())),
                    job_id=i)
             for i, _ in enumerate(values, start=2 * num_runs)))

    ffun = partial(forward_selection, model, data)
    forward_scores_delayed = chain(
            (delayed(ffun)(indices=list(v.keys()), job_id=i)
             for i, v in enumerate(values, start=3 * num_runs)),
            (delayed(ffun)(indices=list(reversed(v.keys())), job_id=i)
             for i, v in enumerate(values, start=4 * num_runs)),
            (delayed(ffun)(
                    indices=np.random.permutation(
                            list(values[i % num_runs].keys())),
                    job_id=i)
             for i, _ in enumerate(values, start=5 * num_runs)))

    all_scores = Parallel(n_jobs=6 * num_runs)(
            chain(backward_scores_delayed, forward_scores_delayed))

    results = {'all_values': values,
               # 'all_histories': all_histories,
               'backward_scores': all_scores[:num_runs],
               'backward_scores_reversed': all_scores[num_runs:2 * num_runs],
               'backward_random_scores': all_scores[2 * num_runs:3 * num_runs],
               'forward_scores': all_scores[3 * num_runs:4 * num_runs],
               'forward_scores_reversed': all_scores[4 * num_runs:5 * num_runs],
               'forward_random_scores': all_scores[5 * num_runs:6 * num_runs],
               'num_points': len(data)}

    return results
