"""
Refs:

[1] CS-Shapley: Class-wise Shapley Values for Data Valuation in Classification (https://arxiv.org/abs/2211.06800)
"""

# TODO rename function, and object refs, when transferred to pyDVL.

import numbers
import operator
from functools import reduce
from typing import Optional, Sequence, Tuple

import numpy as np
import yaml
from numpy._typing import NDArray
from valda.cs_shapley import cs_shapley

from pydvl.utils import MapReduceJob, ParallelConfig, SupervisedModel, Utility

__all__ = ["class_wise_shapley", "CSScorer"]

from sklearn.metrics import accuracy_score
from tqdm import tqdm

from pydvl.utils.numeric import random_powerset_group_conditional
from pydvl.value import StoppingCriterion, ValuationResult


def _estimate_in_out_cls_accuracy(
    model: SupervisedModel, x: np.ndarray, labels: np.ndarray, label: np.int_
) -> Tuple[float, float]:
    """
    Estimate the in and out of class accuracy as defined in [1], Equation 3.

    :param model: A model to be used for predicting the labels.
    :param x: The inputs to be used for measuring the accuracies. Has to match the labels.
    :param labels: The labels ot be used for measuring the accuracies. It is divided further by the passed label.
    :param label: The label of the class, which is currently viewed.
    :return: A tuple, containing the in class accuracy as well as the out of class accuracy.
    """
    n = len(x)
    y_pred = model.predict(x)
    label_set_match = labels == label
    label_set = np.where(label_set_match)[0]
    complement_label_set = np.where(~label_set_match)[0]

    acc_in_cls = (
        accuracy_score(labels[label_set], y_pred[label_set], normalize=False) / n
    )
    acc_out_of_cls = (
        accuracy_score(
            labels[complement_label_set], y_pred[complement_label_set], normalize=False
        )
        / n
    )
    return acc_in_cls, acc_out_of_cls


class CSScorer:
    """
    A Scorer object to be used along with the 'class_wise_shapley' function.
    """

    def __init__(self):
        self.default = 0.0
        self.range = (0.0, 1.0)
        self.label: Optional[np.int_] = None

    def __call__(
        self, model: SupervisedModel, x_test: np.ndarray, y_test: np.ndarray
    ) -> float:
        """
        Estimates the in and out of class accuracies and aggregated them into one float number.
        :param model: A model to be used for predicting the labels.
        :param x_test: The inputs to be used for measuring the accuracies. Has to match the labels.
        :param y_test:  The labels ot be used for measuring the accuracies. It is divided further by the passed label.
        :return: The aggregated number specified by 'in_cls_acc * exp(out_cls_acc)'
        """
        if self.label is None:
            raise ValueError(
                "Please set the label in the class first. By using o.label = <value>."
            )

        in_cls_acc, out_cls_acc = _estimate_in_out_cls_accuracy(
            model, x_test, y_test, self.label
        )
        return float(in_cls_acc * np.exp(out_cls_acc))


def _class_wise_shapley_worker(
    indices: Sequence[int],
    u: Utility,
    *,
    progress: bool = True,
    num_resample_complement_sets: int = 1,
    done: StoppingCriterion,
    eps: float = 1e-4,
) -> ValuationResult:
    r"""Computes the class-wise Shapley value using the formulation with permutations:

    :param u: Utility object with model, data, and scoring function. The scoring function has to be of type CSScorer.
    :param progress: Whether to display progress bars for each job.
    :param done: Criterion on when no new permutation shall be sampled.
    :param num_resample_complement_sets: How often the complement set shall be resampled for each permutation.
    :param eps: The threshold when updating using the truncated monte carlo estimator.
    :return: ValuationResult object with the data values.
    """

    if not all(map(lambda v: isinstance(v, numbers.Integral), u.data.y_train)):
        raise ValueError("The supplied dataset has to be a classification dataset.")

    if not isinstance(u.scorer, CSScorer):
        raise ValueError(
            "Please set CSScorer object as scorer object of utility. See scoring argument of Utility."
        )

    result = ValuationResult.zeros(
        algorithm="class_wise_shapley",
        indices=indices,
        data_names=u.data.data_names[indices],
    )

    x_train, y_train = u.data.get_training_data(indices)
    unique_labels = np.unique(y_train)
    pbar = tqdm(disable=not progress, position=0, total=100, unit="%")

    while not done(result):
        pbar.n = 100 * done.completion()
        pbar.refresh()

        for idx_label, label in enumerate(unique_labels):

            u.scorer.label = label
            active_elements = y_train == label
            label_set = np.where(active_elements)[0]
            complement_label_set = np.where(~active_elements)[0]
            label_set = indices[label_set]
            complement_label_set = indices[complement_label_set]

            _, complement_y_train = u.data.get_training_data(complement_label_set)
            permutation_label_set = np.random.permutation(label_set)

            for kl, subset_complement in enumerate(
                random_powerset_group_conditional(
                    complement_label_set,
                    complement_y_train,
                    n_samples=num_resample_complement_sets,
                )
            ):

                train_set = np.concatenate((label_set, subset_complement))
                final_score = u(train_set)
                prev_score = 0.0

                for i, _ in enumerate(label_set):

                    if np.abs(prev_score - final_score) < eps:
                        score = prev_score

                    else:
                        train_set = np.concatenate(
                            (permutation_label_set[: i + 1], subset_complement)
                        )
                        score = u(train_set)

                    marginal = score - prev_score
                    result.update(permutation_label_set[i], marginal)
                    prev_score = score

    return result


def class_wise_shapley(
    u: Utility,
    *,
    progress: bool = False,
    done: StoppingCriterion,
    eps: float = 1e-4,
    normalize_score: bool = True,
    n_jobs: int = 4,
    config: ParallelConfig = ParallelConfig(),
) -> ValuationResult:

    map_reduce_job: MapReduceJob[NDArray, ValuationResult] = MapReduceJob(
        u.data.indices,
        map_func=_class_wise_shapley_worker,
        reduce_func=lambda results: reduce(operator.add, results),
        map_kwargs=dict(u=u, done=done, progress=progress, eps=eps),
        n_jobs=n_jobs,
        config=config,
    )
    result = map_reduce_job()
    y_train = u.data.y_train
    unique_labels = np.unique(np.concatenate((y_train, u.data.y_test)))

    if normalize_score:

        for idx_label, label in enumerate(unique_labels):

            u.scorer.label = label
            active_elements = y_train == label
            label_set = np.where(active_elements)[0]

            u.model.fit(u.data.x_train, u.data.y_train)
            in_cls_acc, _ = _estimate_in_out_cls_accuracy(
                u.model, u.data.x_test, u.data.y_test, label
            )

            sigma = np.sum(result.values[label_set])
            result.values[label_set] *= in_cls_acc / sigma

    return result
