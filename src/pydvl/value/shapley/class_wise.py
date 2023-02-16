import math
import warnings
from itertools import permutations
from scipy.special import binom
from typing import List, Sequence

import numpy as np
from numpy.typing import NDArray

from pydvl.utils import MapReduceJob, ParallelConfig, Utility, maybe_progress, powerset
from pydvl.utils.status import Status
from pydvl.value.results import ValuationResult

__all__ = ["class_wise_shapley"]


def class_wise_shapley(u: Utility, *, progress: bool = True) -> ValuationResult:
    r"""Computes the class-wise Shapley value using the formulation with permutations:

    :param u: Utility object with model, data, and scoring function
    :param progress: Whether to display progress bars for each job.
    :return: Object with the data values.
    """

    if u.data.y_train.dtype != np.int:
        raise ValueError("The supplied dataset has to be a classification dataset.")
    n = len(u.data)
    # Note that the cache in utility saves most of the refitting because we
    # use frozenset for the input.
    if n > 10:
        warnings.warn(
            f"Large dataset! Computation requires {n}! calls to utility()",
            RuntimeWarning,
        )

    x = u.data.x_test
    y = u.data.y_test
    all_cls = np.unique(y)
    n = len(x)
    in_cls_acc = np.empty(len(u.data.indices))
    pred_y = u.model.predict(x)
    matched_y = pred_y == y

    for cls_num, cls_val in enumerate(all_cls):
        idx = np.where(y == cls_val)[0]
        in_cls_acc[cls_num] = np.sum(matched_y[idx]) / n

    out_of_cls_acc = np.sum(in_cls_acc) - in_cls_acc
    values = np.zeros(n)
    n_factor = n / (n - 1)

    for cls_num, cls_val in maybe_progress(
        enumerate(all_cls),
        progress,
        desc="Class",
        total=len(all_cls),
        position=0
    ):
        cls_idx = np.where(y == cls_val)[0]
        for p in maybe_progress(
            permutations(cls_idx),
            progress,
            desc="Permutation",
            total=math.factorial(len(cls_idx)),
            position=1
        ):
            exp_factor = np.exp(out_of_cls_acc[cls_val])
            exp_factor_red_set = np.exp(out_of_cls_acc[cls_val] * n_factor)
            in_cls_acc_red = (in_cls_acc * n - matched_y[cls_idx]) / (n - 1)
            values[cls_idx] = out_of_cls_acc[cls_num] * exp_factor - in_cls_acc_red * exp_factor_red_set
            values[cls_idx] /= binom(n - 1, len(p) - 1)

    return ValuationResult(
        algorithm="class_wise_shapley",
        status=Status.Converged,
        values=values,
        stderr=None,
        data_names=u.data.data_names,
    )


def invert_idx(p: np.ndarray, n: int) -> np.ndarray:
    mask = np.ones(n, np.bool)
    mask[p] = 0
    return np.arange(n)[mask]