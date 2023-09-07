# Code written by @BastienZim
# Bastien Zimmermann - bastien.zimmermann@craft.ai
# Code implementation of data_oob from Kwon and Zou "Data-OOB: Out-of-bag Estimate as a Simple and Efficient Data Value" ICML 2023
# https://proceedings.mlr.press/v202/kwon23e.html


from __future__ import annotations


from pydvl.utils import Utility, maybe_progress
from pydvl.utils.status import Status
from pydvl.value.result import ValuationResult
import numpy as np
from sklearn.ensemble import BaggingClassifier

__all__ = ["compute_data_oob"]


def compute_data_oob(
    u: Utility, n_est: int = 10, max_samples: float = 0.8, *, progress: bool = True
) -> ValuationResult:
    r"""Computes Data out of bag values

    There is a need to tune n_est and max_samples jointly to ensure all samples are at least 1 time oob, otherwise the result could includ a nan value for that datum.

    :param u: Utility object with model, data, and scoring function
    :param n_est: Number of estimator used in the bagging procedure.
    :param progress: If True, display a progress bar
    :return: Object with the data values.
    """

    n_samples = len(u.data)
    all_counts, all_means, all_M2 = (
        np.zeros(n_samples),
        np.zeros(n_samples),
        np.zeros(n_samples),
    )

    bag = BaggingClassifier(u.model, max_samples=max_samples, n_estimators=n_est)
    bag.fit(u.data.x_train, u.data.y_train)
    #    print(bag.score(u.data.x_test, u.data.y_test))
    for est, samples in maybe_progress(
        zip(bag.estimators_, bag.estimators_samples_), progress
    ):  # The bottleneck is the bag fitting not this part so TQDM is not very useful here
        oob_idx = np.intersect1d(u.data.indices, np.unique(samples))
        all_counts[oob_idx] += 1
        accuracy = est.predict(u.data.x_train[oob_idx]) == u.data.y_train[oob_idx]
        result += ValuationResult(
            algorithm="data_oob",
            indices=oob_idx,
            values=accuracies,
            counts=np.ones_like(accuracies),
        )
    return result
