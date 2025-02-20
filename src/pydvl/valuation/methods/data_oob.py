r"""
This module implements the method described in (Kwon and Zou, 2023)<sup><a
href="kwon_data_2023">1</a></sup>.

A data point's Data-OOB value is defined for bagging models. It is the average loss of
the estimators which were not fit on it.

Let $w_{bj}\in Z$ be the number of times the j-th datum $(x_j, y_j)$ is selected
in the b-th bootstrap dataset. The Data-OOB value is computed as follows:

$$
\psi((x_i,y_i),\Theta_B):=\frac{\sum_{b=1}^{B}\mathbb{1}(w_{bi}=0)T(y_i,
\hat{f}_b(x_i))}{\sum_{b=1}^{B}
\mathbb{1}(w_{bi}=0)},
$$

where $T: Y \times Y \rightarrow \mathbb{R}$ is a score function that represents the
goodness of a weak learner $\hat{f}_b$ at the i-th datum $(x_i, y_i)$.

## References

[^1]: <a name="kwon_data_2023"></a> Kwon, Yongchan, and James Zou. [Data-OOB: Out-of-bag
      Estimate as a Simple and Efficient Data
      Value](https://proceedings.mlr.press/v202/kwon23e.html). In Proceedings of the
      40th International Conference on Machine Learning, 18135–52. PMLR, 2023.
"""

from __future__ import annotations

import logging
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from sklearn.base import is_classifier

# HACK: we use some private sklearn stuff to obtain the indices of the bootstrap samples
#  in RandomForest and ExtraTrees, which do not have the `estimators_samples_` attribute.
from sklearn.ensemble._forest import (
    _generate_unsampled_indices,
    _get_n_samples_bootstrap,
)
from sklearn.utils.validation import check_is_fitted

from pydvl.utils.types import BaggingModel, PointwiseScore
from pydvl.valuation.base import Valuation
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.result import ValuationResult

T = TypeVar("T", bound=np.number)

logger = logging.getLogger(__name__)


class DataOOBValuation(Valuation):
    """Computes Data Out-Of-Bag values.

    This class implements the method described in (Kwon and Zou,
    2023)<sup><a href="kwon_data_2023">1</a></sup>.

    Args:
        model: A fitted bagging model. Bagging models in sklearn include
            [[BaggingClassifier]], [[BaggingRegressor]], [[IsolationForest]], RandomForest*,
            ExtraTrees*, or any model which defines an attribute `estimators_` and uses
            bootstrapped subsamples to compute predictions.
        score: A callable for point-wise comparison of true values with the predictions.
            If `None`, uses point-wise accuracy for classifiers and negative $l_2$
            distance for regressors.

    Returns:
        Object with the data values.
    """

    def __init__(
        self,
        model: BaggingModel,
        score: PointwiseScore | None = None,
    ):
        super().__init__()
        self.model = model
        self.score = score

    def fit(self, data: Dataset):
        # TODO: automate str representation for all Valuations
        algorithm_name = f"Data-OOB-{str(self.model)}"
        self.result = ValuationResult.empty(
            algorithm=algorithm_name,
            indices=data.indices,
            data_names=data.names,
        )

        check_is_fitted(
            self.model,
            msg="The bagging model has to be fitted before calling the valuation method.",
        )

        # This should always be present after fitting
        try:
            estimators = self.model.estimators_  # type: ignore
        except AttributeError:
            raise ValueError(
                "The model has to be an sklearn-compatible bagging model, including "
                "BaggingClassifier, BaggingRegressor, IsolationForest, RandomForest*, "
                "and ExtraTrees*"
            )

        if self.score is None:
            self.score = (
                point_wise_accuracy if is_classifier(self.model) else neg_l2_distance
            )

        if hasattr(self.model, "estimators_samples_"):  # Bagging(Classifier|Regressor)
            unsampled_indices = [
                np.setxor1d(data.indices, np.unique(sampled))
                for sampled in self.model.estimators_samples_
            ]
        else:  # RandomForest*, ExtraTrees*, IsolationForest
            n_samples_bootstrap = _get_n_samples_bootstrap(
                len(data), self.model.max_samples
            )
            unsampled_indices = [
                _generate_unsampled_indices(
                    est.random_state, len(data.indices), n_samples_bootstrap
                )
                for est in estimators
            ]

        for est, oob_indices in zip(estimators, unsampled_indices):
            subset = data[oob_indices].data()
            score_array = self.score(y_true=subset.y, y_pred=est.predict(subset.x))
            self.result += ValuationResult(
                algorithm=algorithm_name,
                indices=oob_indices,
                names=data[oob_indices].names,
                values=score_array,
                counts=np.ones_like(score_array, dtype=data.indices.dtype),
            )


def point_wise_accuracy(y_true: NDArray[T], y_pred: NDArray[T]) -> NDArray[T]:
    """Point-wise accuracy, or 0-1 score between two arrays.

    Higher is better.

    Args:
        y_true: Array of true values (e.g. labels)
        y_pred: Array of estimated values (e.g. model predictions)

    Returns:
        Array with point-wise 0-1 accuracy between labels and model predictions
    """
    return np.array(y_pred == y_true, dtype=y_pred.dtype)


def neg_l2_distance(y_true: NDArray[T], y_pred: NDArray[T]) -> NDArray[T]:
    r"""Point-wise negative $l_2$ distance between two arrays.

    Higher is better.

    Args:
        y_true: Array of true values (e.g. labels)
        y_pred: Array of estimated values (e.g. model predictions)

    Returns:
        Array with point-wise negative $l_2$ distances between labels and model
        predictions
    """
    return -np.square(np.array(y_pred - y_true), dtype=y_pred.dtype)
