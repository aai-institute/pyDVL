r"""
This module implements the method described in Kwon and Zou, (2023).[^1]

Data-OOB value is tailored to bagging models. It defines a data point's value as
the average loss of the estimators which were not fit on it.

As such it is not a semi-value, and it is not based on marginal contributions.

!!! info
    For details on the method and a discussion on how and whether to use it by
    bagging models a posteriori, see the [main
    documentation][data-oob-intro].


## References

[^1]: <a name="kwon_dataoob_2023"></a> Kwon, Yongchan, and James Zou. [Data-OOB:
      Out-of-bag Estimate as a Simple and Efficient Data
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
from typing_extensions import Self

from pydvl.utils.array import ArrayT, to_numpy
from pydvl.valuation.base import Valuation
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.types import BaggingModel, PointwiseScore

T = TypeVar("T", bound=np.number)

logger = logging.getLogger(__name__)


class DataOOBValuation(Valuation):
    """Computes Data Out-Of-Bag values.

    This class implements the method described in Kwon and Zou,
    (2023)<sup><a href="kwon_dataoob_2023">1</a></sup>.

    Args:
        model: A fitted bagging model. Bagging models in sklearn include
            [[BaggingClassifier]], [[BaggingRegressor]], [[IsolationForest]], RandomForest*,
            ExtraTrees*, or any model which defines an attribute `estimators_` and uses
            bootstrapped subsamples to compute predictions.
        score: A callable for point-wise comparison of true values with the predictions.
            If `None`, uses point-wise accuracy for classifiers and negative $l_2$
            distance for regressors.

    !!! tip "New in version 0.11.0"
        Added (partial) support for PyTorch tensors.
    """

    algorithm_name: str = "Data-OOB"

    def __init__(
        self,
        model: BaggingModel,
        score: PointwiseScore | None = None,
    ):
        super().__init__()
        self.model = model
        self.score = score
        self.algorithm_name = f"Data-OOB-{str(self.model)}"

    def fit(self, data: Dataset, continue_from: ValuationResult | None = None) -> Self:
        """Compute the Data-OOB values.

        This requires the bagging model passed upon construction to be fitted.

        Args:
            data: Data for which to compute values
            continue_from: A previously computed valuation result to continue from.

        Returns:
            The fitted object.
        """

        self._result = self._init_or_check_result(data, continue_from)

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
        elif isinstance(
            self.model,
            (
                RandomForestClassifier,
                RandomForestRegressor,
                ExtraTreesClassifier,
                ExtraTreesRegressor,
                IsolationForest,
            ),
        ):
            n_samples_bootstrap = _get_n_samples_bootstrap(
                len(data), self.model.max_samples
            )
            unsampled_indices = [
                _generate_unsampled_indices(
                    est.random_state, len(data.indices), n_samples_bootstrap
                )
                for est in estimators
            ]
        else:
            raise ValueError(
                "The model has to be an sklearn-compatible bagging model, including "
                "BaggingClassifier, BaggingRegressor, IsolationForest, RandomForest*, "
                "and ExtraTrees*, \n"
                "or it must implement pydvl.valuation.types.BaggingModel.\n"
                f"Was: {type(self.model)}"
            )

        for est, oob_indices in zip(estimators, unsampled_indices):
            subset = data[oob_indices].data()
            score_array = self.score(y_true=subset.y, y_pred=est.predict(subset.x))
            self._result += ValuationResult(
                algorithm=str(self),
                indices=oob_indices,
                names=data[oob_indices].names,
                values=score_array,
                counts=np.ones_like(score_array, dtype=data.indices.dtype),
            )
        return self


def point_wise_accuracy(y_true: ArrayT, y_pred: ArrayT) -> NDArray[np.float64]:
    """Point-wise accuracy, or 0-1 score between two arrays.

    Higher is better.

    Args:
        y_true: Array of true values (e.g. labels)
        y_pred: Array of estimated values (e.g. model predictions)

    Returns:
        Array with point-wise 0-1 accuracy between labels and model predictions
    """
    return np.array(to_numpy(y_pred) == to_numpy(y_true), dtype=np.float64)


def neg_l2_distance(y_true: ArrayT, y_pred: ArrayT) -> NDArray[np.float64]:
    r"""Point-wise negative $l_2$ distance between two arrays.

    Higher is better.

    Args:
        y_true: Array of true values (e.g. labels)
        y_pred: Array of estimated values (e.g. model predictions)

    Returns:
        Array with point-wise negative $l_2$ distances between labels and model
        predictions
    """
    return -np.square(to_numpy(y_pred) - to_numpy(y_true), dtype=np.float64)
