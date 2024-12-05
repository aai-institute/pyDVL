r"""
This module implements the method described in (Kwon and Zou, 2023)<sup><a
href="kwon_data_2023">1</a></sup>. It fits a bagging classifier or regressor to the data
with a given model as base estimator. A data point's value is the average loss of the
estimators which were not fit on it.

Let $w_{bj}\in Z$ be the number of times the j-th datum $(x_j, y_j)$ is selected
in the b-th bootstrap dataset.

$$
\psi((x_i,y_i),\Theta_B):=\frac{\sum_{b=1}^{B}\mathbb{1}(w_{bi}=0)T(y_i,
\hat{f}_b(x_i))}{\sum_{b=1}^{B}
\mathbb{1}(w_{bi}=0)},
$$

where $T: Y \times Y \rightarrow \mathbb{R}$ is a score function that represents the
goodness of a weak learner $\hat{f}_b$ at the i-th datum $(x_i, y_i)$.

!!! Warning
    This implementation is a placeholder and does not match exactly the method described
    in the paper.

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
from sklearn.base import is_classifier, is_regressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor

from pydvl.utils.types import Seed, SupervisedModel
from pydvl.valuation.base import Valuation
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.types import LossFunction

T = TypeVar("T", bound=np.number)

logger = logging.getLogger(__name__)


class DataOOBValuation(Valuation):
    """Computes Data Out-Of-Bag values.

    !!! tip
        `n_estimators` and `max_samples` must be tuned jointly to ensure that all
        samples are at least 1 time out-of-bag, otherwise the result could include a
        NaN value for that datum.

    Args:
        data: dataset
        model:
        n_estimators: Number of estimators used in the bagging procedure.
        max_samples: The fraction of samples to draw to train each base estimator.
        loss: A function taking as parameters model prediction and corresponding
            data labels(y_true, y_pred) and returning an array of point-wise errors.
        seed: Either an instance of a numpy random number generator or a seed
            for it.

    Returns:
        Object with the data values.

    FIXME: this is an extended pydvl implementation of the Data-OOB valuation method
      which just bags whatever model is passed to it. The paper only considers bagging
      models as input.
    """

    def __init__(
        self,
        model: SupervisedModel,
        n_estimators: int,
        max_samples: float = 0.8,
        loss: LossFunction | None = None,
        seed: Seed | None = None,
    ):
        super().__init__()
        self.model = model
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.loss = loss
        self.rng = np.random.default_rng(seed)

    def fit(self, data: Dataset):
        # TODO: automate str representation for all Valuations
        algorithm_name = f"Data-OOB-{str(self.model)}"
        self.result = ValuationResult.zeros(
            algorithm=algorithm_name,
            indices=data.indices,
            data_names=data.names,
        )

        random_state = np.random.RandomState(self.rng.bit_generator)

        if is_classifier(self.model):
            logger.info(f"Training BaggingClassifier using {self.model}")
            bag = BaggingClassifier(
                self.model,
                n_estimators=self.n_estimators,
                max_samples=self.max_samples,
                random_state=random_state,
            )
            if self.loss is None:
                self.loss = point_wise_accuracy
        elif is_regressor(self.model):
            logger.info(f"Training BaggingRegressor using {self.model}")
            bag = BaggingRegressor(
                self.model,
                n_estimators=self.n_estimators,
                max_samples=self.max_samples,
                random_state=random_state,
            )
            if self.loss is None:
                self.loss = neg_l2_distance
        else:
            raise Exception(
                "Model has to be a classifier or a regressor in sklearn format."
            )

        bag.fit(data.x, data.y)
        for est, samples in zip(bag.estimators_, bag.estimators_samples_):
            oob_idx = np.setxor1d(data.indices, np.unique(samples))
            array_loss = self.loss(
                y_true=data.y[oob_idx],
                y_pred=est.predict(data.x[oob_idx]),
            )
            self.result += ValuationResult(
                algorithm=algorithm_name,
                indices=oob_idx,
                values=array_loss,
                counts=np.ones_like(array_loss, dtype=data.indices.dtype),
            )


def point_wise_accuracy(y_true: NDArray[T], y_pred: NDArray[T]) -> NDArray[T]:
    r"""Point-wise 0-1 loss between two arrays

    Args:
        y_true: Array of true values (e.g. labels)
        y_pred: Array of estimated values (e.g. model predictions)

    Returns:
        Array with point-wise 0-1 losses between labels and model predictions
    """
    return np.array(y_pred == y_true, dtype=y_pred.dtype)


def neg_l2_distance(y_true: NDArray[T], y_pred: NDArray[T]) -> NDArray[T]:
    r"""Point-wise negative $l_2$ distance between two arrays

    Args:
        y_true: Array of true values (e.g. labels)
        y_pred: Array of estimated values (e.g. model predictions)

    Returns:
        Array with point-wise negative $l_2$ distances between labels and model
        predictions
    """
    return -np.square(np.array(y_pred - y_true), dtype=y_pred.dtype)
