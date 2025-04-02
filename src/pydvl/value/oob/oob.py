"""
## References

[^1]: <a name="kwon_dataoob_2023"></a>Kwon et al.
[Data-OOB: Out-of-bag Estimate as a Simple and Efficient Data Value](https://proceedings.mlr.press/v202/kwon23e.html).
In: Published at ICML 2023

"""

from typing import Optional, TypeVar

import numpy as np
from numpy.typing import NDArray
from sklearn.base import is_classifier, is_regressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from tqdm.auto import tqdm

from pydvl.utils import Seed, Utility
from pydvl.utils.types import PointwiseScore
from pydvl.value.result import ValuationResult

__all__ = ["compute_data_oob"]

T = TypeVar("T", bound=np.number)


def compute_data_oob(
    u: Utility,
    *,
    n_est: int = 10,
    max_samples: float = 0.8,
    loss: Optional[PointwiseScore] = None,
    n_jobs: Optional[int] = None,
    seed: Optional[Seed] = None,
    progress: bool = False,
) -> ValuationResult:
    r"""Computes Data out of bag values

    This implements the method described in
    (Kwon and Zou, 2023)<sup><a href="#kwon_dataoob_2023">1</a></sup>.
    It fits several base estimators provided through u.model through a bagging
    process. The point value corresponds to the average loss of estimators which
    were not fit on it.

    $w_{bj}\in Z$ is the number of times the j-th datum $(x_j, y_j)$ is selected
    in the b-th bootstrap dataset.

    $$\psi((x_i,y_i),\Theta_B):=\frac{\sum_{b=1}^{B}\mathbb{1}(w_{bi}=0)T(y_i,
    \hat{f}_b(x_i))}{\sum_{b=1}^{B}
    \mathbb{1}
    (w_{bi}=0)}$$

    With:

    $$
    T: Y \times Y
    \rightarrow \mathbb{R}
    $$

    T is a score function that represents the goodness of a weak learner
    $\hat{f}_b$ at the i-th datum $(x_i, y_i)$.

    `n_est` and `max_samples` must be tuned jointly to ensure that all samples
    are at least 1 time out-of-bag, otherwise the result could include a NaN
    value for that datum.

    Args:
        u: Utility object with model, data, and scoring function.
        n_est: Number of estimator used in the bagging procedure.
        max_samples: The fraction of samples to draw to train each base
            estimator.
        loss: A function taking as parameters model prediction and corresponding
            data labels(y_true, y_pred) and returning an array of point-wise errors.
        n_jobs: The number of jobs to run in parallel used in the bagging
            procedure for both fit and predict.
        seed: Either an instance of a numpy random number generator or a seed
            for it.
        progress: If True, display a progress bar.

    Returns:
        Object with the data values.
    """
    rng = np.random.default_rng(seed)
    random_state = np.random.RandomState(rng.bit_generator)

    result: ValuationResult[np.int_, np.object_] = ValuationResult.empty(
        algorithm="data_oob", indices=u.data.indices, data_names=u.data.data_names
    )

    if is_classifier(u.model):
        bag = BaggingClassifier(
            u.model,
            n_estimators=n_est,
            max_samples=max_samples,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        if loss is None:
            loss = point_wise_accuracy
    elif is_regressor(u.model):
        bag = BaggingRegressor(
            u.model,
            n_estimators=n_est,
            max_samples=max_samples,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        if loss is None:
            loss = neg_l2_distance
    else:
        raise Exception(
            "Model has to be a classifier or a regressor in sklearn format."
        )

    bag.fit(u.data.x_train, u.data.y_train)

    for est, samples in tqdm(
        zip(bag.estimators_, bag.estimators_samples_), disable=not progress, total=n_est
    ):  # The bottleneck is the bag fitting not this part so TQDM is not very useful here
        oob_idx = np.setxor1d(u.data.indices, np.unique(samples))
        array_loss = loss(
            y_true=u.data.y_train[oob_idx],
            y_pred=est.predict(u.data.x_train[oob_idx]),
        )
        result += ValuationResult(
            algorithm="data_oob",
            indices=oob_idx,
            values=array_loss,
            counts=np.ones_like(array_loss, dtype=u.data.indices.dtype),
        )
    return result


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
