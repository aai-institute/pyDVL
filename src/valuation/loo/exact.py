import numpy as np

from typing import OrderedDict, Union
from valuation.reporting.scores import sort_values
from valuation.utils import Dataset
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor

LinearSmoother = Union[KernelRidge, GaussianProcessRegressor]


def smoothing_diag(model: LinearSmoother, x: np.ndarray) -> np.ndarray:
    if isinstance(model, KernelRidge):
        # FIXME: check this is the right formula AND avoid naive computation
        return np.diag(x @ np.linalg.inv(x.T @ x) @ x.T)
    elif isinstance(model, GaussianProcessRegressor):
        return model.kernel.diag(x)

    else:
        raise ValueError(f"Unknown model type {type(model)}")


def linear_smoother_loo(model: LinearSmoother,
                        data: Dataset,
                        # scoring: Optional[Scorer],
                        progress: bool = True) -> OrderedDict[int, float]:
    """ Computes the LOO score for each training point in the dataset."""
    # u = Utility(model, data, scoring)

    l = smoothing_diag(model, data.x_train)
    y = data.y_train
    yhat = model.predict(data.x_train)
    # FIXME: is this true?
    loss = np.square
    values = loss((y-yhat)/(1-l))

    return sort_values({idx: val for idx, val in enumerate(values)})

