"""
Contains all functions for closed form solution of influences for standard linear regression.
"""
from typing import TYPE_CHECKING, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression

from ..utils.numeric import (
    linear_regression_analytical_derivative_d2_theta,
    linear_regression_analytical_derivative_d_theta,
    linear_regression_analytical_derivative_d_x_d_theta,
)
from .general import InfluenceType

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "compute_linear_influences",
    "influences_up_linear_regression_analytical",
    "influences_perturbation_linear_regression_analytical",
]


def compute_linear_influences(
    x: "NDArray",
    y: "NDArray",
    x_test: "NDArray",
    y_test: "NDArray",
    influence_type: InfluenceType = InfluenceType.Up,
):
    """
    Calculate the linear influences of the training set onto the validation set assuming a linear model Ax+b=y.
    Points with low (or negative) influences are less valuable for model training than higher influence points.

    :param x: A np.ndarray of shape [MxK] containing the features of input data points.
    :param y: A np.ndarray of shape [MxL] containing the targets of input data points.
    :param x_test: A np.ndarray of shape [NxK] containing the features of the test set of data points.
    :param y_test: A np.ndarray of shape [NxL] containing the targets of the test set of data points.
    :param influence_type: Which algorithm to use to calculate influences. Currently supported options: 'up' or 'perturbation'.
    :returns: A np.ndarray of shape [BxC] with the influences of the training points on the test points.
    """

    lr = LinearRegression()
    lr.fit(x, y)
    A = lr.coef_
    b = lr.intercept_

    if influence_type not in list(InfluenceType):
        raise NotImplementedError(
            f"Only upweighting and perturbation influences are supported, but got {influence_type=}"
        )

    if influence_type == InfluenceType.Up:
        return influences_up_linear_regression_analytical(
            (A, b),
            x,
            y,
            x_test,
            y_test,
        )
    elif influence_type == InfluenceType.Perturbation:
        return influences_perturbation_linear_regression_analytical(
            (A, b),
            x,
            y,
            x_test,
            y_test,
        )


def influences_up_linear_regression_analytical(
    linear_model: Tuple["NDArray", "NDArray"],
    x: "NDArray",
    y: "NDArray",
    x_test: "NDArray",
    y_test: "NDArray",
):
    """
    Calculate the influences of the training set onto the validation set for a linear model Ax+b=y.

    :param linear_model: A tuple of np.ndarray' of shape [NxM] and [N] representing A and b respectively.
    :param x_train: A np.ndarray of shape [MxK] containing the features of the train set of data points.
    :param y_train: A np.ndarray of shape [MxL] containing the targets of the train set of data points.
    :param x_test: A np.ndarray of shape [NxK] containing the features of the test set of data points.
    :param y_test: A np.ndarray of shape [NxL] containing the targets of the test set of data points.
    :returns: A np.ndarray of shape [BxC] with the influences of the training points on the test points.
    """

    test_grads_analytical = linear_regression_analytical_derivative_d_theta(
        linear_model,
        x_test,
        y_test,
    )
    train_grads_analytical = linear_regression_analytical_derivative_d_theta(
        linear_model,
        x,
        y,
    )
    hessian_analytical = linear_regression_analytical_derivative_d2_theta(
        linear_model,
        x,
        y,
    )
    s_test_analytical = np.linalg.solve(hessian_analytical, test_grads_analytical.T).T
    result: "NDArray" = np.einsum(
        "ia,ja->ij", s_test_analytical, train_grads_analytical
    )
    return result


def influences_perturbation_linear_regression_analytical(
    linear_model: Tuple["NDArray", "NDArray"],
    x: "NDArray",
    y: "NDArray",
    x_test: "NDArray",
    y_test: "NDArray",
):
    """
    Calculate the influences of each feature of the training set onto the validation set for a linear model Ax+b=y.

    :param linear_model: A tuple of np.ndarray' of shape [NxM] and [N] representing A and b respectively.
    :param x_train: A np.ndarray of shape [MxK] containing the features of input data points.
    :param y_train: A np.ndarray of shape [MxL] containing the targets of input data points.
    :param x_test: A np.ndarray of shape [NxK] containing the features of the test set of data points.
    :param y_test: A np.ndarray of shape [NxL] containing the targets of the test set of data points.
    :returns: A np.ndarray of shape [BxCxM] with the influences of the training points on the test points for each feature.
    """

    test_grads_analytical = linear_regression_analytical_derivative_d_theta(
        linear_model,
        x_test,
        y_test,
    )
    train_second_deriv_analytical = linear_regression_analytical_derivative_d_x_d_theta(
        linear_model,
        x,
        y,
    )

    hessian_analytical = linear_regression_analytical_derivative_d2_theta(
        linear_model,
        x,
        y,
    )
    s_test_analytical = np.linalg.solve(hessian_analytical, test_grads_analytical.T).T
    result: "NDArray" = np.einsum(
        "ia,jab->ijb", s_test_analytical, train_second_deriv_analytical
    )
    return result
