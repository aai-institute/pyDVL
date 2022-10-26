"""
This module contains all functions for the closed form computation of influences
for standard linear regression.
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

__all__ = ["compute_linear_influences"]


def compute_linear_influences(
    x: "NDArray",
    y: "NDArray",
    x_test: "NDArray",
    y_test: "NDArray",
    influence_type: InfluenceType = InfluenceType.Up,
):
    """Calculate the influence each training sample on the loss computed over a
    validation set for an ordinary least squares model ($y = A x + b$ with
    quadratic loss).

    :param x: An array of shape (M, K) containing the features of training data.
    :param y: An array of shape (M, L) containing the targets of training data.
    :param x_test: An array of shape (N, K) containing the features of the
        test set.
    :param y_test: An array of shape (N, L) containing the targets of the test
        set.
    :param influence_type: Which algorithm to use to calculate influences.
        Currently supported options: 'up' or 'perturbation'.
    :returns: An array of shape (B, C) with the influences of the training
        points on the test data.
    """

    lr = LinearRegression()
    lr.fit(x, y)
    A = lr.coef_
    b = lr.intercept_

    if influence_type not in list(InfluenceType):
        raise NotImplementedError(
            f"Only up-weighting and perturbation influences are supported, but got {influence_type=}"
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
    """Calculate the influence each training sample on the loss computed over a
     validation set for an ordinary least squares model (Ax+b=y with quadratic
     loss).

    This method uses the

    :param linear_model: A tuple of arrays of shapes (N, M) and N representing A
        and b respectively.
    :param x: An array of shape (M, K) containing the features of the
        training set.
    :param y: An array of shape (M, L) containing the targets of the
        training set.
    :param x_test: An array of shape (N, K) containing the features of the test
        set.
    :param y_test: An array of shape (N, L) containing the targets of the test
        set.
    :returns: An array of shape (B, C) with the influences of the training points
        on the test points.
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
    """Calculate the influences of each training sample onto the
    validation set for a linear model Ax+b=y.

    :param linear_model: A tuple of np.ndarray' of shape (N, M) and (N)
        representing A and b respectively.
    :param x: An array of shape (M, K) containing the features of the
        input data.
    :param y: An array of shape (M, L) containing the targets of the input
        data.
    :param x_test: An array of shape (N, K) containing the features of the test
        set.
    :param y_test: An array of shape (N, L) containing the targets of the test
        set.
    :returns: An array of shape (B, C, M) with the influences of the training
        points on the test points for each feature.
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
