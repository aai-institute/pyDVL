"""
Contains all functions for closed form solution of influences for standard linear regression.
"""

__all__ = [
    "linear_influences",
    "influences_up_linear_regression_analytical",
    "influences_perturbation_linear_regression_analytical",
]

from typing import Tuple

import numpy as np
from sklearn.linear_model import LinearRegression

from valuation.utils import Dataset
from valuation.utils.numeric import (
    linear_regression_analytical_derivative_d2_theta,
    linear_regression_analytical_derivative_d_theta,
    linear_regression_analytical_derivative_d_x_d_theta,
)


def linear_influences(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    influence_type: str = "up",
):
    """
    Calculate the linear influences of the training set onto the validation set assuming a linear model Ax+b=y.
    :param data: a dataset
    :param influence_type: Which algorithm to use to calculate influences.
        Currently supported options: 'up' or 'perturbation'.
    :returns: A np.ndarray of shape [BxC] with the influences of the training points on the test points.
    """

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    A = lr.coef_
    b = lr.intercept_

    if influence_type == "up":
        return -1 * influences_up_linear_regression_analytical(
            (A, b),
            x_train,
            y_train,
            x_test,
            y_test,
        )
    elif influence_type == "perturbation":
        return -1 * influences_perturbation_linear_regression_analytical(
            (A, b),
            x_train,
            y_train,
            x_test,
            y_test,
        )
    else:
        raise NotImplementedError(
            f"Only upweighting and perturbation influences are supported, but got {influence_type=}"
        )


def influences_up_linear_regression_analytical(
    linear_model: Tuple[np.ndarray, np.ndarray],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
):
    """
    Calculate the influences of the training set onto the validation set for a linear model Ax+b=y.

    :param linear_model: A tuple of np.ndarray' of shape [NxM] and [N] representing A and b respectively.
    :param data: a dataset
    :returns: A np.ndarray of shape [BxC] with the influences of the training points on the test points.
    """

    test_grads_analytical = linear_regression_analytical_derivative_d_theta(
        linear_model, x_test, y_test
    )
    train_grads_analytical = linear_regression_analytical_derivative_d_theta(
        linear_model, x_train, y_train
    )
    hessian_analytical = linear_regression_analytical_derivative_d2_theta(
        linear_model, x_train, y_train
    )
    s_test_analytical = np.linalg.solve(hessian_analytical, test_grads_analytical.T).T
    return -np.einsum("ia,ja->ij", s_test_analytical, train_grads_analytical)


def influences_perturbation_linear_regression_analytical(
    linear_model: Tuple[np.ndarray, np.ndarray],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
):
    """
    Calculate the influences of each feature of the training set onto the validation set for a linear model Ax+b=y.

    :param linear_model: A tuple of np.ndarray' of shape [NxM] and [N] representing A and b respectively.
    :param data: a dataset
    :returns: A np.ndarray of shape [BxCxM] with the influences of the training points on the test points for each feature.
    """

    test_grads_analytical = linear_regression_analytical_derivative_d_theta(
        linear_model, x_test, y_test
    )
    train_second_deriv_analytical = linear_regression_analytical_derivative_d_x_d_theta(
        linear_model, x_train, y_train
    )

    hessian_analytical = linear_regression_analytical_derivative_d2_theta(
        linear_model, x_train, y_train
    )
    s_test_analytical = np.linalg.solve(hessian_analytical, test_grads_analytical.T).T
    return -np.einsum("ia,jab->ijb", s_test_analytical, train_second_deriv_analytical)
