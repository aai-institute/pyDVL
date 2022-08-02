from typing import Tuple

import numpy as np
from sklearn.linear_model import LinearRegression

from valuation.influence.types import InfluenceTypes
from valuation.utils import (
    linear_regression_analytical_derivative_d2_theta,
    linear_regression_analytical_derivative_d_theta,
    linear_regression_analytical_derivative_d_x_d_theta,
    logger,
)


def linear_influences(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray = None,
    y_test: np.ndarray = None,
    influence_type: InfluenceTypes = InfluenceTypes.Up,
):
    """
    Calculate the linear influences of the training set onto the validation set assuming a linear model Ax+b=y.

    :param x_train: A np.ndarray of shape [MxK] containing the features of the train set of data points.
    :param y_train: A np.ndarray of shape [MxL] containing the targets of the train set of data points.
    :param x_test: A np.ndarray of shape [NxK] containing the features of the test set of data points.
    :param y_test: A np.ndarray of shape [NxL] containing the targets of the test set of data points.
    :returns: A np.ndarray of shape [BxC] with the influences of the training points on the test points.
    """

    if x_test is None or y_test is None:
        logger.info("No test data supplied, train data is reused.")
        x_test = x_train
        y_test = y_train

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    A = lr.coef_
    b = lr.intercept_

    if influence_type == InfluenceTypes.Up:
        return influences_up_linear_regression_analytical(
            (A, b), x_train, y_train, x_test, y_test
        )
    elif influence_type == InfluenceTypes.Perturbation:
        return influences_perturbation_linear_regression_analytical(
            (A, b), x_train, y_train, x_test, y_test
        )
    else:
        raise NotImplementedError(
            "Only upweighting and perturbation influences are supported."
        )


def influences_up_linear_regression_analytical(
    linear_model: Tuple[np.ndarray, np.ndarray],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray = None,
    y_test: np.ndarray = None,
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

    if x_test is None or y_test is None:
        logger.info("No test data supplied, train data is reused.")
        x_test = x_train
        y_test = y_train

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
    x_test: np.ndarray = None,
    y_test: np.ndarray = None,
):
    """
    Calculate the influences of each feature of the training set onto the validation set for a linear model Ax+b=y.

    :param linear_model: A tuple of np.ndarray' of shape [NxM] and [N] representing A and b respectively.
    :param x_train: A np.ndarray of shape [MxK] containing the features of the train set of data points.
    :param y_train: A np.ndarray of shape [MxL] containing the targets of the train set of data points.
    :param x_test: A np.ndarray of shape [NxK] containing the features of the test set of data points.
    :param y_test: A np.ndarray of shape [NxL] containing the targets of the test set of data points.
    :returns: A np.ndarray of shape [BxCxM] with the influences of the training points on the test points for each feature.
    """

    if x_test is None or y_test is None:
        logger.info("No test data supplied, train data is reused.")
        x_test = x_train
        y_test = y_train

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
