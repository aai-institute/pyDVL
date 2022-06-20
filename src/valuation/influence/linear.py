from typing import Tuple

import numpy as np
from sklearn.linear_model import LinearRegression

from valuation.influence.types import InfluenceTypes
from valuation.utils import (
    Dataset,
    linear_regression_analytical_derivative_d2_theta,
    linear_regression_analytical_derivative_d_theta,
    linear_regression_analytical_derivative_d_x_d_theta,
)


def linear_influences(
    dataset: Dataset, influence_type: InfluenceTypes = InfluenceTypes.Up
):

    lr = LinearRegression()
    lr.fit(dataset.x_train, dataset.y_train)
    A = lr.coef_
    b = lr.intercept_

    if influence_type == InfluenceTypes.Up:
        return influences_up_linear_regression_analytical((A, b), dataset)
    elif influence_type == InfluenceTypes.Perturbation:
        return influences_perturbation_linear_regression_analytical((A, b), dataset)
    else:
        raise NotImplementedError(
            "Only upweighting and perturbation influences are supported."
        )


def influences_up_linear_regression_analytical(
    linear_model: Tuple[np.ndarray, np.ndarray],
    dataset: Dataset,
):
    """
    Calculate the influences of the training set onto the validation set for a linear model Ax+b=y.

    :param linear_model: A tuple of np.ndarray' of shape [NxM] and [N] representing A and b respectively.
    :param dataset: A dataset with train and test set for input dimension M and output dimension N.
    :returns: A np.ndarray of shape [BxC] with the influences of the training points on the test points.
    """
    test_grads_analytical = linear_regression_analytical_derivative_d_theta(
        linear_model, dataset.x_test, dataset.y_test
    )
    train_grads_analytical = linear_regression_analytical_derivative_d_theta(
        linear_model, dataset.x_train, dataset.y_train
    )
    hessian_analytical = linear_regression_analytical_derivative_d2_theta(
        linear_model, dataset.x_train, dataset.y_train
    )
    s_test_analytical = np.linalg.solve(hessian_analytical, test_grads_analytical.T).T
    return -np.einsum("ia,ja->ij", s_test_analytical, train_grads_analytical)


def influences_perturbation_linear_regression_analytical(
    linear_model: Tuple[np.ndarray, np.ndarray],
    dataset: Dataset,
):
    """
    Calculate the influences of each feature of the training set onto the validation set for a linear model Ax+b=y.

    :param linear_model: A tuple of np.ndarray' of shape [NxM] and [N] representing A and b respectively.
    :param dataset: A dataset with train and test set for input dimension M and output dimension N.
    :returns: A np.ndarray of shape [BxCxM] with the influences of the training points on the test points for each feature.
    """
    test_grads_analytical = linear_regression_analytical_derivative_d_theta(
        linear_model, dataset.x_test, dataset.y_test
    )
    train_second_deriv_analytical = linear_regression_analytical_derivative_d_x_d_theta(
        linear_model, dataset.x_train, dataset.y_train
    )

    hessian_analytical = linear_regression_analytical_derivative_d2_theta(
        linear_model, dataset.x_train, dataset.y_train
    )
    s_test_analytical = np.linalg.solve(hessian_analytical, test_grads_analytical.T).T
    return -np.einsum("ia,jab->ijb", s_test_analytical, train_second_deriv_analytical)
