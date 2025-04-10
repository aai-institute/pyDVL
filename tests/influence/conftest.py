from typing import Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler

from pydvl.influence import InfluenceMode
from pydvl.utils import Dataset, random_matrix_with_condition_number


@pytest.fixture
def input_dimension(request) -> int:
    return request.param


@pytest.fixture
def output_dimension(request) -> int:
    return request.param


@pytest.fixture
def problem_dimension(request) -> int:
    return request.param


@pytest.fixture
def batch_size(request) -> int:
    return request.param


@pytest.fixture
def condition_number(request) -> float:
    return request.param


def linear_model(problem_dimension: Tuple[int, int], condition_number: float):
    output_dimension, input_dimension = problem_dimension
    A = random_matrix_with_condition_number(
        max(input_dimension, output_dimension), condition_number
    )
    A = A[:output_dimension, :input_dimension]
    b = np.random.uniform(size=[output_dimension])
    return A, b


def linear_derivative_analytical(
    linear_model: Tuple[NDArray[np.float64], NDArray[np.float64]],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Given a linear model it returns the first order derivative wrt its parameters.
    More precisely, given a couple of matrices $A(\theta)$ and $b(\theta')$, with
    $\theta$, $\theta'$ representing their generic entry, it calculates the
    derivative wrt. $\theta$ and $\theta'$ of the linear model with the
    following quadratic loss: $L(x,y) = (Ax +b - y)^2$.
    :param linear_model: A tuple of arrays representing the linear model.
    :param x: array, input to the linear model
    :param y: array, output of the linear model
    :returns: An array where each row holds the derivative over $\theta$ of $L(x, y)]$
    """

    A, b = linear_model
    n, m = list(A.shape)
    residuals = x @ A.T + b - y
    kron_product = np.expand_dims(residuals, axis=2) * np.expand_dims(x, axis=1)
    test_grads = np.reshape(kron_product, [-1, n * m])
    full_grads = np.concatenate((test_grads, residuals), axis=1)
    return 2 * full_grads / n  # type: ignore


def linear_hessian_analytical(
    linear_model: Tuple[NDArray[np.float64], NDArray[np.float64]],
    x: NDArray[np.float64],
    lam: float = 0.0,
) -> NDArray[np.float64]:
    """
    Given a linear model it returns the hessian wrt. its parameters.
    More precisely, given a couple of matrices $A(\theta)$ and $b(\theta')$, with
    $\theta$, $\theta'$ representing their generic entry, it calculates the
    second derivative wrt. $\theta$ and $\theta'$ of the linear model with the
    following quadratic loss: $L(x,y) = (Ax +b - y)^2$.
    :param linear_model: A tuple of arrays representing the linear model.
    :param x: array, input to the linear model
    :param y: array, output of the linear model
    :param lam: hessian regularization parameter
    :returns: An matrix where each entry i,j holds the second derivatives over $\theta$
    of $L(x, y)$
    """
    A, b = linear_model
    n, m = tuple(A.shape)
    d2_theta = np.einsum("ia,ib->iab", x, x)
    d2_theta = np.mean(d2_theta, axis=0)
    d2_theta = np.kron(np.eye(n), d2_theta)
    d2_b = np.eye(n)
    mean_x = np.mean(x, axis=0, keepdims=True)
    d_theta_d_b = np.kron(np.eye(n), mean_x)
    top_matrix = np.concatenate((d2_theta, d_theta_d_b.T), axis=1)
    bottom_matrix = np.concatenate((d_theta_d_b, d2_b), axis=1)
    full_matrix = np.concatenate((top_matrix, bottom_matrix), axis=0)
    return 2 * full_matrix / n + lam * np.identity(len(full_matrix))  # type: ignore


def linear_mixed_second_derivative_analytical(
    linear_model: Tuple[NDArray[np.float64], NDArray[np.float64]],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Given a linear model it returns a second order partial derivative wrt its
    parameters .
    More precisely, given a couple of matrices $A(\theta)$ and $b(\theta')$, with
    $\theta$, $\theta'$ representing their generic entry, it calculates the
    second derivative wrt. $\theta$ and $\theta'$ of the linear model with the
    following quadratic loss: $L(x,y) = (Ax +b - y)^2$.
    :param linear_model: A tuple of arrays representing the linear model.
    :param x: array, input to the linear model
    :param y: array, output of the linear model
    :returns: An matrix where each entry i,j holds the mixed second derivatives
    over $\theta$ and $x$ of $L(x, y)$
    """

    A, b = linear_model
    N, M = tuple(A.shape)
    residuals = x @ A.T + b - y
    B = len(x)
    outer_product_matrix = np.einsum("ab,ic->iacb", A, x)
    outer_product_matrix = np.reshape(outer_product_matrix, [B, M * N, M])
    tiled_identity = np.tile(np.expand_dims(np.eye(M), axis=0), [B, N, 1])
    outer_product_matrix += tiled_identity * np.expand_dims(
        np.repeat(residuals, M, axis=1), axis=2
    )
    b_part_derivative = np.tile(np.expand_dims(A, axis=0), [B, 1, 1])
    full_derivative = np.concatenate((outer_product_matrix, b_part_derivative), axis=1)
    return 2 * full_derivative / N  # type: ignore


def linear_analytical_influence_factors(
    linear_model: Tuple[NDArray[np.float64], NDArray[np.float64]],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    x_test: NDArray[np.float64],
    y_test: NDArray[np.float64],
    hessian_regularization: float = 0,
) -> NDArray[np.float64]:
    """
    Given a linear model it calculates its influence factors.
    :param linear_model: A tuple of arrays representing the linear model.
    :param x: array, input to the linear model
    :param y: array, output of the linear model
    :returns: An array with analytical influence factors.
    """
    test_grads_analytical = linear_derivative_analytical(
        linear_model,
        x_test,
        y_test,
    )
    hessian_analytical = linear_hessian_analytical(
        linear_model,
        x,
        hessian_regularization,
    )
    return np.linalg.solve(hessian_analytical, test_grads_analytical.T).T


def add_noise_to_linear_model(
    linear_model: Tuple[NDArray[np.float64], NDArray[np.float64]],
    train_set_size: int,
    test_set_size: int,
    noise: float = 0.01,
) -> Tuple[
    Tuple[NDArray[np.float64], NDArray[np.float64]],
    Tuple[NDArray[np.float64], NDArray[np.float64]],
]:
    A, b = linear_model
    o_d, i_d = tuple(A.shape)
    data_model = lambda x: np.random.normal(x @ A.T + b, noise)

    x_train = np.random.uniform(size=[train_set_size, i_d])
    y_train = data_model(x_train)
    x_test = np.random.uniform(size=[test_set_size, i_d])
    y_test = data_model(x_test)
    dataset = Dataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        is_multi_output=True,
    )

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    dataset.x_train = scaler_x.fit_transform(dataset.x_train)
    dataset.y_train = scaler_y.fit_transform(dataset.y_train)
    dataset.x_test = scaler_x.transform(dataset.x_test)
    dataset.y_test = scaler_y.transform(dataset.y_test)
    return (x_train, y_train), (x_test, y_test)


def analytical_linear_influences(
    linear_model: Tuple[NDArray[np.float64], NDArray[np.float64]],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    x_test: NDArray[np.float64],
    y_test: NDArray[np.float64],
    mode: InfluenceMode = InfluenceMode.Up,
    hessian_regularization: float = 0,
):
    """
    Calculates analytically the influence of each training sample on the
    test samples for an ordinary least squares model (Ax+b=y with quadratic
    loss).

    Args:
        linear_model: A tuple of arrays of shapes (N, M) and N representing A
            and b respectively.
        x: An array of shape (M, K) containing the features of thr training set.
        y: An array of shape (M, L) containing the targets of the training set.
        x_test: An array of shape (N, K) containing the features of the test set.
        y_test: An array of shape (N, L) containing the targets of the test set.
        mode: the type of the influence.
        hessian_regularization: regularization value for the hessian

    Returns:
        An array of shape (B, C) with the influences of the training points
            on the test points if `mode` is "up", an array of shape (K, L, M)
            if `mode` is "perturbation".
    """

    s_test_analytical = linear_analytical_influence_factors(
        linear_model, x, y, x_test, y_test, hessian_regularization
    )
    if mode == InfluenceMode.Up:
        train_grads_analytical = linear_derivative_analytical(
            linear_model,
            x,
            y,
        )
        result = np.einsum("ia,ja->ij", s_test_analytical, train_grads_analytical)
    elif mode == InfluenceMode.Perturbation:
        train_second_deriv_analytical = linear_mixed_second_derivative_analytical(
            linear_model,
            x,
            y,
        )
        result = np.einsum(
            "ia,jab->ijb", s_test_analytical, train_second_deriv_analytical
        )
    return result
