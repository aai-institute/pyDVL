from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from pydvl.utils import Dataset, random_matrix_with_condition_number

try:
    import torch
    from torch import nn
except ImportError:
    pass

if TYPE_CHECKING:
    from numpy.typing import NDArray


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


@pytest.fixture(scope="function")
def quadratic_linear_equation_system(quadratic_matrix: np.ndarray, batch_size: int):
    A = quadratic_matrix
    problem_dimension = A.shape[0]
    b = np.random.random([batch_size, problem_dimension])
    return A, b


@pytest.fixture(scope="function")
def quadratic_matrix(problem_dimension: int, condition_number: float):
    return random_matrix_with_condition_number(problem_dimension, condition_number)


@pytest.fixture(scope="function")
def singular_quadratic_linear_equation_system(
    quadratic_matrix: np.ndarray, batch_size: int
):
    A = quadratic_matrix
    problem_dimension = A.shape[0]
    i, j = tuple(np.random.choice(problem_dimension, replace=False, size=2))
    if j < i:
        i, j = j, i

    v = (A[i] + A[j]) / 2
    A[i], A[j] = v, v
    b = np.random.random([batch_size, problem_dimension])
    return A, b


@pytest.fixture(scope="function")
def linear_model(problem_dimension: Tuple[int, int], condition_number: float):
    output_dimension, input_dimension = problem_dimension
    A = random_matrix_with_condition_number(
        max(input_dimension, output_dimension), condition_number
    )
    A = A[:output_dimension, :input_dimension]
    b = np.random.uniform(size=[output_dimension])
    return A, b


class TorchSimpleNN(nn.Module):
    """
    Creates a simple pytorch neural network. It needs input features, number of layers
    """

    def __init__(
        self,
        n_input: int,
        n_hidden_layers: int,
        n_neurons_per_layer: List[int],
        output_layer: nn.Module,
        init: List[Tuple["NDArray[np.float_]", "NDArray[np.float_]"]] = None,
    ):
        """
        :param n_input: Number of feature in input.
        :param n_output: Output length.
        :param n_neurons_per_layer: Each integer represents the size of a hidden
            layer. Overall this list has K - 2
        :param output_layer: output layer of the neural network
        number of outputs reduce to 1.
        :param init: A list of tuple of np.ndarray representing the internal weights.
        """
        super().__init__()

        layers = [nn.Linear(n_input, n_neurons_per_layer)]
        for num_layer in range(n_hidden_layers):
            linear_layer = nn.Linear(n_neurons_per_layer, n_neurons_per_layer)

            if init is not None:
                A, b = init[num_layer]
                linear_layer.weight.data = A
                linear_layer.bias.data = b

            layers.append(linear_layer)
        layers.append(output_layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward-pass through the network.
        :param x: Tensor input of shape [NxD].
        :returns: Tensor output of shape[NxK].
        """
        return self.layers(x)


def linear_regression_analytical_derivative_d_theta(
    linear_model: Tuple["NDArray", "NDArray"], x: "NDArray", y: "NDArray"
) -> "NDArray":
    """
    :param linear_model: A tuple of np.ndarray' of shape [NxM] and [N] representing A and b respectively.
    :param x: A np.ndarray of shape [BxM].
    :param y: A np.nparray of shape [BxN].
    :returns: A np.ndarray of shape [Bx((N+1)*M)], where each row vector is [d_theta L(x, y), d_b L(x, y)]
    """

    A, b = linear_model
    n, m = list(A.shape)
    residuals = x @ A.T + b - y
    kron_product = np.expand_dims(residuals, axis=2) * np.expand_dims(x, axis=1)
    test_grads = np.reshape(kron_product, [-1, n * m])
    full_grads = np.concatenate((test_grads, residuals), axis=1)
    return full_grads / n  # type: ignore


def linear_regression_analytical_derivative_d2_theta(
    linear_model: Tuple["NDArray", "NDArray"], x: "NDArray", y: "NDArray"
) -> "NDArray":
    """
    :param linear_model: A tuple of np.ndarray' of shape [NxM] and [N] representing A and b respectively.
    :param x: A np.ndarray of shape [BxM],
    :param y: A np.nparray of shape [BxN].
    :returns: A np.ndarray of shape [((N+1)*M)x((N+1)*M)], representing the Hessian. It gets averaged over all samples.
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
    return full_matrix / n  # type: ignore


def linear_regression_analytical_derivative_d_x_d_theta(
    linear_model: Tuple["NDArray", "NDArray"], x: "NDArray", y: "NDArray"
) -> "NDArray":
    """
    :param linear_model: A tuple of np.ndarray of shape [NxM] and [N] representing A and b respectively.
    :param x: A np.ndarray of shape [BxM].
    :param y: A np.nparray of shape [BxN].
    :returns: A np.ndarray of shape [Bx((N+1)*M)xM], representing the derivative.
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
    return full_derivative / N  # type: ignore


def analytical_linear_influences(
    linear_model: Tuple["NDArray", "NDArray"],
    x: "NDArray",
    y: "NDArray",
    x_test: "NDArray",
    y_test: "NDArray",
    influence_type: InfluenceType = InfluenceType.Up,
):
    """Calculates analytically the influence of each training sample on the
     test samples for an ordinary least squares model (Ax+b=y with quadratic
     loss).

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
    :param influence_type: the type of the influence.
    :returns: An array of shape (B, C) with the influences of the training points
        on the test points if influence_type is "up", an array of shape (K, L,
        M) if influence_type is "perturbation".
    """

    test_grads_analytical = linear_regression_analytical_derivative_d_theta(
        linear_model,
        x_test,
        y_test,
    )
    hessian_analytical = linear_regression_analytical_derivative_d2_theta(
        linear_model,
        x,
        y,
    )
    s_test_analytical = np.linalg.solve(hessian_analytical, test_grads_analytical.T).T
    if influence_type == "up":
        train_grads_analytical = linear_regression_analytical_derivative_d_theta(
            linear_model,
            x,
            y,
        )
        result: "NDArray" = np.einsum(
            "ia,ja->ij", s_test_analytical, train_grads_analytical
        )
    elif influence_type == "perturbation":
        train_second_deriv_analytical = (
            linear_regression_analytical_derivative_d_x_d_theta(
                linear_model,
                x,
                y,
            )
        )
        result: "NDArray" = np.einsum(
            "ia,jab->ijb", s_test_analytical, train_second_deriv_analytical
        )
    return result


def create_mock_dataset(
    linear_model: Tuple["NDArray[np.float_]", "NDArray[np.float_]"],
    train_set_size: int,
    test_set_size: int,
    noise: float = 0.01,
) -> Tuple[
    Tuple["NDArray[np.float_]", "NDArray[np.float_]"],
    Tuple["NDArray[np.float_]", "NDArray[np.float_]"],
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
