from typing import Tuple

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

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


def create_mock_dataset(
    linear_model: Tuple[np.ndarray, np.ndarray],
    train_set_size: int,
    test_set_size: int,
    noise: float = 0.01,
) -> Dataset:
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
