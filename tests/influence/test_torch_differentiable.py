"""
Contains tests for LinearRegression, BinaryLogisticRegression as well as TwiceDifferentiable modules and
its associated gradient and matrix vector product calculations. Note that there is no test for the neural network
module.
"""

import itertools
from typing import List, Tuple

import numpy as np
import pytest

from .conftest import (
    linear_derivative_analytical,
    linear_hessian_analytical,
    linear_mixed_second_derivative_analytical,
    linear_model,
)

torch = pytest.importorskip("torch")
import torch
import torch.nn.functional as F
from torch import nn

from pydvl.influence.frameworks.torch_differentiable import (
    TorchTwiceDifferentiable,
    mvp,
)

DATA_OUTPUT_NOISE: float = 0.01

TEST_CONDITION_NUMBERS: List[int] = [5]
TEST_SET_SIZE: List[int] = [20]
TRAINING_SET_SIZE: List[int] = [50]
PROBLEM_DIMENSIONS: List[Tuple[int, int]] = [
    (2, 2),
    (5, 10),
    (10, 5),
    (10, 10),
]

test_cases_linear_regression_derivatives = list(
    itertools.product(
        TRAINING_SET_SIZE,
        PROBLEM_DIMENSIONS,
        TEST_CONDITION_NUMBERS,
    )
)


def lmb_correctness_test_case_to_str(packed_i_test_case):
    i, test_case = packed_i_test_case
    return f"Problem #{i} of dimension {test_case[1]} with train size {test_case[0]} and condition number {test_case[2]}"


correctness_test_case_ids = list(
    map(
        lmb_correctness_test_case_to_str,
        zip(
            range(len(test_cases_linear_regression_derivatives)),
            test_cases_linear_regression_derivatives,
        ),
    )
)


@pytest.mark.torch
@pytest.mark.parametrize(
    "train_set_size,problem_dimension,condition_number",
    test_cases_linear_regression_derivatives,
    ids=correctness_test_case_ids,
)
def test_linear_grad(
    train_set_size: int,
    problem_dimension: Tuple[int, int],
    condition_number: float,
):
    # some settings
    A, b = linear_model(problem_dimension, condition_number)
    output_dimension, input_dimension = tuple(A.shape)

    # generate datasets
    data_model = lambda x: np.random.normal(x @ A.T + b, DATA_OUTPUT_NOISE)
    train_x = np.random.uniform(size=[train_set_size, input_dimension])
    train_y = data_model(train_x)

    model = nn.Linear(input_dimension, output_dimension)
    model.eval()
    model.weight.data = torch.as_tensor(A)
    model.bias.data = torch.as_tensor(b)
    loss = F.mse_loss
    mvp_model = TorchTwiceDifferentiable(model=model, loss=loss)

    train_grads_analytical = linear_derivative_analytical((A, b), train_x, train_y)
    train_grads_autograd = mvp_model.split_grad(train_x, train_y)
    assert np.allclose(train_grads_analytical, train_grads_autograd, rtol=1e-5)


@pytest.mark.torch
@pytest.mark.parametrize(
    "train_set_size,problem_dimension,condition_number",
    test_cases_linear_regression_derivatives,
    ids=correctness_test_case_ids,
)
def test_linear_hessian(
    train_set_size: int,
    problem_dimension: Tuple[int, int],
    condition_number: float,
):
    # some settings
    A, b = linear_model(problem_dimension, condition_number)
    output_dimension, input_dimension = tuple(A.shape)

    # generate datasets
    data_model = lambda x: np.random.normal(x @ A.T + b, DATA_OUTPUT_NOISE)
    train_x = np.random.uniform(size=[train_set_size, input_dimension])
    train_y = data_model(train_x)
    model = nn.Linear(input_dimension, output_dimension)
    model.eval()
    model.weight.data = torch.as_tensor(A)
    model.bias.data = torch.as_tensor(b)
    loss = F.mse_loss
    mvp_model = TorchTwiceDifferentiable(model=model, loss=loss)

    test_hessian_analytical = linear_hessian_analytical((A, b), train_x)
    grad_xy, _ = mvp_model.grad(train_x, train_y)
    estimated_hessian = mvp(
        grad_xy,
        np.eye((input_dimension + 1) * output_dimension),
        mvp_model.parameters(),
    )
    assert np.allclose(test_hessian_analytical, estimated_hessian, rtol=1e-5)


@pytest.mark.torch
@pytest.mark.parametrize(
    "train_set_size,problem_dimension,condition_number",
    test_cases_linear_regression_derivatives,
    ids=correctness_test_case_ids,
)
def test_linear_mixed_derivative(
    train_set_size: int,
    problem_dimension: Tuple[int, int],
    condition_number: float,
):
    # some settings
    A, b = linear_model(problem_dimension, condition_number)
    output_dimension, input_dimension = tuple(A.shape)

    # generate datasets
    data_model = lambda x: np.random.normal(x @ A.T + b, DATA_OUTPUT_NOISE)
    train_x = np.random.uniform(size=[train_set_size, input_dimension])
    train_y = data_model(train_x)
    model = nn.Linear(input_dimension, output_dimension)
    model.eval()
    model.weight.data = torch.as_tensor(A)
    model.bias.data = torch.as_tensor(b)
    loss = F.mse_loss
    mvp_model = TorchTwiceDifferentiable(model=model, loss=loss)

    test_derivative = linear_mixed_second_derivative_analytical(
        (A, b),
        train_x,
        train_y,
    )
    model_mvp = []
    for i in range(len(train_x)):
        grad_xy, tensor_x = mvp_model.grad(train_x[i], train_y[i])
        model_mvp.append(
            mvp(
                grad_xy,
                np.eye((input_dimension + 1) * output_dimension),
                backprop_on=tensor_x,
            )
        )
    estimated_derivative = np.stack(model_mvp, axis=0)
    assert np.allclose(test_derivative, estimated_derivative, rtol=1e-7)
