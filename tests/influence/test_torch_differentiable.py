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
from torch.utils.data import DataLoader

from pydvl.influence.frameworks.torch_differentiable import (
    TorchTwiceDifferentiable,
    mvp,
    solve_batch_cg,
    solve_linear,
    solve_lissa,
)

DATA_OUTPUT_NOISE: float = 0.01

PROBLEM_DIMENSIONS: List[Tuple[int, int]] = [
    (2, 2),
    (5, 10),
    (10, 5),
    (10, 10),
]


def linear_mvp_model(A, b):
    output_dimension, input_dimension = tuple(A.shape)
    model = nn.Linear(input_dimension, output_dimension)
    model.eval()
    model.weight.data = torch.as_tensor(A)
    model.bias.data = torch.as_tensor(b)
    loss = F.mse_loss
    return TorchTwiceDifferentiable(model=model, loss=loss, device=torch.device("cpu"))


@pytest.mark.torch
@pytest.mark.parametrize(
    "problem_dimension",
    PROBLEM_DIMENSIONS,
    ids=[f"problem_dimension={dim}" for dim in PROBLEM_DIMENSIONS],
)
def test_linear_grad(
    problem_dimension: Tuple[int, int],
    train_set_size: int = 50,
    condition_number: float = 5,
):
    # some settings
    A, b = linear_model(problem_dimension, condition_number)
    _, input_dimension = tuple(A.shape)

    # generate datasets
    data_model = lambda x: np.random.normal(x @ A.T + b, DATA_OUTPUT_NOISE)
    train_x = np.random.uniform(size=[train_set_size, input_dimension])
    train_y = data_model(train_x)

    mvp_model = linear_mvp_model(A, b)

    train_grads_analytical = linear_derivative_analytical((A, b), train_x, train_y)
    train_grads_autograd = mvp_model.split_grad(train_x, train_y)
    assert np.allclose(train_grads_analytical, train_grads_autograd, rtol=1e-5)


@pytest.mark.torch
@pytest.mark.parametrize(
    "problem_dimension",
    PROBLEM_DIMENSIONS,
    ids=[f"problem_dimension={dim}" for dim in PROBLEM_DIMENSIONS],
)
def test_linear_hessian(
    problem_dimension: Tuple[int, int],
    train_set_size: int = 50,
    condition_number: float = 5,
):
    # some settings
    A, b = linear_model(problem_dimension, condition_number)
    output_dimension, input_dimension = tuple(A.shape)

    # generate datasets
    data_model = lambda x: np.random.normal(x @ A.T + b, DATA_OUTPUT_NOISE)
    train_x = np.random.uniform(size=[train_set_size, input_dimension])
    train_y = data_model(train_x)
    mvp_model = linear_mvp_model(A, b)

    test_hessian_analytical = linear_hessian_analytical((A, b), train_x)
    grad_xy, _ = mvp_model.grad(train_x, train_y)
    estimated_hessian = mvp(
        grad_xy,
        np.eye((input_dimension + 1) * output_dimension),
        mvp_model.parameters,
    )
    assert np.allclose(test_hessian_analytical, estimated_hessian, rtol=1e-5)


@pytest.mark.torch
@pytest.mark.parametrize(
    "problem_dimension",
    PROBLEM_DIMENSIONS,
    ids=[f"problem_dimension={dim}" for dim in PROBLEM_DIMENSIONS],
)
def test_linear_mixed_derivative(
    problem_dimension: Tuple[int, int],
    train_set_size: int = 50,
    condition_number: float = 5,
):
    # some settings
    A, b = linear_model(problem_dimension, condition_number)
    output_dimension, input_dimension = tuple(A.shape)

    # generate datasets
    data_model = lambda x: np.random.normal(x @ A.T + b, DATA_OUTPUT_NOISE)
    train_x = np.random.uniform(size=[train_set_size, input_dimension])
    train_y = data_model(train_x)

    mvp_model = linear_mvp_model(A, b)

    test_derivative = linear_mixed_second_derivative_analytical(
        (A, b),
        train_x,
        train_y,
    )
    model_mvp = []
    for i in range(len(train_x)):
        grad_xy, tensor_x = mvp_model.grad(train_x[i], train_y[i], x_requires_grad=True)
        model_mvp.append(
            mvp(
                grad_xy,
                np.eye((input_dimension + 1) * output_dimension),
                backprop_on=tensor_x,
            )
        )
    estimated_derivative = np.stack(model_mvp, axis=0)
    assert np.allclose(test_derivative, estimated_derivative, rtol=1e-7)


REDUCED_PROBLEM_DIMENSIONS: List[Tuple[int, int]] = [(5, 10), (2, 5)]


@pytest.mark.torch
@pytest.mark.parametrize(
    "problem_dimension",
    REDUCED_PROBLEM_DIMENSIONS,
    ids=[f"problem_dimension={dim}" for dim in REDUCED_PROBLEM_DIMENSIONS],
)
def test_inversion_methods(
    problem_dimension: Tuple[int, int],
    train_set_size: int = 50,
    condition_number: float = 5,
):
    # some settings
    A, b = linear_model(problem_dimension, condition_number)
    output_dimension, input_dimension = tuple(A.shape)

    # generate datasets
    data_model = lambda x: np.random.normal(x @ A.T + b, DATA_OUTPUT_NOISE)
    train_x = np.random.uniform(size=[train_set_size, input_dimension])
    train_y = data_model(train_x)
    mvp_model = linear_mvp_model(A, b)

    train_data_loader = DataLoader(list(zip(train_x, train_y)), batch_size=128)
    b = torch.rand(size=(10, mvp_model.num_params), dtype=torch.float64)

    linear_inverse = solve_linear(mvp_model, train_data_loader, b)
    linear_cg = solve_batch_cg(mvp_model, train_data_loader, b)
    linear_lissa = solve_lissa(mvp_model, train_data_loader, b, maxiter=5000, scale=5)

    assert np.allclose(linear_inverse, linear_cg, rtol=1e-1)
    assert np.allclose(linear_inverse, linear_lissa, rtol=1e-1)
