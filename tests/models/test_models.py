import itertools
import sys
from typing import List, Tuple

import numpy as np
import pytest

from valuation.utils import (
    linear_regression_analytical_derivative_d2_theta,
    linear_regression_analytical_derivative_d_theta,
    linear_regression_analytical_derivative_d_x_d_theta,
)

try:
    import torch.nn.functional as F

    from valuation.models.linear_regression_torch_model import LRTorchModel
    from valuation.models.pytorch_model import PyTorchOptimizer, PyTorchSupervisedModel
except ImportError:
    pass


class ModelTestSettings:
    DATA_OUTPUT_NOISE: float = 0.01
    ACCEPTABLE_ABS_TOL_MODEL: float = 0.02
    ACCEPTABLE_ABS_TOL_DERIVATIVE: float = 1e-5

    TEST_CONDITION_NUMBERS: List[int] = [5]
    TEST_SET_SIZE: List[int] = [20]
    TRAINING_SET_SIZE: List[int] = [500]
    PROBLEM_DIMENSIONS: List[Tuple[int, int]] = [
        (2, 2),
        (10, 10),
        (5, 10),
        (10, 5),
        (10, 20),
        (20, 10),
    ]


test_cases_model_fit = list(
    itertools.product(
        ModelTestSettings.TRAINING_SET_SIZE,
        ModelTestSettings.TEST_SET_SIZE,
        ModelTestSettings.PROBLEM_DIMENSIONS,
        ModelTestSettings.TEST_CONDITION_NUMBERS,
    )
)


test_cases_model_correctness = list(
    itertools.product(
        ModelTestSettings.TRAINING_SET_SIZE,
        ModelTestSettings.PROBLEM_DIMENSIONS,
        ModelTestSettings.TEST_CONDITION_NUMBERS,
    )
)


def lmb_fit_test_case_to_str(packed_i_test_case):
    i, test_case = packed_i_test_case
    return f"Problem #{i} of dimension {test_case[2]} with train size {test_case[0]}, test size {test_case[1]} and condition number {test_case[3]}"


def lmb_correctness_test_case_to_str(packed_i_test_case):
    i, test_case = packed_i_test_case
    return f"Problem #{i} of dimension {test_case[1]} with train size {test_case[0]} and condition number {test_case[2]}"


fit_test_case_ids = list(
    map(
        lmb_fit_test_case_to_str,
        zip(range(len(test_cases_model_fit)), test_cases_model_fit),
    )
)
correctness_test_case_ids = list(
    map(
        lmb_correctness_test_case_to_str,
        zip(range(len(test_cases_model_correctness)), test_cases_model_correctness),
    )
)


@pytest.mark.torch
@pytest.mark.skip()
@pytest.mark.parametrize(
    "train_set_size,test_set_size,problem_dimension,condition_number",
    test_cases_model_fit,
    ids=fit_test_case_ids,
)
def test_linear_regression_model_fit(
    train_set_size: int,
    test_set_size: int,
    condition_number: float,
    linear_model: Tuple[np.ndarray, np.ndarray],
):

    # some settings
    A, b = linear_model
    output_dimension, input_dimension = tuple(A.shape)

    # generate datasets
    data_model = lambda x: np.random.normal(
        x @ A.T + b, ModelTestSettings.DATA_OUTPUT_NOISE
    )
    train_x = np.random.uniform(size=[train_set_size, input_dimension])
    train_y = data_model(train_x)

    model = PyTorchSupervisedModel(
        model=LRTorchModel((output_dimension, input_dimension)),
        objective=F.mse_loss,
        num_epochs=1000,
        batch_size=32,
        optimizer=PyTorchOptimizer.ADAM_W,
        optimizer_kwargs={"lr": 0.05},
    )
    model.fit(train_x, train_y)
    learned_A = model.model.A.detach().numpy()
    max_A_diff = np.max(np.abs(learned_A - A))
    assert (
        max_A_diff < ModelTestSettings.ACCEPTABLE_ABS_TOL_MODEL
    ), "A did not converged to target solution."

    learned_b = model.model.b.detach().numpy()
    max_b_diff = np.max(np.abs(learned_b - b))
    assert (
        max_b_diff < ModelTestSettings.ACCEPTABLE_ABS_TOL_MODEL
    ), "b did not converged to target solution."


@pytest.mark.torch
@pytest.mark.parametrize(
    "train_set_size,problem_dimension,condition_number",
    test_cases_model_correctness,
    ids=correctness_test_case_ids,
)
def test_linear_regression_model_grad(
    train_set_size: int,
    condition_number: float,
    linear_model: Tuple[np.ndarray, np.ndarray],
):
    # some settings
    A, b = linear_model
    output_dimension, input_dimension = tuple(A.shape)

    # generate datasets
    data_model = lambda x: np.random.normal(
        x @ A.T + b, ModelTestSettings.DATA_OUTPUT_NOISE
    )
    train_x = np.random.uniform(size=[train_set_size, input_dimension])
    train_y = data_model(train_x)

    model = PyTorchSupervisedModel(
        model=LRTorchModel(dim=(input_dimension, output_dimension), init=linear_model),
        objective=F.mse_loss,
    )

    train_grads_analytical = 2 * linear_regression_analytical_derivative_d_theta(
        (A, b), train_x, train_y
    )
    train_grads_autograd = model.grad(train_x, train_y)
    train_grads_max_diff = np.max(np.abs(train_grads_analytical - train_grads_autograd))
    assert (
        train_grads_max_diff < ModelTestSettings.ACCEPTABLE_ABS_TOL_DERIVATIVE
    ), "Train set produces wrong gradients."


@pytest.mark.torch
@pytest.mark.parametrize(
    "train_set_size,problem_dimension,condition_number",
    test_cases_model_correctness,
    ids=correctness_test_case_ids,
)
def test_linear_regression_model_hessian(
    train_set_size: int,
    condition_number: float,
    linear_model: Tuple[np.ndarray, np.ndarray],
):
    # some settings
    A, b = linear_model
    output_dimension, input_dimension = tuple(A.shape)

    # generate datasets
    data_model = lambda x: np.random.normal(
        x @ A.T + b, ModelTestSettings.DATA_OUTPUT_NOISE
    )
    train_x = np.random.uniform(size=[train_set_size, input_dimension])
    train_y = data_model(train_x)

    model = PyTorchSupervisedModel(
        model=LRTorchModel(dim=(input_dimension, output_dimension), init=linear_model),
        objective=F.mse_loss,
    )

    test_hessian_analytical = 2 * linear_regression_analytical_derivative_d2_theta(
        (A, b), train_x, train_y
    )
    estimated_hessian = model.mvp(
        train_x, train_y, np.eye((input_dimension + 1) * output_dimension)
    )
    test_hessian_max_diff = np.max(np.abs(test_hessian_analytical - estimated_hessian))
    assert (
        test_hessian_max_diff < ModelTestSettings.ACCEPTABLE_ABS_TOL_DERIVATIVE
    ), "Hessian was wrong."


@pytest.mark.torch
@pytest.mark.parametrize(
    "train_set_size,problem_dimension,condition_number",
    test_cases_model_correctness,
    ids=correctness_test_case_ids,
)
def test_linear_regression_model_d_x_d_theta(
    train_set_size: int,
    condition_number: float,
    linear_model: Tuple[np.ndarray, np.ndarray],
):
    # some settings
    A, b = linear_model
    output_dimension, input_dimension = tuple(A.shape)

    # generate datasets
    data_model = lambda x: np.random.normal(
        x @ A.T + b, ModelTestSettings.DATA_OUTPUT_NOISE
    )
    train_x = np.random.uniform(size=[train_set_size, input_dimension])
    train_y = data_model(train_x)

    model = PyTorchSupervisedModel(
        model=LRTorchModel(dim=(input_dimension, output_dimension), init=(A, b)),
        objective=F.mse_loss,
        num_epochs=1000,
        batch_size=32,
        optimizer=PyTorchOptimizer.ADAM_W,
        optimizer_kwargs={"lr": 0.02},
    )

    test_derivative = 2 * linear_regression_analytical_derivative_d_x_d_theta(
        (A, b), train_x, train_y
    )
    estimated_derivative = np.stack(
        [
            model.mvp(
                train_x[i],
                train_y[i],
                np.eye((input_dimension + 1) * output_dimension),
                second_x=True,
            )
            for i in range(len(train_x))
        ],
        axis=0,
    )
    test_hessian_max_diff = np.max(np.abs(test_derivative - estimated_derivative))
    assert (
        test_hessian_max_diff < ModelTestSettings.ACCEPTABLE_ABS_TOL_DERIVATIVE
    ), "Hessian was wrong."
