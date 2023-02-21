import itertools
from typing import List, Tuple

import numpy as np
import pytest

from pydvl.utils.numeric import (
    linear_regression_analytical_derivative_d2_theta,
    linear_regression_analytical_derivative_d_theta,
    linear_regression_analytical_derivative_d_x_d_theta,
)

from .conftest import create_mock_dataset

try:
    import torch.nn.functional as F

    from pydvl.influence.general import InfluenceType, compute_influences
    from pydvl.influence.model_wrappers import TorchLinearRegression, TorchMLP
    from pydvl.utils.dataset import load_wine_dataset
except ImportError:
    pass


class InfluenceTestSettings:
    DATA_OUTPUT_NOISE: float = 0.01
    ACCEPTABLE_ABS_TOL_INFLUENCE: float = 5e-4
    ACCEPTABLE_ABS_TOL_INFLUENCE_CG: float = 1e-3

    INFLUENCE_TEST_CONDITION_NUMBERS: List[int] = [5]
    INFLUENCE_TRAINING_SET_SIZE: List[int] = [500]
    INFLUENCE_TEST_SET_SIZE: List[int] = [20]
    INFLUENCE_N_JOBS: List[int] = [1]
    INFLUENCE_DIMENSIONS: List[Tuple[int, int]] = [
        (10, 10),
        (20, 10),
        (3, 20),
        (20, 20),
    ]


test_cases = list(
    itertools.product(
        InfluenceTestSettings.INFLUENCE_TRAINING_SET_SIZE,
        InfluenceTestSettings.INFLUENCE_TEST_SET_SIZE,
        InfluenceTestSettings.INFLUENCE_DIMENSIONS,
        InfluenceTestSettings.INFLUENCE_TEST_CONDITION_NUMBERS,
        InfluenceTestSettings.INFLUENCE_N_JOBS,
    )
)


def lmb_test_case_to_str(packed_i_test_case):
    i, test_case = packed_i_test_case
    return (
        f"Problem #{i} of dimension {test_case[2]} with train size {test_case[0]}, "
        f"test size {test_case[1]}, condition number {test_case[3]} and {test_case[4]} jobs."
    )


test_case_ids = list(map(lmb_test_case_to_str, zip(range(len(test_cases)), test_cases)))


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


@pytest.mark.torch
@pytest.mark.parametrize(
    "train_set_size,test_set_size,problem_dimension,condition_number,n_jobs",
    test_cases,
    ids=test_case_ids,
)
def test_upweighting_influences_lr_analytical_cg(
    train_set_size: int,
    test_set_size: int,
    condition_number: float,
    linear_model: Tuple[np.ndarray, np.ndarray],
    n_jobs: int,
):
    A, _ = linear_model
    train_data, test_data = create_mock_dataset(
        linear_model, train_set_size, test_set_size
    )

    model = TorchLinearRegression(A.shape[0], A.shape[1], init=linear_model)
    loss = F.mse_loss

    influence_values_analytical = 2 * influences_up_linear_regression_analytical(
        linear_model,
        *train_data,
        *test_data,
    )

    influence_values = compute_influences(
        model,
        loss,
        *train_data,
        *test_data,
        progress=True,
        influence_type="up",
        inversion_method="cg",
        inversion_method_kwargs={"rtol": 10e-7},
    )
    assert np.logical_not(np.any(np.isnan(influence_values)))
    assert influence_values.shape == (len(test_data[0]), len(train_data[0]))
    influences_max_abs_diff = np.max(
        np.abs(influence_values - influence_values_analytical)
    )
    assert (
        influences_max_abs_diff < InfluenceTestSettings.ACCEPTABLE_ABS_TOL_INFLUENCE_CG
    ), "Upweighting influence values were wrong."


@pytest.mark.torch
@pytest.mark.parametrize(
    "train_set_size,test_set_size,problem_dimension,condition_number,n_jobs",
    test_cases,
    ids=test_case_ids,
)
def test_upweighting_influences_lr_analytical(
    train_set_size: int,
    test_set_size: int,
    condition_number: float,
    linear_model: Tuple[np.ndarray, np.ndarray],
    n_jobs: int,
):

    A, _ = tuple(linear_model)
    train_data, test_data = create_mock_dataset(
        linear_model, train_set_size, test_set_size
    )

    model = TorchLinearRegression(A.shape[0], A.shape[1], init=linear_model)
    loss = F.mse_loss

    influence_values_analytical = 2 * influences_up_linear_regression_analytical(
        linear_model,
        *train_data,
        *test_data,
    )

    influence_values = compute_influences(
        model,
        loss,
        *train_data,
        *test_data,
        progress=True,
        influence_type="up",
    )
    assert np.logical_not(np.any(np.isnan(influence_values)))
    assert influence_values.shape == (len(test_data[0]), len(train_data[0]))
    influences_max_abs_diff = np.max(
        np.abs(influence_values - influence_values_analytical)
    )
    assert (
        influences_max_abs_diff < InfluenceTestSettings.ACCEPTABLE_ABS_TOL_INFLUENCE
    ), "Upweighting influence values were wrong."


@pytest.mark.torch
@pytest.mark.parametrize(
    "train_set_size,test_set_size,problem_dimension,condition_number,n_jobs",
    test_cases,
    ids=test_case_ids,
)
def test_perturbation_influences_lr_analytical_cg(
    train_set_size: int,
    test_set_size: int,
    problem_dimension: int,
    condition_number: float,
    linear_model: Tuple[np.ndarray, np.ndarray],
    n_jobs: int,
):
    train_data, test_data = create_mock_dataset(
        linear_model, train_set_size, test_set_size
    )
    A, _ = linear_model

    model = TorchLinearRegression(A.shape[0], A.shape[1], init=linear_model)
    loss = F.mse_loss

    influence_values_analytical = (
        2
        * influences_perturbation_linear_regression_analytical(
            linear_model,
            *train_data,
            *test_data,
        )
    )
    influence_values = compute_influences(
        model,
        loss,
        *train_data,
        *test_data,
        progress=True,
        influence_type="perturbation",
        inversion_method="cg",
        inversion_method_kwargs={"rtol": 10e-7},
    )
    assert np.logical_not(np.any(np.isnan(influence_values)))
    assert influence_values.shape == (
        len(test_data[0]),
        len(train_data[0]),
        A.shape[1],
    )
    influences_max_abs_diff = np.max(
        np.abs(influence_values - influence_values_analytical)
    )
    assert (
        influences_max_abs_diff < InfluenceTestSettings.ACCEPTABLE_ABS_TOL_INFLUENCE
    ), "Perturbation influence values were wrong."


@pytest.mark.torch
@pytest.mark.parametrize(
    "train_set_size,test_set_size",
    test_cases,
    ids=test_case_ids,
)
def test_perturbation_influences_lr_analytical(
    train_set_size: int,
    test_set_size: int,
    linear_model: Tuple[np.ndarray, np.ndarray],
):
    train_data, test_data = create_mock_dataset(
        linear_model, train_set_size, test_set_size
    )
    A, _ = linear_model

    model = TorchLinearRegression(A.shape[0], A.shape[1], init=linear_model)
    loss = F.mse_loss

    influence_values_analytical = (
        2
        * influences_perturbation_linear_regression_analytical(
            linear_model,
            *train_data,
            *test_data,
        )
    )
    influence_values = compute_influences(
        model,
        loss,
        *train_data,
        *test_data,
        progress=True,
        influence_type="perturbation",
    )
    assert np.logical_not(np.any(np.isnan(influence_values)))
    assert influence_values.shape == (
        len(test_data[0]),
        len(train_data[0]),
        A.shape[1],
    )
    influences_max_abs_diff = np.max(
        np.abs(influence_values - influence_values_analytical)
    )
    assert (
        influences_max_abs_diff < InfluenceTestSettings.ACCEPTABLE_ABS_TOL_INFLUENCE
    ), "Perturbation influence values were wrong."
