import itertools
from copy import copy
from typing import List, Tuple

import numpy as np
import pytest
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam, lr_scheduler

from tests.conftest import create_mock_dataset
from valuation.influence.general import influences
from valuation.influence.linear import (
    influences_perturbation_linear_regression_analytical,
    influences_up_linear_regression_analytical,
    linear_influences,
)
from valuation.utils.dataset import load_wine_dataset

try:
    import torch
    import torch.nn.functional as F

    from valuation.influence.frameworks import TorchTwiceDifferentiable
    from valuation.influence.model_wrappers import (
        TorchLinearRegression,
        TorchNeuralNetwork,
    )
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

    influence_values_analytical = -2 * influences_up_linear_regression_analytical(
        linear_model,
        *train_data,
        *test_data,
    )

    influence_values = influences(
        model,
        loss,
        *train_data,
        *test_data,
        progress=True,
        influence_type="up",
        inversion_method="cg",
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

    influence_values_analytical = -2 * influences_up_linear_regression_analytical(
        linear_model,
        *train_data,
        *test_data,
    )

    influence_values = influences(
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
        -2
        * influences_perturbation_linear_regression_analytical(
            linear_model,
            *train_data,
            *test_data,
        )
    )
    influence_values = influences(
        model,
        loss,
        *train_data,
        *test_data,
        progress=True,
        influence_type="perturbation",
        inversion_method="cg",
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
    "train_set_size,test_set_size,problem_dimension,condition_number,n_jobs",
    test_cases,
    ids=test_case_ids,
)
def test_perturbation_influences_lr_analytical(
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
        -2
        * influences_perturbation_linear_regression_analytical(
            linear_model,
            *train_data,
            *test_data,
        )
    )
    influence_values = influences(
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


@pytest.mark.torch
@pytest.mark.parametrize(
    "train_set_size,test_set_size,problem_dimension,condition_number",
    itertools.product(
        InfluenceTestSettings.INFLUENCE_TRAINING_SET_SIZE,
        InfluenceTestSettings.INFLUENCE_TEST_SET_SIZE,
        InfluenceTestSettings.INFLUENCE_DIMENSIONS,
        InfluenceTestSettings.INFLUENCE_TEST_CONDITION_NUMBERS,
    ),
)
def test_linear_influences_up_perturbations_analytical(
    train_set_size: int,
    test_set_size: int,
    problem_dimension: int,
    condition_number: float,
    linear_model: Tuple[np.ndarray, np.ndarray],
):
    train_data, test_data = create_mock_dataset(
        linear_model, train_set_size, test_set_size
    )
    up_influences = linear_influences(
        *train_data,
        *test_data,
        influence_type="up",
    )
    assert np.logical_not(np.any(np.isnan(up_influences)))
    assert up_influences.shape == (len(test_data[0]), len(train_data[0]))

    pert_influences = linear_influences(
        *train_data,
        *test_data,
        influence_type="perturbation",
    )
    assert np.logical_not(np.any(np.isnan(pert_influences)))
    assert pert_influences.shape == (
        len(test_data[0]),
        len(train_data[0]),
        train_data[0].shape[1],
    )


@pytest.mark.torch
def test_influences_with_neural_network_explicit_hessian():
    train_ds, val_ds, test_ds = load_wine_dataset(train_size=0.3, test_size=0.6)
    feature_dimension = train_ds[0].shape[1]
    unique_classes = np.unique(np.concatenate((train_ds[1], test_ds[1])))
    num_classes = len(unique_classes)
    num_epochs = 300
    network_size = [16, 16]
    nn = TorchNeuralNetwork(feature_dimension, num_classes, network_size)
    optimizer = Adam(params=nn.parameters(), lr=0.001, weight_decay=0.001)
    loss = F.cross_entropy
    nn.fit(
        *train_ds,
        *test_ds,
        num_epochs=num_epochs,
        batch_size=32,
        loss=loss,
        optimizer=optimizer,
        scheduler=lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs),
    )

    model = nn
    loss = loss

    train_influences = influences(
        model,
        loss,
        *train_ds,
        *test_ds,
        inversion_method="direct",
    )

    assert np.all(np.logical_not(np.isnan(train_influences)))
