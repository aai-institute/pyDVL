import itertools
from copy import copy
from typing import List, Tuple

import numpy as np
import pytest
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler

from tests.conftest import create_mock_dataset
from valuation.influence.linear import (
    influences_perturbation_linear_regression_analytical,
    influences_up_linear_regression_analytical,
    linear_influences,
)
from valuation.utils import Dataset

try:
    import torch
    import torch.nn.functional as F
    from torch.optim import Adam, lr_scheduler

    from valuation.influence.general import influences
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
    dataset = create_mock_dataset(linear_model, train_set_size, test_set_size)

    model = TorchLinearRegression(A.shape[0], A.shape[1], init=linear_model)
    loss = F.mse_loss

    influence_values_analytical = 2 * influences_up_linear_regression_analytical(
        linear_model,
        dataset,
    )

    influence_values = influences(
        model,
        loss,
        dataset,
        progress=True,
        influence_type="up",
        inversion_method="cg",
    )
    assert np.logical_not(np.any(np.isnan(influence_values)))
    assert influence_values.shape == (len(dataset.x_test), len(dataset.x_train))
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
    dataset = create_mock_dataset(linear_model, train_set_size, test_set_size)

    model = TorchLinearRegression(A.shape[0], A.shape[1], init=linear_model)
    loss = F.mse_loss

    influence_values_analytical = 2 * influences_up_linear_regression_analytical(
        linear_model,
        dataset,
    )

    influence_values = influences(
        model,
        loss,
        dataset,
        progress=True,
        influence_type="up",
    )
    assert np.logical_not(np.any(np.isnan(influence_values)))
    assert influence_values.shape == (len(dataset.x_test), len(dataset.x_train))
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
    dataset = create_mock_dataset(linear_model, train_set_size, test_set_size)
    A, _ = linear_model

    model = TorchLinearRegression(A.shape[0], A.shape[1], init=linear_model)
    loss = F.mse_loss

    influence_values_analytical = (
        2
        * influences_perturbation_linear_regression_analytical(
            linear_model,
            dataset,
        )
    )
    influence_values = influences(
        model,
        loss,
        dataset,
        progress=True,
        influence_type="perturbation",
        inversion_method="cg",
    )
    assert np.logical_not(np.any(np.isnan(influence_values)))
    assert influence_values.shape == (
        len(dataset.x_test),
        len(dataset.x_train),
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
    dataset = create_mock_dataset(linear_model, train_set_size, test_set_size)
    A, _ = linear_model

    model = TorchLinearRegression(A.shape[0], A.shape[1], init=linear_model)
    loss = F.mse_loss

    influence_values_analytical = (
        2
        * influences_perturbation_linear_regression_analytical(
            linear_model,
            dataset,
        )
    )
    influence_values = influences(
        model,
        loss,
        dataset,
        progress=True,
        influence_type="perturbation",
    )
    assert np.logical_not(np.any(np.isnan(influence_values)))
    assert influence_values.shape == (
        len(dataset.x_test),
        len(dataset.x_train),
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
    dataset = create_mock_dataset(linear_model, train_set_size, test_set_size)
    up_influences = linear_influences(
        dataset,
        influence_type="up",
    )
    assert np.logical_not(np.any(np.isnan(up_influences)))
    assert up_influences.shape == (len(dataset.x_test), len(dataset.x_train))

    pert_influences = linear_influences(
        dataset,
        influence_type="perturbation",
    )
    assert np.logical_not(np.any(np.isnan(pert_influences)))
    assert pert_influences.shape == (
        len(dataset.x_test),
        len(dataset.x_train),
        dataset.x_train.shape[1],
    )


@pytest.mark.torch
def test_influences_with_neural_network_explicit_hessian():
    dataset = Dataset.from_sklearn(load_wine())
    x_transformer = MinMaxScaler()
    transformed_dataset = copy(dataset)
    transformed_dataset.x_train = torch.tensor(
        x_transformer.fit_transform(transformed_dataset.x_train), dtype=torch.float
    )
    transformed_dataset.y_train = torch.tensor(
        transformed_dataset.y_train, dtype=torch.long
    )
    transformed_dataset.x_test = torch.tensor(
        x_transformer.transform(transformed_dataset.x_test), dtype=torch.float
    )
    transformed_dataset.y_test = torch.tensor(
        transformed_dataset.y_test, dtype=torch.long
    )
    feature_dimension = dataset.x_train.shape[1]
    unique_classes = np.unique(np.concatenate((dataset.y_train, dataset.y_test)))
    num_classes = len(unique_classes)
    num_epochs = 300
    network_size = [16, 16]
    nn = TorchNeuralNetwork(feature_dimension, num_classes, network_size)
    optimizer = Adam(params=nn.parameters(), lr=0.001, weight_decay=0.001)
    loss = F.cross_entropy
    nn.fit(
        x=transformed_dataset.x_train,
        y=transformed_dataset.y_train,
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
        transformed_dataset,
        inversion_method="direct",
    )

    assert np.all(np.logical_not(np.isnan(train_influences)))
