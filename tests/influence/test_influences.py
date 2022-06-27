import itertools
from typing import List, Tuple

import numpy as np
import pytest

from tests.conftest import create_mock_dataset
from valuation.influence.general import influences
from valuation.influence.linear import (
    influences_perturbation_linear_regression_analytical,
    influences_up_linear_regression_analytical,
    linear_influences,
)
from valuation.influence.types import InfluenceTypes

try:
    import torch.nn.functional as F

    from valuation.models.linear_regression_torch_model import LRTorchModel
    from valuation.models.pytorch_model import PyTorchOptimizer, PyTorchSupervisedModel
except ImportError:
    pass


class InfluenceTestSettings:
    DATA_OUTPUT_NOISE: float = 0.01
    ACCEPTABLE_ABS_TOL_INFLUENCE: float = 1e-4

    INFLUENCE_TEST_CONDITION_NUMBERS: List[int] = [5]
    INFLUENCE_TRAINING_SET_SIZE: List[int] = [500]
    INFLUENCE_TEST_SET_SIZE: List[int] = [20]
    INFLUENCE_N_JOBS: List[int] = [1, 2]
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
    return f"Problem #{i} of dimension {test_case[2]} with train size {test_case[0]}, test size {test_case[1]}, condition number {test_case[3]} and {test_case[4]} jobs."


test_case_ids = list(map(lmb_test_case_to_str, zip(range(len(test_cases)), test_cases)))


@pytest.mark.torch
@pytest.mark.skip("Conjugate gradient sometimes is not accurate.")
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

    A, _ = tuple(linear_model)
    dataset = create_mock_dataset(linear_model, train_set_size, test_set_size)

    model = PyTorchSupervisedModel(
        model=LRTorchModel(dim=tuple(A.shape), init=linear_model),
        objective=F.mse_loss,
    )

    influence_values_analytical = 2 * influences_up_linear_regression_analytical(
        linear_model, dataset
    )

    influence_values = influences(
        model,
        dataset,
        progress=True,
        n_jobs=n_jobs,
        influence_type=InfluenceTypes.Up,
        use_conjugate_gradient=True,
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
def test_upweighting_influences_lr_analytical(
    train_set_size: int,
    test_set_size: int,
    condition_number: float,
    linear_model: Tuple[np.ndarray, np.ndarray],
    n_jobs: int,
):

    A, _ = tuple(linear_model)
    dataset = create_mock_dataset(linear_model, train_set_size, test_set_size)

    model = PyTorchSupervisedModel(
        model=LRTorchModel(dim=tuple(A.shape), init=linear_model),
        objective=F.mse_loss,
    )

    influence_values_analytical = 2 * influences_up_linear_regression_analytical(
        linear_model, dataset
    )

    influence_values = influences(
        model,
        dataset,
        progress=True,
        n_jobs=n_jobs,
        influence_type=InfluenceTypes.Up,
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
@pytest.mark.skip("Conjugate gradient sometimes is not accurate.")
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
    model = PyTorchSupervisedModel(
        model=LRTorchModel(dim=tuple(A.shape), init=linear_model), objective=F.mse_loss
    )

    influence_values_analytical = (
        2 * influences_perturbation_linear_regression_analytical(linear_model, dataset)
    )
    influence_values = influences(
        model,
        dataset,
        progress=True,
        n_jobs=n_jobs,
        influence_type=InfluenceTypes.Perturbation,
        use_conjugate_gradient=True,
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
    model = PyTorchSupervisedModel(
        model=LRTorchModel(dim=tuple(A.shape), init=linear_model), objective=F.mse_loss
    )

    influence_values_analytical = (
        2 * influences_perturbation_linear_regression_analytical(linear_model, dataset)
    )
    influence_values = influences(
        model,
        dataset,
        progress=True,
        n_jobs=n_jobs,
        influence_type=InfluenceTypes.Perturbation,
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
    up_influences = linear_influences(dataset, influence_type=InfluenceTypes.Up)
    assert np.logical_not(np.any(np.isnan(up_influences)))
    assert up_influences.shape == (len(dataset.x_test), len(dataset.x_train))

    pert_influences = linear_influences(
        dataset, influence_type=InfluenceTypes.Perturbation
    )
    assert np.logical_not(np.any(np.isnan(pert_influences)))
    assert pert_influences.shape == (
        len(dataset.x_test),
        len(dataset.x_train),
        dataset.x_train.shape[1],
    )
