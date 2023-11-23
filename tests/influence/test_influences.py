from dataclasses import dataclass
from math import prod
from typing import Callable, Dict, NamedTuple, Tuple, Optional

import numpy as np
import pytest
from numpy.typing import NDArray

from pydvl.influence.torch.influence_model import (
    ArnoldiInfluence,
    BatchCgInfluence,
    DirectInfluence,
    LissaInfluence, TorchInfluence,
)
from pydvl.influence.twice_differentiable import Influence

from .torch.conftest import minimal_training

torch = pytest.importorskip("torch")

import torch
import torch.nn.functional as F
from pytest_cases import fixture, parametrize, parametrize_with_cases
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pydvl.influence import InfluenceType, InversionMethod, compute_influences
from pydvl.influence.torch import TorchTwiceDifferentiable, model_hessian_low_rank

from .conftest import (
    add_noise_to_linear_model,
    analytical_linear_influences,
    linear_model,
)

# Mark the entire module
pytestmark = pytest.mark.torch


def create_conv3d_nn():
    return nn.Sequential(
        nn.Conv3d(in_channels=5, out_channels=3, kernel_size=2),
        nn.Flatten(),
        nn.Linear(24, 3),
    )


def create_conv2d_nn():
    return nn.Sequential(
        nn.Conv2d(in_channels=5, out_channels=3, kernel_size=3),
        nn.Flatten(),
        nn.Linear(27, 3),
    )


def create_conv1d_nn():
    return nn.Sequential(
        nn.Conv1d(in_channels=5, out_channels=3, kernel_size=2),
        nn.Flatten(),
        nn.Linear(6, 3),
    )


def create_simple_nn_regr():
    return nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 3), nn.Linear(3, 1))


class TestCase(NamedTuple):
    module_factory: Callable[[], nn.Module]
    input_dim: Tuple[int, ...]
    output_dim: int
    loss: nn.modules.loss._Loss
    influence_type: InfluenceType
    hessian_reg: float = 1e3
    train_data_len: int = 20
    test_data_len: int = 10
    batch_size: int = 10


class InfluenceTestCases:
    def case_conv3d_nn_up(self) -> TestCase:
        return TestCase(
            module_factory=create_conv3d_nn,
            input_dim=(5, 3, 3, 3),
            output_dim=3,
            loss=nn.MSELoss(),
            influence_type=InfluenceType.Up,
        )

    def case_conv3d_nn_pert(self) -> TestCase:
        return TestCase(
            module_factory=create_conv3d_nn,
            input_dim=(5, 3, 3, 3),
            output_dim=3,
            loss=nn.SmoothL1Loss(),
            influence_type=InfluenceType.Perturbation,
        )

    def case_conv2d_nn_up(self) -> TestCase:
        return TestCase(
            module_factory=create_conv2d_nn,
            input_dim=(5, 5, 5),
            output_dim=3,
            loss=nn.MSELoss(),
            influence_type=InfluenceType.Up,
        )

    def case_conv2d_nn_pert(self) -> TestCase:
        return TestCase(
            module_factory=create_conv2d_nn,
            input_dim=(5, 5, 5),
            output_dim=3,
            loss=nn.SmoothL1Loss(),
            influence_type=InfluenceType.Perturbation,
        )

    def case_conv1d_nn_up(self) -> TestCase:
        return TestCase(
            module_factory=create_conv1d_nn,
            input_dim=(5, 3),
            output_dim=3,
            loss=nn.MSELoss(),
            influence_type=InfluenceType.Up,
        )

    def case_conv1d_nn_pert(self) -> TestCase:
        return TestCase(
            module_factory=create_conv1d_nn,
            input_dim=(5, 3),
            output_dim=3,
            loss=nn.SmoothL1Loss(),
            influence_type=InfluenceType.Perturbation,
        )

    def case_simple_nn_up(self) -> TestCase:
        return TestCase(
            module_factory=create_simple_nn_regr,
            input_dim=(10,),
            output_dim=1,
            loss=nn.MSELoss(),
            influence_type=InfluenceType.Up,
        )

    def case_simple_nn_pert(self) -> TestCase:
        return TestCase(
            module_factory=create_simple_nn_regr,
            input_dim=(10,),
            output_dim=1,
            loss=nn.SmoothL1Loss(),
            influence_type=InfluenceType.Perturbation,
        )


@fixture
@parametrize_with_cases(
    "case",
    cases=InfluenceTestCases,
    scope="module",
)
def test_case(case: TestCase) -> TestCase:
    return case


@fixture
def model_and_data(
    test_case: TestCase,
) -> Tuple[
    torch.nn.Module,
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    x_train = torch.rand((test_case.train_data_len, *test_case.input_dim))
    y_train = torch.rand((test_case.train_data_len, test_case.output_dim))
    x_test = torch.rand((test_case.test_data_len, *test_case.input_dim))
    y_test = torch.rand((test_case.test_data_len, test_case.output_dim))

    train_dataloader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=test_case.batch_size
    )

    model = test_case.module_factory()
    model = minimal_training(
        model, train_dataloader, test_case.loss, lr=0.3, epochs=100
    )
    model.eval()
    return model, test_case.loss, x_train, y_train, x_test, y_test


@fixture
def direct_influence(model_and_data, test_case: TestCase) -> NDArray:
    model, loss, x_train, y_train, x_test, y_test = model_and_data
    train_dataloader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=test_case.batch_size
    )
    direct_influence = DirectInfluence(
        model, loss, test_case.hessian_reg, train_dataloader=train_dataloader
    )
    return direct_influence.values(
        x_test, y_test, x_train, y_train, influence_type=test_case.influence_type
    ).numpy()

   
@pytest.mark.parametrize(
    "influence_type",
    InfluenceType,
    ids=[ifl.value for ifl in InfluenceType],
)
@pytest.mark.parametrize(
    "train_set_size",
    [200],
    ids=["train_set_size_200"],
)
@pytest.mark.parametrize(
    "inversion_method, inversion_method_kwargs, rtol",
    [
        [InversionMethod.Direct, {}, 1e-7],
        [InversionMethod.Cg, {}, 1e-1],
        [InversionMethod.Lissa, {"maxiter": 6000, "scale": 100}, 0.3],
    ],
    ids=[inv.value for inv in InversionMethod if inv is not InversionMethod.Arnoldi],
)
def test_influence_linear_model(
    influence_type: InfluenceType,
    inversion_method: InversionMethod,
    inversion_method_kwargs: Dict,
    rtol: float,
    train_set_size: int,
    hessian_reg: float = 0.1,
    test_set_size: int = 20,
    problem_dimension: Tuple[int, int] = (4, 20),
    condition_number: float = 2,
):
    A, b = linear_model(problem_dimension, condition_number)
    train_data, test_data = add_noise_to_linear_model(
        (A, b), train_set_size, test_set_size
    )

    linear_layer = nn.Linear(A.shape[0], A.shape[1])
    linear_layer.eval()
    linear_layer.weight.data = torch.as_tensor(A)
    linear_layer.bias.data = torch.as_tensor(b)
    loss = F.mse_loss

    analytical_influences = analytical_linear_influences(
        (A, b),
        *train_data,
        *test_data,
        influence_type=influence_type,
        hessian_regularization=hessian_reg,
    )

    train_data_set = TensorDataset(*list(map(torch.from_numpy, train_data)))
    test_data_set = TensorDataset(*list(map(torch.from_numpy, test_data)))
    train_data_loader = DataLoader(train_data_set, batch_size=40, num_workers=0)
    input_data = DataLoader(train_data_set, batch_size=40)
    test_data_loader = DataLoader(
        test_data_set,
        batch_size=40,
    )

    influence_values = compute_influences(
        TorchTwiceDifferentiable(linear_layer, loss),
        training_data=train_data_loader,
        test_data=test_data_loader,
        input_data=input_data,
        progress=False,
        influence_type=influence_type,
        inversion_method=inversion_method,
        hessian_regularization=hessian_reg,
        **inversion_method_kwargs,
    )

    influence_values = influence_values.numpy()

    assert np.logical_not(np.any(np.isnan(influence_values)))
    abs_influence = np.abs(influence_values)
    upper_quantile_mask = abs_influence > np.quantile(abs_influence, 0.9)
    assert np.allclose(
        influence_values[upper_quantile_mask],
        analytical_influences[upper_quantile_mask],
        rtol=rtol,
    )


@parametrize(
    "influence_factory",
    [
        lambda model, loss, train_dataLoader, hessian_reg: BatchCgInfluence(
            model, loss, train_dataLoader, hessian_regularization=hessian_reg
        ),
        lambda model, loss, train_dataLoader, hessian_reg: LissaInfluence(
            model,
            loss,
            train_dataLoader,
            hessian_regularization=hessian_reg,
            maxiter=150,
            scale=10000,
        ),
    ],
    ids=["cg", "lissa"],
)
def test_influences_nn(
    test_case: TestCase,
    model_and_data: Tuple[
        torch.nn.Module,
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    direct_influence,
    influence_factory,
):
    model, loss, x_train, y_train, x_test, y_test = model_and_data

    train_dataloader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=test_case.batch_size
    )
    influence_model = influence_factory(
        model, loss, train_dataloader, test_case.hessian_reg
    )
    approx_influences = influence_model.values(
        x_test, y_test, x_train, y_train, influence_type=test_case.influence_type
    ).numpy()

    assert not np.any(np.isnan(approx_influences))

    assert np.allclose(approx_influences, direct_influence, rtol=1e-1)

    if test_case.influence_type == InfluenceType.Up:
        assert approx_influences.shape == (
            test_case.test_data_len,
            test_case.train_data_len,
        )

    if test_case.influence_type == InfluenceType.Perturbation:
        assert approx_influences.shape == (
            test_case.test_data_len,
            test_case.train_data_len,
            prod(test_case.input_dim),
        )

    # check that influences are not all constant
    assert not np.all(approx_influences == approx_influences.item(0))

    assert np.allclose(approx_influences, direct_influence, rtol=1e-1)


def test_influences_arnoldi(
    test_case: TestCase,
    model_and_data: Tuple[
        torch.nn.Module,
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    direct_influence,
):
    model, loss, x_train, y_train, x_test, y_test = model_and_data

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_dataloader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=test_case.batch_size
    )
    arnoldi_influence = ArnoldiInfluence(
        model,
        loss,
        train_dataloader=train_dataloader,
        hessian_regularization=test_case.hessian_reg,
        rank_estimate=num_parameters - 1,
    )
    low_rank_influence = arnoldi_influence.values(
        x_test, y_test, x_train, y_train, influence_type=test_case.influence_type
    ).numpy()

    assert np.allclose(direct_influence, low_rank_influence)

    precomputed_low_rank = model_hessian_low_rank(
        model,
        loss,
        training_data=train_dataloader,
        hessian_perturbation=0.0,
        rank_estimate=num_parameters - 1,
    )
    arnoldi_influence = ArnoldiInfluence(
        model,
        loss,
        low_rank_representation=precomputed_low_rank,
        hessian_regularization=test_case.hessian_reg,
        rank_estimate=num_parameters - 1,
    )
    precomputed_low_rank_influence = arnoldi_influence.values(
        x_test, y_test, x_train, y_train, influence_type=test_case.influence_type
    ).numpy()

    assert np.allclose(direct_influence, precomputed_low_rank_influence)
