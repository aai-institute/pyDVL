from math import prod
from typing import Callable, NamedTuple, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from pydvl.influence.base_influence_function_model import (
    NotImplementedLayerRepresentationException,
)
from pydvl.influence.torch.influence_function_model import (
    ArnoldiInfluence,
    CgInfluence,
    DirectInfluence,
    EkfacInfluence,
    LissaInfluence,
    NystroemSketchInfluence,
)
from pydvl.influence.torch.pre_conditioner import (
    JacobiPreConditioner,
    NystroemPreConditioner,
    PreConditioner,
)
from pydvl.utils.exceptions import NotFittedException
from tests.influence.torch.conftest import minimal_training

torch = pytest.importorskip("torch")

import torch
import torch.nn.functional as F
from pytest_cases import fixture, parametrize, parametrize_with_cases
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pydvl.influence import InfluenceMode
from tests.influence.conftest import (
    add_noise_to_linear_model,
    analytical_linear_influences,
    linear_model,
)
from tests.influence.torch.test_util import (
    are_active_layers_linear,
    check_influence_correlations,
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
        nn.Linear(6, 2),
    )


def create_simple_nn_regr():
    return nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 3), nn.Linear(3, 1))


def create_conv1d_no_grad():
    return nn.Sequential(
        nn.Conv1d(in_channels=5, out_channels=3, kernel_size=2).requires_grad_(False),
        nn.Flatten(),
        nn.Linear(6, 2),
    )


def create_simple_nn_no_grad():
    return nn.Sequential(
        nn.Linear(10, 10).requires_grad_(False),
        nn.Linear(10, 5),
    )


class TestCase(NamedTuple):
    module_factory: Callable[[], nn.Module]
    input_dim: Tuple[int, ...]
    output_dim: int
    loss: nn.modules.loss._Loss
    mode: InfluenceMode
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
            mode=InfluenceMode.Up,
        )

    def case_conv3d_nn_pert(self) -> TestCase:
        return TestCase(
            module_factory=create_conv3d_nn,
            input_dim=(5, 3, 3, 3),
            output_dim=3,
            loss=nn.SmoothL1Loss(),
            mode=InfluenceMode.Perturbation,
        )

    def case_conv2d_nn_up(self) -> TestCase:
        return TestCase(
            module_factory=create_conv2d_nn,
            input_dim=(5, 5, 5),
            output_dim=3,
            loss=nn.MSELoss(),
            mode=InfluenceMode.Up,
        )

    def case_conv2d_nn_pert(self) -> TestCase:
        return TestCase(
            module_factory=create_conv2d_nn,
            input_dim=(5, 5, 5),
            output_dim=3,
            loss=nn.SmoothL1Loss(),
            mode=InfluenceMode.Perturbation,
        )

    def case_conv1d_nn_up(self) -> TestCase:
        return TestCase(
            module_factory=create_conv1d_nn,
            input_dim=(5, 3),
            output_dim=2,
            loss=nn.MSELoss(),
            mode=InfluenceMode.Up,
        )

    def case_conv1d_nn_pert(self) -> TestCase:
        return TestCase(
            module_factory=create_conv1d_nn,
            input_dim=(5, 3),
            output_dim=2,
            loss=nn.SmoothL1Loss(),
            mode=InfluenceMode.Perturbation,
        )

    def case_simple_nn_up(self) -> TestCase:
        return TestCase(
            module_factory=create_simple_nn_regr,
            input_dim=(10,),
            output_dim=1,
            loss=nn.MSELoss(),
            mode=InfluenceMode.Up,
        )

    def case_simple_nn_pert(self) -> TestCase:
        return TestCase(
            module_factory=create_simple_nn_regr,
            input_dim=(10,),
            output_dim=1,
            loss=nn.SmoothL1Loss(),
            mode=InfluenceMode.Perturbation,
        )

    def case_conv1d_no_grad_up(self) -> TestCase:
        return TestCase(
            module_factory=create_conv1d_no_grad,
            input_dim=(5, 3),
            output_dim=2,
            loss=nn.CrossEntropyLoss(),
            mode=InfluenceMode.Up,
        )

    def case_simple_nn_class_up(self) -> TestCase:
        return TestCase(
            module_factory=create_simple_nn_no_grad,
            input_dim=(10,),
            output_dim=5,
            loss=nn.CrossEntropyLoss(),
            mode=InfluenceMode.Up,
            train_data_len=100,
            test_data_len=30,
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
    x_test = torch.rand((test_case.test_data_len, *test_case.input_dim))
    if isinstance(test_case.loss, nn.CrossEntropyLoss):
        y_train = torch.randint(
            0, test_case.output_dim, (test_case.train_data_len,), dtype=torch.long
        )
        y_test = torch.randint(
            0, test_case.output_dim, (test_case.test_data_len,), dtype=torch.long
        )
    else:
        y_train = torch.rand((test_case.train_data_len, test_case.output_dim))
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
def direct_influence_function_model(model_and_data, test_case: TestCase):
    model, loss, x_train, y_train, x_test, y_test = model_and_data
    train_dataloader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=test_case.batch_size
    )
    return DirectInfluence(model, loss, test_case.hessian_reg).fit(train_dataloader)


@fixture
def direct_influences(
    direct_influence_function_model: DirectInfluence,
    model_and_data,
    test_case: TestCase,
) -> NDArray:
    model, loss, x_train, y_train, x_test, y_test = model_and_data
    return direct_influence_function_model.influences(
        x_test, y_test, x_train, y_train, mode=test_case.mode
    ).numpy()


@fixture
def direct_sym_influences(
    direct_influence_function_model: DirectInfluence,
    model_and_data,
    test_case: TestCase,
) -> NDArray:
    model, loss, x_train, y_train, x_test, y_test = model_and_data
    return direct_influence_function_model.influences(
        x_train, y_train, mode=test_case.mode
    ).numpy()


@fixture
def direct_factors(
    direct_influence_function_model: DirectInfluence,
    model_and_data,
    test_case: TestCase,
) -> NDArray:
    model, loss, x_train, y_train, x_test, y_test = model_and_data
    return direct_influence_function_model.influence_factors(x_train, y_train).numpy()


@pytest.mark.parametrize(
    "mode",
    InfluenceMode,
    ids=[ifl.value for ifl in InfluenceMode],
)
@pytest.mark.parametrize(
    "train_set_size",
    [200],
    ids=["train_set_size_200"],
)
@pytest.mark.parametrize(
    ["influence_factory", "rtol"],
    [
        [
            lambda model, loss, train_dataLoader, hessian_reg: CgInfluence(
                model, loss, hessian_regularization=hessian_reg
            ).fit(train_dataLoader),
            1e-1,
        ],
        [
            lambda model, loss, train_dataLoader, hessian_reg: LissaInfluence(
                model,
                loss,
                hessian_reg,
                maxiter=6000,
                scale=100,
            ).fit(train_dataLoader),
            0.3,
        ],
        [
            lambda model, loss, train_dataLoader, hessian_reg: DirectInfluence(
                model,
                loss,
                hessian_reg,
            ).fit(train_dataLoader),
            1e-4,
        ],
        [
            lambda model, loss, train_dataLoader, hessian_reg: CgInfluence(
                model,
                loss,
                hessian_regularization=hessian_reg,
                pre_conditioner=NystroemPreConditioner(10),
                use_block_cg=True,
            ).fit(train_dataLoader),
            1e-4,
        ],
    ],
    ids=["cg", "lissa", "direct", "block-cg"],
)
def test_influence_linear_model(
    influence_factory: Callable,
    rtol,
    mode: InfluenceMode,
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
        mode=mode,
        hessian_regularization=hessian_reg,
    )
    sym_analytical_influences = analytical_linear_influences(
        (A, b),
        *train_data,
        *train_data,
        mode=mode,
        hessian_regularization=hessian_reg,
    )

    train_data_set = TensorDataset(*list(map(torch.from_numpy, train_data)))
    train_data_loader = DataLoader(train_data_set, batch_size=40, num_workers=0)
    influence = influence_factory(linear_layer, loss, train_data_loader, hessian_reg)

    x_train, y_train = tuple(map(torch.from_numpy, train_data))
    x_test, y_test = tuple(map(torch.from_numpy, test_data))
    influence_values = influence.influences(
        x_test, y_test, x_train, y_train, mode=mode
    ).numpy()
    sym_influence_values = influence.influences(
        x_train, y_train, x_train, y_train, mode=mode
    ).numpy()

    with pytest.raises(ValueError):
        influence.influences(x_test, y_test, x=x_train, mode=mode)

    def upper_quantile_equivalence(
        approx_inf: NDArray, analytical_inf: NDArray, quantile: float
    ):
        abs_influence = np.abs(approx_inf)
        upper_quantile_mask = abs_influence > np.quantile(abs_influence, quantile)
        return np.allclose(
            approx_inf[upper_quantile_mask],
            analytical_inf[upper_quantile_mask],
            rtol=rtol,
        )

    assert np.logical_not(np.any(np.isnan(influence_values)))
    assert np.logical_not(np.any(np.isnan(sym_influence_values)))
    assert upper_quantile_equivalence(influence_values, analytical_influences, 0.9)
    assert upper_quantile_equivalence(
        sym_influence_values, sym_analytical_influences, 0.9
    )


@parametrize(
    "influence_factory",
    [
        lambda model, loss, train_dataLoader, hessian_reg: LissaInfluence(
            model,
            loss,
            hessian_regularization=hessian_reg,
            maxiter=150,
            scale=10000,
        ).fit(train_dataLoader),
    ],
    ids=["lissa"],
)
def test_influences_lissa(
    test_case: TestCase,
    model_and_data: Tuple[
        torch.nn.Module,
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    direct_influences,
    influence_factory,
):
    model, loss, x_train, y_train, x_test, y_test = model_and_data

    train_dataloader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=test_case.batch_size
    )
    influence_model = influence_factory(
        model, loss, train_dataloader, test_case.hessian_reg
    )
    approx_influences = influence_model.influences(
        x_test, y_test, x_train, y_train, mode=test_case.mode
    ).numpy()

    assert not np.any(np.isnan(approx_influences))

    assert np.allclose(approx_influences, direct_influences, rtol=1e-1)

    if test_case.mode == InfluenceMode.Up:
        assert approx_influences.shape == (
            test_case.test_data_len,
            test_case.train_data_len,
        )

    if test_case.mode == InfluenceMode.Perturbation:
        assert approx_influences.shape == (
            test_case.test_data_len,
            test_case.train_data_len,
            *test_case.input_dim,
        )

    # check that influences are not all constant
    assert not np.all(approx_influences == approx_influences.item(0))

    assert np.allclose(approx_influences, direct_influences, rtol=1e-1)


@pytest.mark.parametrize(
    "influence_factory",
    [
        lambda model, loss, hessian_reg, rank: ArnoldiInfluence(
            model,
            loss,
            hessian_regularization=hessian_reg,
            rank_estimate=rank,
            precompute_grad=True,
        ),
        lambda model, loss, hessian_reg, rank: NystroemSketchInfluence(
            model, loss, hessian_regularization=hessian_reg, rank=rank
        ),
    ],
    ids=["arnoldi", "nystroem"],
)
def test_influences_low_rank(
    test_case: TestCase,
    model_and_data: Tuple[
        torch.nn.Module,
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    direct_influences,
    direct_sym_influences,
    direct_factors,
    influence_factory,
):
    atol = 1e-8
    rtol = 1e-5
    model, loss, x_train, y_train, x_test, y_test = model_and_data

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_dataloader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=test_case.batch_size
    )

    influence_func_model = influence_factory(
        model,
        loss,
        test_case.hessian_reg,
        num_parameters - 1,
    )

    with pytest.raises(NotFittedException):
        influence_func_model.influences(
            x_test, y_test, x_train, y_train, mode=test_case.mode
        )

    with pytest.raises(NotFittedException):
        influence_func_model.influence_factors(x_test, y_test)

    influence_func_model = influence_func_model.fit(train_dataloader)

    low_rank_influence = influence_func_model.influences(
        x_test, y_test, x_train, y_train, mode=test_case.mode
    ).numpy()

    sym_low_rank_influence = influence_func_model.influences(
        x_train, y_train, mode=test_case.mode
    ).numpy()

    low_rank_factors = influence_func_model.influence_factors(x_test, y_test)
    assert np.allclose(
        direct_factors,
        influence_func_model.influence_factors(x_train, y_train).numpy(),
        atol=atol,
        rtol=rtol,
    )

    if test_case.mode is InfluenceMode.Up:
        low_rank_influence_transpose = influence_func_model.influences(
            x_train, y_train, x_test, y_test, mode=test_case.mode
        ).numpy()
        assert np.allclose(
            low_rank_influence_transpose, low_rank_influence.swapaxes(0, 1)
        )

    low_rank_values_from_factors = influence_func_model.influences_from_factors(
        low_rank_factors, x_train, y_train, mode=test_case.mode
    ).numpy()
    assert np.allclose(direct_influences, low_rank_influence, atol=atol, rtol=rtol)
    assert np.allclose(
        direct_sym_influences, sym_low_rank_influence, atol=atol, rtol=rtol
    )
    assert np.allclose(
        low_rank_influence, low_rank_values_from_factors, atol=atol, rtol=rtol
    )

    with pytest.raises(ValueError):
        influence_func_model.influences(x_test, y_test, x=x_train, mode=test_case.mode)
    with pytest.raises(ValueError):
        influence_func_model.influences(x_test, y_test, y=y_train, mode=test_case.mode)


def test_influences_ekfac(
    test_case: TestCase,
    model_and_data: Tuple[
        torch.nn.Module,
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    direct_influences,
    direct_sym_influences,
):
    model, loss, x_train, y_train, x_test, y_test = model_and_data

    train_dataloader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=test_case.batch_size
    )

    ekfac_influence = EkfacInfluence(
        model,
        update_diagonal=True,
        hessian_regularization=test_case.hessian_reg,
    )

    with pytest.raises(NotFittedException):
        ekfac_influence.influences(
            x_test, y_test, x_train, y_train, mode=test_case.mode
        )

    with pytest.raises(NotFittedException):
        ekfac_influence.influence_factors(x_test, y_test)

    if not are_active_layers_linear:
        with pytest.raises(NotImplementedLayerRepresentationException):
            ekfac_influence.fit(train_dataloader)
    elif isinstance(loss, nn.CrossEntropyLoss):
        ekfac_influence = ekfac_influence.fit(train_dataloader)
        ekfac_influence_values = ekfac_influence.influences(
            x_test, y_test, x_train, y_train, mode=test_case.mode
        ).numpy()

        ekfac_influences_by_layer = ekfac_influence.influences_by_layer(
            x_test, y_test, x_train, y_train, mode=test_case.mode
        )

        accumulated_inf_by_layer = np.zeros_like(ekfac_influence_values)
        for layer, infl in ekfac_influences_by_layer.items():
            accumulated_inf_by_layer += infl.detach().numpy()

        ekfac_self_influence = ekfac_influence.influences(
            x_train, y_train, mode=test_case.mode
        ).numpy()

        ekfac_factors = ekfac_influence.influence_factors(x_test, y_test)

        influence_from_factors = ekfac_influence.influences_from_factors(
            ekfac_factors, x_train, y_train, mode=test_case.mode
        ).numpy()

        assert np.allclose(ekfac_influence_values, influence_from_factors)
        assert np.allclose(ekfac_influence_values, accumulated_inf_by_layer)
        check_influence_correlations(direct_influences, ekfac_influence_values)
        check_influence_correlations(direct_sym_influences, ekfac_self_influence)


@pytest.mark.torch
@pytest.mark.parametrize("use_block_cg", [True, False])
@pytest.mark.parametrize(
    "pre_conditioner",
    [
        JacobiPreConditioner(),
        NystroemPreConditioner(rank=5),
        None,
    ],
)
def test_influences_cg(
    test_case: TestCase,
    model_and_data: Tuple[
        torch.nn.Module,
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    direct_influences,
    direct_factors,
    use_block_cg: bool,
    pre_conditioner: PreConditioner,
):
    model, loss, x_train, y_train, x_test, y_test = model_and_data

    train_dataloader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=test_case.batch_size
    )
    influence_model = CgInfluence(
        model,
        loss,
        test_case.hessian_reg,
        maxiter=5,
        pre_conditioner=pre_conditioner,
        use_block_cg=use_block_cg,
    )
    influence_model = influence_model.fit(train_dataloader)

    approx_influences = influence_model.influences(
        x_test, y_test, x_train, y_train, mode=test_case.mode
    ).numpy()

    assert not np.any(np.isnan(approx_influences))

    assert np.allclose(approx_influences, direct_influences, atol=1e-6, rtol=1e-4)

    if test_case.mode == InfluenceMode.Up:
        assert approx_influences.shape == (
            test_case.test_data_len,
            test_case.train_data_len,
        )

    if test_case.mode == InfluenceMode.Perturbation:
        assert approx_influences.shape == (
            test_case.test_data_len,
            test_case.train_data_len,
            *test_case.input_dim,
        )

    # check that influences are not all constant
    assert not np.all(approx_influences == approx_influences.item(0))

    assert np.allclose(approx_influences, direct_influences, atol=1e-6, rtol=1e-4)

    # check that block variant returns the correct vector, if only one right hand side
    # is provided
    if use_block_cg:
        single_influence = influence_model.influence_factors(
            x_train[0].unsqueeze(0), y_train[0].unsqueeze(0)
        ).numpy()
        assert np.allclose(single_influence, direct_factors[0], atol=1e-6, rtol=1e-4)
