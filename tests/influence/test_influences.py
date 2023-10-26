from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn
from torch.optim import LBFGS
from torch.utils.data import DataLoader, TensorDataset

from pydvl.influence import InfluenceType, InversionMethod, compute_influences
from pydvl.influence.torch import TorchTwiceDifferentiable, model_hessian_low_rank

from .conftest import (
    add_noise_to_linear_model,
    linear_analytical_influence_factors,
    linear_derivative_analytical,
    linear_mixed_second_derivative_analytical,
    linear_model,
)


def analytical_linear_influences(
    linear_model: Tuple[NDArray[np.float_], NDArray[np.float_]],
    x: NDArray[np.float_],
    y: NDArray[np.float_],
    x_test: NDArray[np.float_],
    y_test: NDArray[np.float_],
    influence_type: InfluenceType = InfluenceType.Up,
    hessian_regularization: float = 0,
):
    """Calculates analytically the influence of each training sample on the
     test samples for an ordinary least squares model (Ax+b=y with quadratic
     loss).

    :param linear_model: A tuple of arrays of shapes (N, M) and N representing A
        and b respectively.
    :param x: An array of shape (M, K) containing the features of the
        training set.
    :param y: An array of shape (M, L) containing the targets of the
        training set.
    :param x_test: An array of shape (N, K) containing the features of the test
        set.
    :param y_test: An array of shape (N, L) containing the targets of the test
        set.
    :param influence_type: the type of the influence.
    :param hessian_regularization: regularization value for the hessian
    :returns: An array of shape (B, C) with the influences of the training points
        on the test points if influence_type is "up", an array of shape (K, L,
        M) if influence_type is "perturbation".
    """

    s_test_analytical = linear_analytical_influence_factors(
        linear_model, x, y, x_test, y_test, hessian_regularization
    )
    if influence_type == InfluenceType.Up:
        train_grads_analytical = linear_derivative_analytical(
            linear_model,
            x,
            y,
        )
        result: NDArray = np.einsum(
            "ia,ja->ij", s_test_analytical, train_grads_analytical
        )
    elif influence_type == InfluenceType.Perturbation:
        train_second_deriv_analytical = linear_mixed_second_derivative_analytical(
            linear_model,
            x,
            y,
        )
        result: NDArray = np.einsum(
            "ia,jab->ijb", s_test_analytical, train_second_deriv_analytical
        )
    return result


@pytest.mark.torch
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

    train_data_loader = DataLoader(list(zip(*train_data)), batch_size=40, shuffle=True)
    input_data = DataLoader(list(zip(*train_data)), batch_size=40)
    test_data_loader = DataLoader(
        list(zip(*test_data)),
        batch_size=40,
    )

    influence_values = compute_influences(
        TorchTwiceDifferentiable(linear_layer, loss),
        training_data=train_data_loader,
        test_data=test_data_loader,
        input_data=input_data,
        progress=True,
        influence_type=influence_type,
        inversion_method=inversion_method,
        hessian_regularization=hessian_reg,
        **inversion_method_kwargs,
    ).numpy()

    assert np.logical_not(np.any(np.isnan(influence_values)))
    abs_influence = np.abs(influence_values)
    upper_quantile_mask = abs_influence > np.quantile(abs_influence, 0.9)
    assert np.allclose(
        influence_values[upper_quantile_mask],
        analytical_influences[upper_quantile_mask],
        rtol=rtol,
    )


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


@dataclass
class TestCase:
    case_id: str
    module_factory: Callable[[], nn.Module]
    input_dim: Tuple[int, ...]
    output_dim: int
    loss: nn.modules.loss._Loss
    influence_type: InfluenceType


@pytest.fixture
def test_case(request):
    return request.param


test_cases = [
    TestCase(
        case_id="conv3d_nn_up",
        module_factory=create_conv3d_nn,
        input_dim=(5, 3, 3, 3),
        output_dim=3,
        loss=nn.MSELoss(),
        influence_type=InfluenceType.Up,
    ),
    TestCase(
        case_id="conv3d_nn_pert",
        module_factory=create_conv3d_nn,
        input_dim=(5, 3, 3, 3),
        output_dim=3,
        loss=nn.SmoothL1Loss(),
        influence_type=InfluenceType.Perturbation,
    ),
    TestCase(
        case_id="conv2d_nn_up",
        module_factory=create_conv2d_nn,
        input_dim=(5, 5, 5),
        output_dim=3,
        loss=nn.MSELoss(),
        influence_type=InfluenceType.Up,
    ),
    TestCase(
        case_id="conv2d_nn_pert",
        module_factory=create_conv2d_nn,
        input_dim=(5, 5, 5),
        output_dim=3,
        loss=nn.SmoothL1Loss(),
        influence_type=InfluenceType.Perturbation,
    ),
    TestCase(
        case_id="conv1d_nn_up",
        module_factory=create_conv1d_nn,
        input_dim=(5, 3),
        output_dim=3,
        loss=nn.MSELoss(),
        influence_type=InfluenceType.Up,
    ),
    TestCase(
        case_id="conv1d_nn_pert",
        module_factory=create_conv1d_nn,
        input_dim=(5, 3),
        output_dim=3,
        loss=nn.SmoothL1Loss(),
        influence_type=InfluenceType.Perturbation,
    ),
    TestCase(
        case_id="simple_nn_up",
        module_factory=create_simple_nn_regr,
        input_dim=(10,),
        output_dim=1,
        loss=nn.MSELoss(),
        influence_type=InfluenceType.Up,
    ),
    TestCase(
        case_id="simple_nn_pert",
        module_factory=create_simple_nn_regr,
        input_dim=(10,),
        output_dim=1,
        loss=nn.SmoothL1Loss(),
        influence_type=InfluenceType.Perturbation,
    ),
]


def create_random_data_loader(
    input_dim: Tuple[int],
    output_dim: int,
    data_len: int,
    batch_size: int = 1,
) -> DataLoader:
    """
    Creates DataLoader instances with random data for testing purposes.

    :param input_dim: The dimensions of the input data.
    :param output_dim: The dimension of the output data.
    :param data_len: The length of the training dataset to be generated.
    :param batch_size: The size of the batches to be used in the DataLoader.

    :return: DataLoader instances for data.
    """
    x = torch.rand((data_len, *input_dim))
    y = torch.rand((data_len, output_dim))

    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


@pytest.mark.torch
@pytest.mark.parametrize(
    "inversion_method,inversion_method_kwargs",
    [
        ("cg", {}),
        (
            "lissa",
            {
                "maxiter": 150,
                "scale": 10000,
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "test_case",
    test_cases,
    ids=[case.case_id for case in test_cases],
    indirect=["test_case"],
)
def test_influences_nn(
    test_case: TestCase,
    inversion_method: InversionMethod,
    inversion_method_kwargs: Dict,
    data_len: int = 20,
    hessian_reg: float = 1e3,
    test_data_len: int = 10,
    batch_size: int = 10,
):
    module_factory = test_case.module_factory
    input_dim = test_case.input_dim
    output_dim = test_case.output_dim
    loss = test_case.loss
    influence_type = test_case.influence_type

    train_data_loader = create_random_data_loader(
        input_dim, output_dim, data_len, batch_size
    )
    test_data_loader = create_random_data_loader(
        input_dim, output_dim, test_data_len, batch_size
    )

    model = module_factory()
    model.eval()
    model = TorchTwiceDifferentiable(model, loss)

    direct_influence = compute_influences(
        model,
        training_data=train_data_loader,
        test_data=test_data_loader,
        progress=True,
        influence_type=influence_type,
        inversion_method=InversionMethod.Direct,
        hessian_regularization=hessian_reg,
    )

    approx_influences = compute_influences(
        model,
        training_data=train_data_loader,
        test_data=test_data_loader,
        progress=True,
        influence_type=influence_type,
        inversion_method=inversion_method,
        hessian_regularization=hessian_reg,
        **inversion_method_kwargs,
    ).numpy()
    assert not np.any(np.isnan(approx_influences))

    assert np.allclose(approx_influences, direct_influence, rtol=1e-1)

    if influence_type == InfluenceType.Up:
        assert approx_influences.shape == (test_data_len, data_len)

    if influence_type == InfluenceType.Perturbation:
        assert approx_influences.shape == (test_data_len, data_len, *input_dim)

    # check that influences are not all constant
    assert not np.all(approx_influences == approx_influences.item(0))


def minimal_training(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_function: torch.nn.modules.loss._Loss,
    lr=0.01,
    epochs=50,
):
    """
    Trains a PyTorch model using L-BFGS optimizer.

    :param model: The PyTorch model to be trained.
    :param dataloader: DataLoader providing the training data.
    :param loss_function: The loss function to be used for training.
    :param lr: The learning rate for the L-BFGS optimizer. Defaults to 0.01.
    :param epochs: The number of training epochs. Defaults to 50.

    :return: The trained model.
    """
    model = model.train()
    optimizer = LBFGS(model.parameters(), lr=lr)

    for epoch in range(epochs):
        data = torch.cat([inputs for inputs, targets in dataloader])
        targets = torch.cat([targets for inputs, targets in dataloader])

        def closure():
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_function(outputs, targets)
            loss.backward()
            return loss

        optimizer.step(closure)

    return model


@pytest.mark.torch
@pytest.mark.parametrize(
    "test_case",
    test_cases,
    ids=[case.case_id for case in test_cases],
    indirect=["test_case"],
)
def test_influences_arnoldi(
    test_case: TestCase,
    data_len: int = 20,
    hessian_reg: float = 20.0,
    test_data_len: int = 10,
):
    module_factory = test_case.module_factory
    input_dim = test_case.input_dim
    output_dim = test_case.output_dim
    loss = test_case.loss
    influence_type = test_case.influence_type

    train_data_loader = create_random_data_loader(input_dim, output_dim, data_len)
    test_data_loader = create_random_data_loader(input_dim, output_dim, test_data_len)

    nn_architecture = module_factory()
    nn_architecture = minimal_training(
        nn_architecture, train_data_loader, loss, lr=0.3, epochs=100
    )
    nn_architecture = nn_architecture.eval()

    model = TorchTwiceDifferentiable(nn_architecture, loss)

    direct_influence = compute_influences(
        model,
        training_data=train_data_loader,
        test_data=test_data_loader,
        progress=True,
        influence_type=influence_type,
        inversion_method=InversionMethod.Direct,
        hessian_regularization=hessian_reg,
    )

    num_parameters = sum(
        p.numel() for p in nn_architecture.parameters() if p.requires_grad
    )

    low_rank_influence = compute_influences(
        model,
        training_data=train_data_loader,
        test_data=test_data_loader,
        progress=True,
        influence_type=influence_type,
        inversion_method=InversionMethod.Arnoldi,
        hessian_regularization=hessian_reg,
        # as the hessian of the small shallow networks is in general not low rank, so for these test cases, we choose
        # the rank estimate as high as possible
        rank_estimate=num_parameters - 1,
    )

    assert np.allclose(direct_influence, low_rank_influence, rtol=1e-1)

    precomputed_low_rank = model_hessian_low_rank(
        model.model,
        model.loss,
        training_data=train_data_loader,
        hessian_perturbation=hessian_reg,
        rank_estimate=num_parameters - 1,
    )

    precomputed_low_rank_influence = compute_influences(
        model,
        training_data=train_data_loader,
        test_data=test_data_loader,
        progress=True,
        influence_type=influence_type,
        inversion_method=InversionMethod.Arnoldi,
        low_rank_representation=precomputed_low_rank,
    )

    assert np.allclose(direct_influence, precomputed_low_rank_influence, rtol=1e-1)
