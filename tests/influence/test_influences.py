import itertools
from typing import List, Tuple

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader

from pydvl.influence import TorchTwiceDifferentiable, compute_influences
from pydvl.influence.general import InfluenceType, InversionMethod

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
def test_influence_linear_model(
    influence_type: InfluenceType,
    train_set_size: int,
    hessian_reg: float = 0.1,
    test_set_size: int = 20,
    problem_dimension: Tuple[int, int] = (3, 15),
    condition_number: float = 3,
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

    direct_influences = compute_influences(
        TorchTwiceDifferentiable(linear_layer, loss),
        training_data=train_data_loader,
        test_data=test_data_loader,
        input_data=input_data,
        progress=True,
        influence_type=influence_type,
        inversion_method="direct",
        hessian_regularization=hessian_reg,
    ).numpy()

    cg_influences = compute_influences(
        TorchTwiceDifferentiable(linear_layer, loss),
        training_data=train_data_loader,
        test_data=test_data_loader,
        input_data=input_data,
        progress=True,
        influence_type=influence_type,
        inversion_method="cg",
        hessian_regularization=hessian_reg,
    ).numpy()

    lissa_influences = compute_influences(
        TorchTwiceDifferentiable(linear_layer, loss),
        training_data=train_data_loader,
        test_data=test_data_loader,
        input_data=input_data,
        progress=True,
        influence_type=influence_type,
        inversion_method="lissa",
        maxiter=5000,
        scale=100,
        hessian_regularization=hessian_reg,
    ).numpy()
    assert np.logical_not(np.any(np.isnan(direct_influences)))
    assert np.logical_not(np.any(np.isnan(cg_influences)))
    assert np.allclose(direct_influences, analytical_influences, rtol=1e-7)
    assert np.allclose(cg_influences, analytical_influences, rtol=1e-1)
    abs_influence = np.abs(lissa_influences)
    upper_quantile_mask = abs_influence > np.quantile(abs_influence, 0.9)
    assert np.allclose(
        lissa_influences[upper_quantile_mask],
        analytical_influences[upper_quantile_mask],
        rtol=0.1,
    )


conv3d_nn = nn.Sequential(
    nn.Conv3d(in_channels=5, out_channels=3, kernel_size=2),
    nn.Flatten(),
    nn.Linear(24, 3),
)
conv2d_nn = nn.Sequential(
    nn.Conv2d(in_channels=5, out_channels=3, kernel_size=3),
    nn.Flatten(),
    nn.Linear(27, 3),
)
conv1d_nn = nn.Sequential(
    nn.Conv1d(in_channels=5, out_channels=3, kernel_size=2),
    nn.Flatten(),
    nn.Linear(6, 3),
)
simple_nn_regr = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 3), nn.Linear(3, 1))

test_cases = {
    "conv3d_nn_up": [
        conv3d_nn,
        (5, 3, 3, 3),
        3,
        nn.MSELoss(),
        InfluenceType.Up,
    ],
    "conv3d_nn_pert": [
        conv3d_nn,
        (5, 3, 3, 3),
        3,
        nn.SmoothL1Loss(),
        InfluenceType.Perturbation,
    ],
    "conv_2d_nn_up": [conv2d_nn, (5, 5, 5), 3, nn.MSELoss(), InfluenceType.Up],
    "conv_2d_nn_pert": [
        conv2d_nn,
        (5, 5, 5),
        3,
        nn.SmoothL1Loss(),
        InfluenceType.Perturbation,
    ],
    "conv_1d_nn_up": [conv1d_nn, (5, 3), 3, nn.MSELoss(), InfluenceType.Up],
    "conv_1d_pert": [
        conv1d_nn,
        (5, 3),
        3,
        nn.SmoothL1Loss(),
        InfluenceType.Perturbation,
    ],
    "simple_nn_up": [simple_nn_regr, (10,), 1, nn.MSELoss(), InfluenceType.Up],
    "simple_nn_pert": [
        simple_nn_regr,
        (10,),
        1,
        nn.SmoothL1Loss(),
        InfluenceType.Perturbation,
    ],
}


@pytest.mark.torch
@pytest.mark.parametrize(
    "nn_architecture, input_dim, output_dim, loss, influence_type",
    test_cases.values(),
    ids=test_cases.keys(),
)
def test_influences_nn(
    nn_architecture: nn.Module,
    input_dim: Tuple[int],
    output_dim: int,
    loss: nn.modules.loss._Loss,
    influence_type: InfluenceType,
    data_len: int = 20,
    hessian_reg: float = 1e3,
    test_data_len: int = 10,
    batch_size: int = 10,
):
    x_train = torch.rand((data_len, *input_dim))
    y_train = torch.rand((data_len, output_dim))
    x_test = torch.rand((test_data_len, *input_dim))
    y_test = torch.rand((test_data_len, output_dim))
    nn_architecture.eval()

    inversion_method_kwargs = {
        "direct": {},
        "cg": {},
        "lissa": {
            "maxiter": 100,
            "scale": 10000,
        },
    }
    train_data_loader = DataLoader(list(zip(x_train, y_train)), batch_size=batch_size)
    test_data_loader = DataLoader(
        list(zip(x_test, y_test)),
        batch_size=batch_size,
    )
    multiple_influences = {}
    for inversion_method in InversionMethod:
        influences = compute_influences(
            TorchTwiceDifferentiable(nn_architecture, loss),
            training_data=train_data_loader,
            test_data=test_data_loader,
            progress=True,
            influence_type=influence_type,
            inversion_method=inversion_method,
            hessian_regularization=hessian_reg,
            **inversion_method_kwargs[inversion_method],
        ).numpy()
        assert not np.any(np.isnan(influences))
        multiple_influences[inversion_method] = influences

    for infl_type, influences in multiple_influences.items():
        if infl_type == "direct":
            continue
        assert np.allclose(
            influences,
            multiple_influences["direct"],
            rtol=1e-1,
        ), f"Failed method {infl_type}"
        if influence_type == InfluenceType.Up:
            assert influences.shape == (test_data_len, data_len)
        elif influence_type == InfluenceType.Perturbation:
            assert influences.shape == (test_data_len, data_len, *input_dim)
    # check that influences are not all constant
    assert not np.all(influences == influences.item(0))
