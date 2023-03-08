import itertools
from typing import List, Tuple

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn

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
    :param hessian_retularization: regularization value for the hessian
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


INFLUENCE_TEST_CONDITION_NUMBERS: List[int] = [3]
INFLUENCE_TRAINING_SET_SIZE: List[int] = [50, 30]
INFLUENCE_TEST_SET_SIZE: List[int] = [20]
INFLUENCE_DIMENSIONS: List[Tuple[int, int]] = [
    (10, 10),
    (3, 20),
]
HESSIAN_REGULARIZATION: List[float] = [0, 1]


test_cases = list(
    itertools.product(
        INFLUENCE_TRAINING_SET_SIZE,
        INFLUENCE_TEST_SET_SIZE,
        InfluenceType,
        INFLUENCE_DIMENSIONS,
        INFLUENCE_TEST_CONDITION_NUMBERS,
        HESSIAN_REGULARIZATION,
    )
)


def lmb_test_case_to_str(packed_i_test_case):
    i, test_case = packed_i_test_case
    return (
        f"Problem #{i} of dimension {test_case[3]} with train size {test_case[0]}, "
        f"test size {test_case[1]}, if_type {test_case[2]}, condition number {test_case[4]} and lam {test_case[5]}."
    )


test_case_ids = list(map(lmb_test_case_to_str, zip(range(len(test_cases)), test_cases)))


@pytest.mark.torch
@pytest.mark.parametrize(
    "train_set_size,test_set_size,influence_type,problem_dimension,condition_number, hessian_reg",
    test_cases,
    ids=test_case_ids,
)
def test_influence_linear_model(
    train_set_size: int,
    test_set_size: int,
    influence_type: InfluenceType,
    problem_dimension: Tuple[int, int],
    condition_number: float,
    hessian_reg: float,
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

    direct_influences = compute_influences(
        TorchTwiceDifferentiable(linear_layer, loss),
        *train_data,
        *test_data,
        progress=True,
        influence_type=influence_type,
        inversion_method="direct",
        hessian_regularization=hessian_reg,
    ).numpy()

    cg_influences = compute_influences(
        TorchTwiceDifferentiable(linear_layer, loss),
        *train_data,
        *test_data,
        progress=True,
        influence_type=influence_type,
        inversion_method="cg",
        hessian_regularization=hessian_reg,
    ).numpy()
    assert np.logical_not(np.any(np.isnan(direct_influences)))
    assert np.logical_not(np.any(np.isnan(cg_influences)))
    assert np.allclose(direct_influences, analytical_influences, rtol=1e-7)
    assert np.allclose(cg_influences, analytical_influences, rtol=1e-1)


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
        10,
        (5, 3, 3, 3),
        3,
        nn.MSELoss(),
        InfluenceType.Up,
    ],
    "conv3d_nn_pert": [
        conv3d_nn,
        10,
        (5, 3, 3, 3),
        3,
        nn.SmoothL1Loss(),
        InfluenceType.Perturbation,
    ],
    "conv_2d_nn_up": [conv2d_nn, 10, (5, 5, 5), 3, nn.MSELoss(), InfluenceType.Up],
    "conv_2d_nn_pert": [
        conv2d_nn,
        10,
        (5, 5, 5),
        3,
        nn.MSELoss(),
        InfluenceType.Perturbation,
    ],
    "conv_1d_nn_up": [conv1d_nn, 10, (5, 3), 3, nn.MSELoss(), InfluenceType.Up],
    "conv_1d_pert": [
        conv1d_nn,
        10,
        (5, 3),
        3,
        nn.SmoothL1Loss(),
        InfluenceType.Perturbation,
    ],
    "simple_nn_up": [simple_nn_regr, 10, (10,), 1, nn.MSELoss(), InfluenceType.Up],
    "simple_nn_pert": [
        simple_nn_regr,
        10,
        (10,),
        1,
        nn.MSELoss(),
        InfluenceType.Perturbation,
    ],
}


@pytest.mark.torch
@pytest.mark.parametrize(
    "nn_architecture, batch_size, input_dim, output_dim, loss, influence_type",
    test_cases.values(),
    ids=test_cases.keys(),
)
def test_influences_nn(
    nn_architecture: nn.Module,
    batch_size: int,
    input_dim: Tuple[int],
    output_dim: int,
    loss: nn.modules.loss._Loss,
    influence_type: InfluenceType,
    hessian_reg: float = 100,
    test_data_len: int = 10,
):
    x_train = torch.rand((batch_size, *input_dim))
    y_train = torch.rand((batch_size, output_dim))
    x_test = torch.rand((test_data_len, *input_dim))
    y_test = torch.rand((test_data_len, output_dim))
    nn_architecture.eval()

    multiple_influences = []
    for inversion_method in InversionMethod:
        influences = compute_influences(
            TorchTwiceDifferentiable(nn_architecture, loss),
            x_train,
            y_train,
            x_test,
            y_test,
            progress=True,
            influence_type=influence_type,
            inversion_method=inversion_method,
            hessian_regularization=hessian_reg,
        ).numpy()
        assert not np.any(np.isnan(influences))
        multiple_influences.append(influences)
    if influence_type == InfluenceType.Up:
        assert np.allclose(*multiple_influences, rtol=1e-3)
        assert influences.shape == (test_data_len, batch_size)
    elif influence_type == InfluenceType.Perturbation:
        assert np.allclose(*multiple_influences, rtol=1e-1)
        assert influences.shape == (test_data_len, batch_size, *input_dim)
    else:
        raise ValueError(f"Unknown influence type: {influence_type}")
    # check that influences are not all equal
    assert not np.all(influences == influences.item(0))
