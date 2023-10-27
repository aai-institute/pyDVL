from dataclasses import astuple, dataclass
from typing import Any, Dict, Tuple

import pytest

torch = pytest.importorskip("torch")
import torch.nn
from numpy.typing import NDArray
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, TensorDataset

from pydvl.influence.torch.functional import (
    batch_loss_function,
    get_batch_hvp,
    get_hvp_function,
    hvp,
)
from pydvl.influence.torch.torch_differentiable import lanzcos_low_rank_hessian_approx
from pydvl.influence.torch.util import (
    TorchTensorContainerType,
    align_structure,
    flatten_dimensions,
)
from tests.influence.conftest import linear_hessian_analytical, linear_model


@dataclass
class ModelParams:
    dimension: Tuple[int, int]
    condition_number: float
    train_size: int


@dataclass
class UtilTestParameters:
    """
    Helper class to add more test parameter combinations
    """

    model_params: ModelParams
    batch_size: int
    rank_estimate: int
    regularization: float


test_parameters = [
    UtilTestParameters(
        ModelParams(dimension=(30, 16), condition_number=4, train_size=60),
        batch_size=4,
        rank_estimate=200,
        regularization=0.0001,
    ),
    UtilTestParameters(
        ModelParams(dimension=(32, 35), condition_number=1e6, train_size=100),
        batch_size=5,
        rank_estimate=70,
        regularization=0.001,
    ),
    UtilTestParameters(
        ModelParams(dimension=(25, 15), condition_number=1e3, train_size=90),
        batch_size=10,
        rank_estimate=50,
        regularization=0.0001,
    ),
    UtilTestParameters(
        ModelParams(dimension=(30, 15), condition_number=1e4, train_size=120),
        batch_size=8,
        rank_estimate=160,
        regularization=0.00001,
    ),
    UtilTestParameters(
        ModelParams(dimension=(40, 13), condition_number=1e5, train_size=900),
        batch_size=4,
        rank_estimate=250,
        regularization=0.00001,
    ),
]


def linear_torch_model_from_numpy(A: NDArray, b: NDArray) -> torch.nn.Module:
    """
    Given numpy arrays representing the model $xA^t + b$, the function returns the corresponding torch model
    :param A:
    :param b:
    :return:
    """
    output_dimension, input_dimension = tuple(A.shape)
    model = torch.nn.Linear(input_dimension, output_dimension)
    model.eval()
    model.weight.data = torch.as_tensor(A, dtype=torch.get_default_dtype())
    model.bias.data = torch.as_tensor(b, dtype=torch.get_default_dtype())
    return model


@pytest.fixture
def model_data(request):
    dimension, condition_number, train_size = request.param
    A, b = linear_model(dimension, condition_number)
    x = torch.rand(train_size, dimension[-1])
    y = torch.rand(train_size, dimension[0])
    torch_model = linear_torch_model_from_numpy(A, b)
    vec = flatten_dimensions(
        tuple(
            torch.rand(*p.shape)
            for name, p in torch_model.named_parameters()
            if p.requires_grad
        )
    )
    H_analytical = linear_hessian_analytical((A, b), x.numpy())
    H_analytical = torch.as_tensor(H_analytical)
    return torch_model, x, y, vec, H_analytical.to(torch.float32)


@pytest.mark.torch
@pytest.mark.parametrize(
    "model_data, tol",
    [(astuple(tp.model_params), 1e-5) for tp in test_parameters],
    indirect=["model_data"],
)
def test_batch_hvp(model_data, tol: float):
    torch_model, x, y, vec, H_analytical = model_data
    Hvp_autograd = get_batch_hvp(torch_model, torch.nn.functional.mse_loss)(x, y, vec)
    assert torch.allclose(Hvp_autograd, H_analytical @ vec, rtol=tol)


@pytest.mark.torch
@pytest.mark.parametrize(
    "use_avg, tol", [(True, 1e-5), (False, 1e-5)], ids=["avg", "full"]
)
@pytest.mark.parametrize(
    "model_data, batch_size",
    [(astuple(tp.model_params), tp.batch_size) for tp in test_parameters],
    indirect=["model_data"],
)
def test_get_hvp_function(model_data, tol: float, use_avg: bool, batch_size: int):
    torch_model, x, y, vec, H_analytical = model_data
    data_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)

    Hvp_autograd = get_hvp_function(
        torch_model, mse_loss, data_loader, use_hessian_avg=use_avg
    )(vec)

    assert torch.allclose(Hvp_autograd, H_analytical @ vec, rtol=tol)


@pytest.mark.torch
@pytest.mark.parametrize(
    "model_data, batch_size, rank_estimate, regularization",
    [astuple(tp) for tp in test_parameters],
    indirect=["model_data"],
)
def test_lanzcos_low_rank_hessian_approx(
    model_data, batch_size: int, rank_estimate, regularization
):
    _, _, _, vec, H_analytical = model_data

    reg_H_analytical = H_analytical + regularization * torch.eye(H_analytical.shape[0])
    low_rank_approx = lanzcos_low_rank_hessian_approx(
        lambda z: reg_H_analytical @ z,
        reg_H_analytical.shape,
        rank_estimate=rank_estimate,
    )
    approx_result = low_rank_approx.projections @ (
        torch.diag_embed(low_rank_approx.eigen_vals)
        @ (low_rank_approx.projections.t() @ vec.t())
    )
    assert torch.allclose(approx_result, reg_H_analytical @ vec, rtol=1e-1)


@pytest.mark.torch
def test_lanzcos_low_rank_hessian_approx_exception():
    """
    In case cuda is not available, and cupy is not installed, the call should raise an import exception
    """
    if not torch.cuda.is_available():
        with pytest.raises(ImportError):
            lanzcos_low_rank_hessian_approx(
                lambda x: x, (3, 3), eigen_computation_on_gpu=True
            )


@pytest.mark.parametrize(
    "source,target",
    [
        (
            {"a": torch.randn(5, 5), "b": torch.randn(5, 5)},
            {"a": torch.randn(5, 5), "b": torch.randn(5, 5)},
        ),
        (
            {"a": torch.randn(5, 5), "b": torch.randn(5, 5)},
            (torch.randn(5, 5), torch.randn(5, 5)),
        ),
        ({"a": torch.randn(5, 5), "b": torch.randn(5, 5)}, torch.randn(50)),
    ],
)
def test_align_structure_success(
    source: Dict[str, torch.Tensor], target: TorchTensorContainerType
):
    result = align_structure(source, target)
    assert isinstance(result, dict)
    assert list(result.keys()) == list(source.keys())
    assert all([result[k].shape == source[k].shape for k in source.keys()])


@pytest.mark.parametrize(
    "source,target",
    [
        (
            {"a": torch.randn(5, 5), "b": torch.randn(5, 5)},
            {"a": torch.randn(5, 5), "b": torch.randn(3, 3)},
        ),
        (
            {"a": torch.randn(5, 5), "b": torch.randn(5, 5)},
            {"c": torch.randn(5, 5), "d": torch.randn(5, 5)},
        ),
        (
            {"a": torch.randn(5, 5), "b": torch.randn(5, 5)},
            "unsupported",
        ),
    ],
)
def test_align_structure_error(source: Dict[str, torch.Tensor], target: Any):
    with pytest.raises(ValueError):
        align_structure(source, target)
