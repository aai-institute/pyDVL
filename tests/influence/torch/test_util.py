from dataclasses import astuple, dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn
from numpy.typing import NDArray
from scipy.stats import pearsonr, spearmanr
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, TensorDataset

from pydvl.influence.torch.functional import (
    create_batch_hvp_function,
    create_hvp_function,
    lanzcos_low_rank_hessian_approx,
)
from pydvl.influence.torch.util import (
    BlockMode,
    ModelParameterDictBuilder,
    TorchLinalgEighException,
    TorchTensorContainerType,
    align_structure,
    flatten_dimensions,
    inverse_rank_one_update,
    inverse_rank_one_update_dict,
    rank_one_mvp,
    safe_torch_linalg_eigh,
    torch_dataset_to_dask_array,
)
from tests.conftest import is_osx_arm64
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
    model_params = {
        k: v.detach() for k, v in torch_model.named_parameters() if v.requires_grad
    }
    Hvp_autograd = create_batch_hvp_function(torch_model, torch.nn.functional.mse_loss)(
        model_params, x, y, vec
    )
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

    Hvp_autograd = create_hvp_function(
        torch_model, mse_loss, data_loader, use_average=use_avg
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


@pytest.mark.torch
@pytest.mark.parametrize("chunk_size", [5, 6])
@pytest.mark.parametrize("total_size", [50, 30, 45])
@pytest.mark.parametrize("tailing_dimensions", [(3,), (5, 8), (3, 7, 2)])
def test_torch_dataset_to_dask_array(
    chunk_size: int, total_size: int, tailing_dimensions: Tuple[int, ...]
):
    x_torch = torch.rand(*tuple([total_size, *tailing_dimensions]))
    y_torch = torch.rand(
        *tuple([total_size, *list(map(lambda x: x - 1, tailing_dimensions))])
    )

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, *tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return tuple([t[index] for t in self.tensors])

    data_set = CustomDataset(x_torch, y_torch)
    tensor_data_set = torch.utils.data.TensorDataset(x_torch)

    for d_set in [data_set, tensor_data_set]:
        x_da = torch_dataset_to_dask_array(d_set, chunk_size=chunk_size)
        assert x_da[0].shape == x_torch.shape
        assert sum(x_da[0].chunks[0]) == total_size
        assert all(
            [
                len(c) == 1 and c[0] == tailing_dimensions[k]
                for k, c in enumerate(x_da[0].chunks[1:])
            ]
        )
        assert np.allclose(x_torch.numpy(), x_da[0].compute())

    y_da = torch_dataset_to_dask_array(data_set, chunk_size=chunk_size)[1]
    assert np.allclose(y_torch.numpy(), y_da.compute())
    assert y_da.shape == y_torch.shape
    assert sum(y_da.chunks[0]) == total_size
    assert all(
        [
            len(c) == 1 and c[0] == tailing_dimensions[k] - 1
            for k, c in enumerate(y_da.chunks[1:])
        ]
    )

    with pytest.raises(ValueError):
        torch_dataset_to_dask_array(
            tensor_data_set, chunk_size=chunk_size, total_size=total_size + 1
        )


def check_influence_correlations(true_infl, approx_infl, threshold=0.95):
    for axis in range(0, true_infl.ndim):
        mean_true_infl = np.mean(true_infl, axis=axis)
        mean_approx_infl = np.mean(approx_infl, axis=axis)
        assert np.all(pearsonr(mean_true_infl, mean_approx_infl).statistic > threshold)
        assert np.all(spearmanr(mean_true_infl, mean_approx_infl).statistic > threshold)


def are_active_layers_linear(model):
    for module in model.modules():
        if len(list(module.children())) == 0 and len(list(module.parameters())) > 0:
            if not isinstance(module, torch.nn.Linear):
                param_requires_grad = [p.requires_grad for p in module.parameters()]
                if any(param_requires_grad):
                    return False
    return True


@pytest.mark.torch
def test_safe_torch_linalg_eigh():
    t = torch.randn([10, 10])
    t = t @ t.t()
    safe_eigs, safe_eigvec = safe_torch_linalg_eigh(t)
    eigs, eigvec = torch.linalg.eigh(t)
    assert torch.allclose(safe_eigs, eigs)
    assert torch.allclose(safe_eigvec, eigvec)


@pytest.mark.torch
@pytest.mark.slow
@pytest.mark.skipif(not is_osx_arm64(), reason="Requires macOS ARM64.")
def test_safe_torch_linalg_eigh_exception():
    with pytest.raises(TorchLinalgEighException):
        safe_torch_linalg_eigh(torch.randn([53000, 53000]))


@pytest.mark.torch
@pytest.mark.parametrize(
    "x_dim_0, x_dim_1, v_dim_0",
    [(10, 1, 12), (3, 2, 5), (4, 5, 30), (6, 6, 6), (1, 7, 7)],
)
def test_rank_one_mvp(x_dim_0, x_dim_1, v_dim_0):
    X = torch.randn(x_dim_0, x_dim_1)
    V = torch.randn(v_dim_0, x_dim_1)

    expected = (
        (torch.vmap(lambda x: x.unsqueeze(-1) * x.unsqueeze(-1).t())(X) @ V.t())
        .sum(dim=0)
        .t()
    )

    result = rank_one_mvp(X, V)

    assert result.shape == V.shape
    assert torch.allclose(result, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.torch
@pytest.mark.parametrize(
    "x_dim_0, x_dim_1, v_dim_0",
    [(10, 1, 12), (3, 2, 5), (4, 5, 10), (6, 6, 6), (1, 7, 7)],
)
@pytest.mark.parametrize("reg", [0.1, 100, 1.0, 10])
def test_inverse_rank_one_update(x_dim_0, x_dim_1, v_dim_0, reg):
    X = torch.randn(x_dim_0, x_dim_1)
    V = torch.randn(v_dim_0, x_dim_1)

    inverse_result = torch.zeros_like(V)

    for x in X:
        rank_one_matrix = x.unsqueeze(-1) * x.unsqueeze(-1).t()
        inverse_result += torch.linalg.solve(
            rank_one_matrix + reg * torch.eye(rank_one_matrix.shape[0]), V, left=False
        )

    inverse_result /= X.shape[0]
    result = inverse_rank_one_update(X, V, reg)

    assert torch.allclose(result, inverse_result, atol=1e-5)


@pytest.mark.torch
@pytest.mark.parametrize(
    "x_dim_1",
    [{"1": (4, 2, 3), "2": (5, 7), "3": ()}, {"1": (3, 6, 8, 9), "2": (1, 2)}, {"1": (1,)}],
)
@pytest.mark.parametrize(
    "x_dim_0, v_dim_0",
    [(10, 12), (3, 5), (4, 10), (6, 6), (1, 7)],
)
@pytest.mark.parametrize("reg", [0.5, 100, 1.0, 10])
def test_inverse_rank_one_update(x_dim_0, x_dim_1, v_dim_0, reg):
    X_dict = {k: torch.randn(x_dim_0, *d) for k, d in x_dim_1.items()}
    V_dict = {k: torch.randn(v_dim_0, *d) for k, d in x_dim_1.items()}

    X = flatten_dimensions(X_dict.values(), shape=(x_dim_0, -1))
    V = flatten_dimensions(V_dict.values(), shape=(v_dim_0, -1))
    result = inverse_rank_one_update(X, V, reg)

    inverse_result = flatten_dimensions(
        inverse_rank_one_update_dict(X_dict, V_dict, reg).values(), shape=(v_dim_0, -1)
    )

    assert torch.allclose(result, inverse_result, atol=1e-5, rtol=1e-3)


class TestModelParameterDictBuilder:
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(5, 10)
            self.fc2 = torch.nn.Linear(10, 5)
            self.fc1.weight.requires_grad = False

    @pytest.fixture
    def model(self):
        return TestModelParameterDictBuilder.SimpleModel()

    @pytest.mark.parametrize("block_mode", [mode for mode in BlockMode])
    def test_build(self, block_mode, model):
        builder = ModelParameterDictBuilder(
            model=model,
            detach=True,
        )
        param_dict = builder.build_from_block_mode(block_mode)

        if block_mode is BlockMode.FULL:
            assert "" in param_dict
            assert "fc1.weight" not in param_dict[""]
        elif block_mode is BlockMode.PARAMETER_WISE:
            assert "fc2.bias" in param_dict
            assert len(param_dict["fc2.bias"]) > 0
            assert "fc1.weight" not in param_dict
        elif block_mode is BlockMode.LAYER_WISE:
            assert "fc2" in param_dict
            assert "fc2.bias" in param_dict["fc2"]
            assert "fc1.weight" not in param_dict["fc1"]
            assert "fc1.bias" in param_dict["fc1"]

        assert all(
            (not p.requires_grad for q in param_dict.values() for p in q.values())
        )
