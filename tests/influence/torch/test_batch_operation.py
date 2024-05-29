from dataclasses import astuple

import pytest
import torch

from pydvl.influence.torch.base import TorchBatch
from pydvl.influence.torch.batch_operation import (
    GaussNewtonBatchOperation,
    HessianBatchOperation,
    InverseHarmonicMeanBatchOperation,
)
from pydvl.influence.torch.util import align_structure, flatten_dimensions

from .test_util import model_data, test_parameters, torch


@pytest.mark.torch
@pytest.mark.parametrize(
    "model_data, tol",
    [(astuple(tp.model_params), 1e-5) for tp in test_parameters],
    indirect=["model_data"],
)
def test_hessian_batch_operation(model_data, tol: float, pytorch_seed):
    torch_model, x, y, vec, h_analytical = model_data

    params = {k: p.detach() for k, p in torch_model.named_parameters()}

    hessian_op = HessianBatchOperation(
        torch_model, torch.nn.functional.mse_loss, restrict_to=params
    )
    batch_size = 10
    rand_mat_dict = {k: torch.randn(batch_size, *t.shape) for k, t in params.items()}
    flat_rand_mat = flatten_dimensions(rand_mat_dict.values(), shape=(batch_size, -1))
    hvp_autograd_mat_dict = hessian_op.apply_to_dict(TorchBatch(x, y), rand_mat_dict)

    hvp_autograd = hessian_op.apply(TorchBatch(x, y), vec)
    hvp_autograd_dict = hessian_op.apply_to_dict(
        TorchBatch(x, y), align_structure(params, vec)
    )
    hvp_autograd_dict_flat = flatten_dimensions(hvp_autograd_dict.values())

    assert torch.allclose(hvp_autograd, h_analytical @ vec, rtol=tol)
    assert torch.allclose(hvp_autograd_dict_flat, h_analytical @ vec, rtol=tol)

    op_then_flat = flatten_dimensions(
        hvp_autograd_mat_dict.values(), shape=(batch_size, -1)
    )
    flat_then_op_analytical = torch.einsum("ik, jk -> ji", h_analytical, flat_rand_mat)

    assert torch.allclose(
        op_then_flat,
        flat_then_op_analytical,
        atol=1e-5,
        rtol=tol,
    )
    assert torch.allclose(
        hessian_op._apply_to_mat(TorchBatch(x, y), flat_rand_mat), op_then_flat
    )


@pytest.mark.torch
@pytest.mark.parametrize(
    "model_data, tol",
    [(astuple(tp.model_params), 1e-3) for tp in test_parameters],
    indirect=["model_data"],
)
def test_gauss_newton_batch_operation(model_data, tol: float, pytorch_seed):
    torch_model, x, y, vec, _ = model_data

    y_pred = torch_model(x)
    out_features = y_pred.shape[1]
    dl_dw = torch.vmap(
        lambda r, s, t: 2 / float(out_features) * (s - t).view(-1, 1) @ r.view(1, -1)
    )(x, y_pred, y)
    dl_db = torch.vmap(lambda s, t: 2 / float(out_features) * (s - t))(y_pred, y)
    grad_analytical = torch.cat([dl_dw.reshape(x.shape[0], -1), dl_db], dim=-1)
    gn_mat_analytical = (
        torch.sum(
            torch.func.vmap(lambda t: t.unsqueeze(-1) * t.unsqueeze(-1).t())(
                grad_analytical
            ),
            dim=0,
        )
        / x.shape[0]
    )

    params = dict(torch_model.named_parameters())

    gn_op = GaussNewtonBatchOperation(
        torch_model, torch.nn.functional.mse_loss, restrict_to=params
    )
    batch_size = 10

    gn_autograd = gn_op.apply(TorchBatch(x, y), vec)
    gn_autograd_dict = gn_op.apply_to_dict(
        TorchBatch(x, y), align_structure(params, vec)
    )
    gn_autograd_dict_flat = flatten_dimensions(gn_autograd_dict.values())
    analytical_vec = gn_mat_analytical @ vec
    assert torch.allclose(gn_autograd, analytical_vec, atol=1e-5, rtol=tol)
    assert torch.allclose(gn_autograd_dict_flat, analytical_vec, atol=1e-5, rtol=tol)

    rand_mat_dict = {k: torch.randn(batch_size, *t.shape) for k, t in params.items()}
    flat_rand_mat = flatten_dimensions(rand_mat_dict.values(), shape=(batch_size, -1))
    gn_autograd_mat_dict = gn_op.apply_to_dict(TorchBatch(x, y), rand_mat_dict)

    op_then_flat = flatten_dimensions(
        gn_autograd_mat_dict.values(), shape=(batch_size, -1)
    )
    flat_then_op = gn_op._apply_to_mat(TorchBatch(x, y), flat_rand_mat)

    assert torch.allclose(
        op_then_flat,
        flat_then_op,
        atol=1e-5,
        rtol=tol,
    )

    flat_then_op_analytical = torch.einsum(
        "ik, jk -> ji", gn_mat_analytical, flat_rand_mat
    )

    assert torch.allclose(
        op_then_flat,
        flat_then_op_analytical,
        atol=1e-5,
        rtol=1e-2,
    )


@pytest.mark.torch
@pytest.mark.parametrize(
    "model_data, tol",
    [(astuple(tp.model_params), 1e-3) for tp in test_parameters],
    indirect=["model_data"],
)
@pytest.mark.parametrize("reg", [0.4])
def test_inverse_harmonic_mean_batch_operation(
    model_data, tol: float, reg, pytorch_seed
):
    torch_model, x, y, vec, _ = model_data
    y_pred = torch_model(x)
    out_features = y_pred.shape[1]
    dl_dw = torch.vmap(
        lambda r, s, t: 2 / float(out_features) * (s - t).view(-1, 1) @ r.view(1, -1)
    )(x, y_pred, y)
    dl_db = torch.vmap(lambda s, t: 2 / float(out_features) * (s - t))(y_pred, y)
    grad_analytical = torch.cat([dl_dw.reshape(x.shape[0], -1), dl_db], dim=-1)
    params = {
        k: p.detach() for k, p in torch_model.named_parameters() if p.requires_grad
    }

    ihm_mat_analytical = torch.sum(
        torch.func.vmap(
            lambda z: torch.linalg.inv(
                z.unsqueeze(-1) * z.unsqueeze(-1).t() + reg * torch.eye(len(z))
            )
        )(grad_analytical),
        dim=0,
    )
    ihm_mat_analytical /= x.shape[0]

    gn_op = InverseHarmonicMeanBatchOperation(
        torch_model, torch.nn.functional.mse_loss, reg, restrict_to=params
    )
    batch_size = 10

    gn_autograd = gn_op.apply(TorchBatch(x, y), vec)
    gn_autograd_dict = gn_op.apply_to_dict(
        TorchBatch(x, y), align_structure(params, vec)
    )
    gn_autograd_dict_flat = flatten_dimensions(gn_autograd_dict.values())
    analytical = ihm_mat_analytical @ vec

    assert torch.allclose(gn_autograd, analytical, atol=1e-5, rtol=tol)
    assert torch.allclose(gn_autograd_dict_flat, analytical, atol=1e-5, rtol=tol)

    rand_mat_dict = {k: torch.randn(batch_size, *t.shape) for k, t in params.items()}
    flat_rand_mat = flatten_dimensions(rand_mat_dict.values(), shape=(batch_size, -1))
    gn_autograd_mat_dict = gn_op.apply_to_dict(TorchBatch(x, y), rand_mat_dict)

    op_then_flat = flatten_dimensions(
        gn_autograd_mat_dict.values(), shape=(batch_size, -1)
    )
    flat_then_op = gn_op._apply_to_mat(TorchBatch(x, y), flat_rand_mat)

    assert torch.allclose(
        op_then_flat,
        flat_then_op,
        atol=1e-5,
        rtol=tol,
    )

    flat_then_op_analytical = torch.einsum(
        "ik, jk -> ji", ihm_mat_analytical, flat_rand_mat
    )

    assert torch.allclose(
        op_then_flat,
        flat_then_op_analytical,
        atol=1e-5,
        rtol=1e-2,
    )


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
    ) / x_dim_0

    result = GaussNewtonBatchOperation._rank_one_mvp(X, V)

    assert result.shape == V.shape
    assert torch.allclose(result, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.torch
@pytest.mark.parametrize(
    "x_dim_1",
    [
        [(4, 2, 3), (5, 7), (5,)],
        [(3, 6, 8, 9), (1, 2)],
        [(1,)],
    ],
)
@pytest.mark.parametrize(
    "x_dim_0, v_dim_0",
    [(10, 12), (3, 5), (4, 10), (6, 6), (1, 7)],
)
def test_generate_rank_one_mvp(x_dim_0, x_dim_1, v_dim_0):
    x_list = [torch.randn(x_dim_0, *d) for d in x_dim_1]
    v_list = [torch.randn(v_dim_0, *d) for d in x_dim_1]

    x = flatten_dimensions(x_list, shape=(x_dim_0, -1))
    v = flatten_dimensions(v_list, shape=(v_dim_0, -1))
    result = GaussNewtonBatchOperation._rank_one_mvp(x, v)

    inverse_result = flatten_dimensions(
        GaussNewtonBatchOperation._generate_rank_one_mvp(x_list, v_list),
        shape=(v_dim_0, -1),
    )

    assert torch.allclose(result, inverse_result, atol=1e-5, rtol=1e-3)


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
    result = InverseHarmonicMeanBatchOperation._inverse_rank_one_update(X, V, reg)

    assert torch.allclose(result, inverse_result, atol=1e-5)


@pytest.mark.torch
@pytest.mark.parametrize(
    "x_dim_1",
    [
        [(4, 2, 3), (5, 7), (5,)],
        [(3, 6, 8, 9), (1, 2)],
        [(1,)],
    ],
)
@pytest.mark.parametrize(
    "x_dim_0, v_dim_0",
    [(10, 12), (3, 5), (4, 10), (6, 6), (1, 7)],
)
@pytest.mark.parametrize("reg", [0.5, 100, 1.0, 10])
def test_generate_inverse_rank_one_updates(
    x_dim_0, x_dim_1, v_dim_0, reg, pytorch_seed
):
    x_list = [torch.randn(x_dim_0, *d) for d in x_dim_1]
    v_list = [torch.randn(v_dim_0, *d) for d in x_dim_1]

    x = flatten_dimensions(x_list, shape=(x_dim_0, -1))
    v = flatten_dimensions(v_list, shape=(v_dim_0, -1))
    result = InverseHarmonicMeanBatchOperation._inverse_rank_one_update(x, v, reg)

    inverse_result = flatten_dimensions(
        InverseHarmonicMeanBatchOperation._generate_inverse_rank_one_updates(
            x_list, v_list, reg
        ),
        shape=(v_dim_0, -1),
    )

    assert torch.allclose(result, inverse_result, atol=1e-5, rtol=1e-3)
