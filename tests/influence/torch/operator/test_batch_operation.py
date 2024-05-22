from dataclasses import astuple

import pytest
import torch

from pydvl.influence.torch.base import TorchBatch
from pydvl.influence.torch.batch_operation import (
    GaussNewtonBatchOperation,
    HessianBatchOperation,
)

from ..test_util import model_data, test_parameters


@pytest.mark.torch
@pytest.mark.parametrize(
    "model_data, tol",
    [(astuple(tp.model_params), 1e-5) for tp in test_parameters],
    indirect=["model_data"],
)
def test_hessian_batch_operation(model_data, tol: float):
    torch_model, x, y, vec, h_analytical = model_data

    params = dict(torch_model.named_parameters())

    hessian_op = HessianBatchOperation(
        torch_model, torch.nn.functional.mse_loss, restrict_to=params
    )
    hvp_autograd = hessian_op.apply_to_vec(TorchBatch(x, y), vec)

    assert torch.allclose(hvp_autograd, h_analytical @ vec, rtol=tol)


@pytest.mark.torch
@pytest.mark.parametrize(
    "model_data, tol",
    [(astuple(tp.model_params), 1e-3) for tp in test_parameters],
    indirect=["model_data"],
)
def test_gauss_newton_batch_operation(model_data, tol: float):
    torch_model, x, y, vec, _ = model_data

    y_pred = torch_model(x)
    out_features = y_pred.shape[1]
    dl_dw = torch.vmap(
        lambda r, s, t: 2 / float(out_features) * (s - t).view(-1, 1) @ r.view(1, -1)
    )(x, y_pred, y)
    dl_db = torch.vmap(lambda s, t: 2 / float(out_features) * (s - t))(y_pred, y)
    grad_analytical = torch.cat([dl_dw.reshape(x.shape[0], -1), dl_db], dim=-1)
    gn_mat_analytical = torch.sum(
        torch.func.vmap(lambda t: t.unsqueeze(-1) * t.unsqueeze(-1).t())(
            grad_analytical
        ),
        dim=0,
    )

    params = dict(torch_model.named_parameters())

    gn_op = GaussNewtonBatchOperation(
        torch_model, torch.nn.functional.mse_loss, restrict_to=params
    )
    gn_autograd = gn_op.apply_to_vec(TorchBatch(x, y), vec)

    gn_analytical = gn_mat_analytical @ vec

    assert torch.allclose(gn_autograd, gn_analytical, atol=1e-5, rtol=tol)
