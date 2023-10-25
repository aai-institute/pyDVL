from dataclasses import astuple

import pytest

from tests.influence.conftest import (
    linear_mixed_second_derivative_analytical,
    linear_model,
)
from tests.influence.test_torch_differentiable import (
    DATA_OUTPUT_NOISE,
    linear_mvp_model,
)

torch = pytest.importorskip("torch")
import numpy as np
import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, TensorDataset

from pydvl.influence.torch.functional import (
    batch_loss_function,
    get_hessian,
    get_hvp_function,
    hvp,
    matrix_jacobian_product,
    per_sample_gradient,
    per_sample_mixed_derivative,
)
from pydvl.influence.torch.util import align_structure, flatten_dimensions
from tests.influence.test_util import model_data, test_parameters


@pytest.mark.torch
@pytest.mark.parametrize(
    "model_data, tol",
    [(astuple(tp.model_params), 1e-5) for tp in test_parameters],
    indirect=["model_data"],
)
def test_hvp(model_data, tol: float):
    torch_model, x, y, vec, H_analytical = model_data

    params = dict(torch_model.named_parameters())

    f = batch_loss_function(torch_model, torch.nn.functional.mse_loss)

    Hvp_autograd = hvp(lambda p: f(p, x, y), params, align_structure(params, vec))

    flat_Hvp_autograd = flatten_dimensions(Hvp_autograd.values())
    assert torch.allclose(flat_Hvp_autograd, H_analytical @ vec, rtol=tol)


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
        torch_model,
        mse_loss,
        data_loader,
        use_hessian_avg=use_avg,
    )(vec)

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
def test_get_hessian(
    model_data,
    tol: float,
    use_avg: bool,
    batch_size: int,
):
    torch_model, x, y, _, H_analytical = model_data
    data_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)

    Hvp_autograd = get_hessian(
        torch_model,
        mse_loss,
        data_loader,
        use_hessian_avg=use_avg,
    )

    assert torch.allclose(Hvp_autograd, H_analytical, rtol=tol)


@pytest.mark.torch
@pytest.mark.parametrize(
    "in_features, out_features, batch_size",
    [(46, 6, 632), (50, 3, 120), (100, 5, 120), (25, 10, 550)],
)
def test_per_sample_gradient(in_features, out_features, batch_size):
    model = torch.nn.Linear(in_features, out_features)
    loss = torch.nn.functional.mse_loss

    x = torch.randn(batch_size, in_features, requires_grad=True)
    y = torch.randn(batch_size, out_features)
    params = {k: p.detach() for k, p in model.named_parameters() if p.requires_grad}
    gradients = per_sample_gradient(model, loss)(params, x, y)

    # Compute analytical gradients
    y_pred = model(x)
    dL_dw = torch.vmap(
        lambda r, s, t: 2 / float(out_features) * (s - t).view(-1, 1) @ r.view(1, -1)
    )(x, y_pred, y)
    dL_db = torch.vmap(lambda s, t: 2 / float(out_features) * (s - t))(y_pred, y)

    # Assert the gradient values for equality with analytical gradients
    assert torch.allclose(gradients["weight"], dL_dw, atol=1e-5)
    assert torch.allclose(gradients["bias"], dL_db, atol=1e-5)


@pytest.mark.torch
@pytest.mark.parametrize(
    "in_features, out_features, batch_size",
    [(46, 1, 632), (50, 3, 120), (100, 5, 110), (25, 10, 500)],
)
def test_matrix_jacobian_product(in_features, out_features, batch_size):
    model = torch.nn.Linear(in_features, out_features)
    params = {k: p for k, p in model.named_parameters() if p.requires_grad}

    x = torch.randn(batch_size, in_features, requires_grad=True)
    y = torch.randn(batch_size, out_features, requires_grad=True)
    y_pred = model(x)

    G = torch.randn((10, out_features * (in_features + 1)))
    mjp = matrix_jacobian_product(model, torch.nn.functional.mse_loss, G)(params, x, y)

    dL_dw = torch.vmap(
        lambda r, s, t: 2 / float(out_features) * (s - t).view(-1, 1) @ r.view(1, -1)
    )(x, y_pred, y)
    dL_db = torch.vmap(lambda s, t: 2 / float(out_features) * (s - t))(y_pred, y)
    analytic_grads = torch.cat([dL_dw.reshape(dL_dw.shape[0], -1), dL_db], dim=1)
    analytical_mjp = G @ analytic_grads.T

    assert torch.allclose(analytical_mjp, mjp, atol=1e-5)


@pytest.mark.torch
@pytest.mark.parametrize(
    "in_features, out_features, train_set_size",
    [(46, 1, 1000), (50, 3, 100), (100, 5, 512), (25, 10, 734)],
)
def test_mixed_derivatives(in_features, out_features, train_set_size):
    A, b = linear_model((out_features, in_features), 5)
    loss = torch.nn.functional.mse_loss
    model = linear_mvp_model(A, b).model

    data_model = lambda x: np.random.normal(x @ A.T + b, DATA_OUTPUT_NOISE)
    train_x = np.random.uniform(size=[train_set_size, in_features])
    train_y = data_model(train_x)

    params = {k: p for k, p in model.named_parameters() if p.requires_grad}

    test_derivative = linear_mixed_second_derivative_analytical(
        (A, b),
        train_x,
        train_y,
    )

    functorch_mixed_derivatives = per_sample_mixed_derivative(model, loss)(
        params, torch.as_tensor(train_x), torch.as_tensor(train_y)
    )
    flat_functorch_mixed_derivatives = flatten_dimensions(
        functorch_mixed_derivatives.values(), keep_first_n=2
    )
    assert torch.allclose(
        torch.as_tensor(test_derivative),
        flat_functorch_mixed_derivatives.transpose(2, 1),
    )
