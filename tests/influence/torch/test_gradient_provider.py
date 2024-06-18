import numpy as np
import pytest
import torch

from pydvl.influence.torch.base import TorchBatch, TorchGradientProvider

from ..conftest import linear_mixed_second_derivative_analytical, linear_model
from .conftest import DATA_OUTPUT_NOISE, linear_mvp_model


class TestTorchPerSampleAutograd:
    @pytest.mark.torch
    @pytest.mark.parametrize(
        "in_features, out_features, batch_size",
        [(46, 6, 632), (50, 3, 120), (100, 5, 120), (25, 10, 550)],
    )
    def test_per_sample_gradient(self, in_features, out_features, batch_size):
        model = torch.nn.Linear(in_features, out_features)
        loss = torch.nn.functional.mse_loss

        x = torch.randn(batch_size, in_features, requires_grad=True)
        y = torch.randn(batch_size, out_features)
        params = {k: p.detach() for k, p in model.named_parameters() if p.requires_grad}

        gp = TorchGradientProvider(model, loss, restrict_to=params)
        gradients = gp.grads(TorchBatch(x, y))
        flat_gradients = gp.flat_grads(TorchBatch(x, y))

        # Compute analytical gradients
        y_pred = model(x)
        dL_dw = torch.vmap(
            lambda r, s, t: 2
            / float(out_features)
            * (s - t).view(-1, 1)
            @ r.view(1, -1)
        )(x, y_pred, y)
        dL_db = torch.vmap(lambda s, t: 2 / float(out_features) * (s - t))(y_pred, y)

        # Assert the gradient values for equality with analytical gradients
        assert torch.allclose(gradients["weight"], dL_dw, atol=1e-5)
        assert torch.allclose(gradients["bias"], dL_db, atol=1e-5)
        assert torch.allclose(
            flat_gradients,
            torch.cat([dL_dw.reshape(batch_size, -1), dL_db], dim=-1),
            atol=1e-5,
        )

    @pytest.mark.torch
    @pytest.mark.parametrize(
        "in_features, out_features, train_set_size",
        [(46, 1, 1000), (50, 3, 100), (100, 5, 512), (25, 10, 734)],
    )
    def test_mixed_derivatives(self, in_features, out_features, train_set_size):
        A, b = linear_model((out_features, in_features), 5)
        loss = torch.nn.functional.mse_loss
        model = linear_mvp_model(A, b)

        data_model = lambda x: np.random.normal(x @ A.T + b, DATA_OUTPUT_NOISE)
        train_x = np.random.uniform(size=[train_set_size, in_features])
        train_y = data_model(train_x)

        params = {k: p for k, p in model.named_parameters() if p.requires_grad}

        test_derivative = linear_mixed_second_derivative_analytical(
            (A, b),
            train_x,
            train_y,
        )

        torch_train_x = torch.as_tensor(train_x)
        torch_train_y = torch.as_tensor(train_y)
        gp = TorchGradientProvider(model, loss, restrict_to=params)
        flat_functorch_mixed_derivatives = gp.flat_mixed_grads(
            TorchBatch(torch_train_x, torch_train_y)
        )
        assert torch.allclose(
            torch.as_tensor(test_derivative),
            flat_functorch_mixed_derivatives.transpose(2, 1),
        )

    @pytest.mark.torch
    @pytest.mark.parametrize(
        "in_features, out_features, batch_size",
        [(46, 1, 632), (50, 3, 120), (100, 5, 110), (25, 10, 500)],
    )
    def test_matrix_jacobian_product(
        self, in_features, out_features, batch_size, pytorch_seed
    ):
        model = torch.nn.Linear(in_features, out_features)
        params = {k: p for k, p in model.named_parameters() if p.requires_grad}

        x = torch.randn(batch_size, in_features, requires_grad=True)
        y = torch.randn(batch_size, out_features, requires_grad=True)
        y_pred = model(x)

        gp = TorchGradientProvider(
            model, torch.nn.functional.mse_loss, restrict_to=params
        )

        G = torch.randn((10, out_features * (in_features + 1)))
        mjp = gp.jacobian_prod(TorchBatch(x, y), G)

        dL_dw = torch.vmap(
            lambda r, s, t: 2
            / float(out_features)
            * (s - t).view(-1, 1)
            @ r.view(1, -1)
        )(x, y_pred, y)
        dL_db = torch.vmap(lambda s, t: 2 / float(out_features) * (s - t))(y_pred, y)
        analytic_grads = torch.cat([dL_dw.reshape(dL_dw.shape[0], -1), dL_db], dim=1)
        analytical_mjp = G @ analytic_grads.T

        assert torch.allclose(analytical_mjp, mjp, atol=1e-5, rtol=1e-3)
