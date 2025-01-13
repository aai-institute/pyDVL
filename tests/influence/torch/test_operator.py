from dataclasses import astuple

import pytest
import torch.nn.functional
from torch.utils.data import DataLoader, TensorDataset

from pydvl.influence.torch.operator import HessianOperator

from .test_util import test_parameters


@pytest.mark.torch
@pytest.mark.parametrize(
    "model_data, batch_size",
    [(astuple(tp.model_params), tp.batch_size) for tp in test_parameters],
    indirect=["model_data"],
)
def test_hessian_operator(model_data, batch_size: int):
    torch_model, x, y, vec, h_analytical = model_data
    data_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)
    hvp_autograd = HessianOperator(
        torch_model, torch.nn.functional.mse_loss, data_loader
    ).apply(vec)
    assert torch.allclose(hvp_autograd, h_analytical @ vec)
