import dask.array as da
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pydvl.influence import InversionMethod
from pydvl.influence.dask import DaskInfluence
from pydvl.influence.torch.influence_model import (
    ArnoldiInfluence,
    BatchCgInfluence,
    DirectInfluence,
)
from tests.influence.test_influences import minimal_training

dimensions = (50, 2)
num_params = (dimensions[0] + 1) * dimensions[1]
num_data = 100
chunk_size = 10
test_chunk_size = 5
t_x = torch.rand(num_data, dimensions[0])
t_y = torch.rand(num_data, dimensions[1])
t_x_test = torch.rand(int(num_data / 10), dimensions[0])
t_y_test = torch.rand(int(num_data / 10), dimensions[1])
data_loader = DataLoader(TensorDataset(t_x, t_y), batch_size=chunk_size)
da_x = da.from_array(t_x.numpy(), chunks=(chunk_size, -1))
da_y = da.from_array(t_y.numpy(), chunks=(chunk_size, -1))


@pytest.fixture(scope="session")
def trained_linear_model():
    torch_model = torch.nn.Linear(*dimensions, bias=True)
    return minimal_training(torch_model, data_loader, torch.nn.functional.mse_loss)


@pytest.fixture(
    params=[InversionMethod.Cg, InversionMethod.Direct, InversionMethod.Arnoldi],
    scope="session",
)
def influence_model(request, trained_linear_model):
    if request.param is InversionMethod.Direct:
        influence = DirectInfluence(
            trained_linear_model,
            torch.nn.functional.mse_loss,
            hessian_regularization=0.1,
            train_dataloader=data_loader,
        )
    elif request.param is InversionMethod.Arnoldi:
        influence = ArnoldiInfluence(
            trained_linear_model,
            torch.nn.functional.mse_loss,
            hessian_regularization=0.1,
            train_dataloader=data_loader,
        )
    else:
        influence = BatchCgInfluence(
            trained_linear_model,
            torch.nn.functional.mse_loss,
            hessian_regularization=0.1,
            train_dataloader=data_loader,
            maxiter=10,
        )
    return influence


@pytest.mark.torch
def test_dask_influence(influence_model):
    dask_inf = DaskInfluence(
        influence_model, lambda t: t.numpy(), lambda t: torch.from_numpy(t)
    )
    dask_fac = dask_inf.factors(da_x, da_y)
    dask_fac = dask_fac.x.compute(scheduler="processes")
    torch_fac = influence_model.factors(t_x, t_y).x.numpy()
    assert np.allclose(dask_fac, torch_fac, rtol=1e-2)
