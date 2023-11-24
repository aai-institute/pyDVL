import dask.array as da
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pydvl.influence import InfluenceType, InversionMethod
from pydvl.influence.dask import DaskInfluence
from pydvl.influence.torch.influence_model import (
    ArnoldiInfluence,
    BatchCgInfluence,
    DirectInfluence,
)
from tests.influence.torch.conftest import minimal_training

dimensions = (50, 2)
num_params = (dimensions[0] + 1) * dimensions[1]
num_data = 50
chunk_size = 25
t_x = torch.rand(num_data, dimensions[0])
t_y = torch.rand(num_data, dimensions[1])
t_x_test = torch.rand(int(num_data / 10), dimensions[0])
t_y_test = torch.rand(int(num_data / 10), dimensions[1])
data_loader = DataLoader(TensorDataset(t_x, t_y), batch_size=chunk_size)
da_x = da.from_array(t_x.numpy(), chunks=(chunk_size, -1))
da_y = da.from_array(t_y.numpy(), chunks=(chunk_size, -1))
da_x_test = da.from_array(t_x_test.numpy(), chunks=(int(chunk_size / 10), -1))
da_y_test = da.from_array(t_y_test.numpy(), chunks=(int(chunk_size / 10), -1))


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
def test_dask_influence_factors(influence_model):
    dask_inf = DaskInfluence(
        influence_model, lambda t: t.numpy(), lambda t: torch.from_numpy(t)
    )
    dask_fac = dask_inf.factors(da_x, da_y)
    dask_fac = dask_fac.compute(scheduler="processes")
    torch_fac = influence_model.factors(t_x, t_y).numpy()
    assert np.allclose(dask_fac, torch_fac, atol=1e-6, rtol=1e-3)


@pytest.mark.parametrize(
    "influence_type", [InfluenceType.Up, InfluenceType.Perturbation]
)
@pytest.mark.torch
def test_dask_influence_values(influence_model, influence_type):
    dask_inf = DaskInfluence(
        influence_model, lambda t: t.numpy(), lambda t: torch.from_numpy(t)
    )
    dask_fac = dask_inf.values(
        da_x_test, da_y_test, da_x, da_y, influence_type=influence_type
    )
    dask_fac = dask_fac.compute(scheduler="processes")
    torch_fac = influence_model.values(
        t_x_test, t_y_test, t_x, t_y, influence_type=influence_type
    ).numpy()
    assert np.allclose(dask_fac, torch_fac, atol=1e-6, rtol=1e-3)
