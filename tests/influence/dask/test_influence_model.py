import dask.array as da
import numpy as np
import pytest
import torch
from distributed import Client
from torch.utils.data import DataLoader, TensorDataset

from pydvl.influence import InfluenceType, InversionMethod
from pydvl.influence.base_influence_model import UnSupportedInfluenceTypeException
from pydvl.influence.dask import DaskInfluence
from pydvl.influence.dask.influence_model import (
    DimensionChunksException,
    UnalignedChunksException,
)
from pydvl.influence.torch.influence_model import (
    ArnoldiInfluence,
    BatchCgInfluence,
    DirectInfluence,
)
from tests.influence.torch.conftest import minimal_training

dimensions = (50, 1)
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
    dask_fac = dask_fac.compute(scheduler="synchronous")
    torch_fac = influence_model.factors(t_x, t_y).numpy()
    assert np.allclose(dask_fac, torch_fac, atol=1e-5, rtol=1e-3)


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
    dask_fac = dask_fac.compute(scheduler="synchronous")
    torch_fac = influence_model.values(
        t_x_test, t_y_test, t_x, t_y, influence_type=influence_type
    ).numpy()
    assert np.allclose(dask_fac, torch_fac, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize(
    "influence_type", [InfluenceType.Up, InfluenceType.Perturbation]
)
@pytest.mark.torch
def test_dask_influence_nn(influence_type):
    input_dim = (5, 5, 5)
    output_dim = 3

    x_train = torch.rand((20, *input_dim))
    y_train = torch.rand((20, output_dim))
    x_test = torch.rand((10, *input_dim))
    y_test = torch.rand((10, output_dim))

    train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=5)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=5, out_channels=3, kernel_size=3),
        torch.nn.Flatten(),
        torch.nn.Linear(27, 3),
    )
    inf_model = ArnoldiInfluence(
        model,
        torch.nn.functional.mse_loss,
        hessian_regularization=0.1,
        train_dataloader=train_dataloader,
    )

    dask_influence = DaskInfluence(
        inf_model, lambda x: x.numpy(), lambda x: torch.from_numpy(x)
    )

    da_x_train = da.from_array(x_train.numpy(), chunks=(4, -1, -1, -1))
    da_y_train = da.from_array(y_train.numpy(), chunks=(4, -1))
    da_x_test = da.from_array(x_test.numpy(), chunks=(4, -1, -1, -1))
    da_y_test = da.from_array(y_test.numpy(), chunks=(4, -1))

    da_factors = dask_influence.factors(da_x_test, da_y_test)
    torch_factors = inf_model.factors(x_test, y_test)
    assert np.allclose(
        da_factors.compute(scheduler="synchronous"),
        torch_factors.numpy(),
        atol=1e-6,
        rtol=1e-3,
    )

    torch_values_from_factors = inf_model.values_from_factors(
        torch_factors, x_train, y_train, influence_type=influence_type
    )

    da_values_from_factors = dask_influence.values_from_factors(
        da_factors, da_x_train, da_y_train, influence_type=influence_type
    )

    assert np.allclose(
        da_values_from_factors.compute(scheduler="synchronous"),
        torch_values_from_factors.numpy(),
        atol=1e-6,
        rtol=1e-3,
    )

    da_values = dask_influence.values(
        da_x_test, da_y_test, da_x_train, da_y_train, influence_type=influence_type
    ).compute(scheduler="synchronous")
    torch_values = inf_model.values(
        x_test, y_test, x_train, y_train, influence_type=influence_type
    )
    assert np.allclose(da_values, torch_values.numpy(), atol=1e-6, rtol=1e-3)

    da_sym_values = dask_influence.values(
        da_x_train, da_y_train, influence_type=influence_type
    ).compute(scheduler="synchronous")
    torch_sym_values = inf_model.values(x_train, y_train, influence_type=influence_type)
    assert np.allclose(da_sym_values, torch_sym_values.numpy(), atol=1e-6, rtol=1e-3)

    with pytest.raises(UnSupportedInfluenceTypeException):
        dask_influence.values(
            da_x_test,
            da_y_test,
            da_x_train,
            da_y_train,
            influence_type="fancy_influence",
        )

    with pytest.raises(ValueError):
        dask_influence.values(da_x_test, da_y_test, da_x_train)

    with pytest.raises(ValueError):
        dask_influence.values(da_x_test, da_y_test, x=None, y=da_y_train)

    # test distributed scheduler
    if influence_type == InfluenceType.Up:
        with Client(threads_per_worker=1):
            dask_influence_client = DaskInfluence(
                inf_model, lambda x: x.numpy(), lambda x: torch.from_numpy(x)
            )
            da_factors_client = dask_influence_client.factors(da_x_test, da_y_test)
            np.allclose(torch_factors.numpy(), da_factors_client.compute())

    with pytest.raises(DimensionChunksException):
        da_x_test_wrong_chunks = da.from_array(x_test.numpy(), chunks=(4, 2, -1, -1))
        da_y_test_wrong_chunks = da.from_array(y_test.numpy(), chunks=(4, -1))
        dask_influence.factors(da_x_test_wrong_chunks, da_y_test_wrong_chunks)

    with pytest.raises(UnalignedChunksException):
        da_x_test_unaligned_chunks = da.from_array(
            x_test.numpy(), chunks=(4, -1, -1, -1)
        )
        da_y_test_unaligned_chunks = da.from_array(y_test.numpy(), chunks=(3, -1))
        dask_influence.factors(da_x_test_unaligned_chunks, da_y_test_unaligned_chunks)
