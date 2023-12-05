import dask.array as da
import numpy as np
import pytest
import torch
from distributed import Client
from torch.utils.data import DataLoader, TensorDataset

from pydvl.influence import InfluenceType
from pydvl.influence.base_influence_model import UnSupportedInfluenceTypeException
from pydvl.influence.dask import DaskInfluenceCalculator
from pydvl.influence.dask.influence_calculator import (
    DimensionChunksException,
    UnalignedChunksException,
)
from pydvl.influence.torch.influence_model import (
    ArnoldiInfluence,
    BatchCgInfluence,
    DirectInfluence,
)
from tests.influence.torch.test_influence_model import model_and_data, test_case


@pytest.fixture
@pytest.mark.parametrize(
    "influence_factory",
    [
        lambda model, loss, train_dataLoader, hessian_reg: BatchCgInfluence(
            model, loss, train_dataLoader
        ).fit(train_dataLoader),
        lambda model, loss, train_dataLoader, hessian_reg: DirectInfluence(
            model, loss, hessian_reg
        ).fit(train_dataLoader),
        lambda model, loss, train_dataLoader, hessian_reg: ArnoldiInfluence(
            model,
            loss,
            hessian_regularization=hessian_reg,
        ).fit(train_dataLoader),
    ],
    ids=["cg", "direct", "arnoldi"],
)
def influence_model(model_and_data, test_case, influence_factory):
    model, _, x_train, y_train, _, _ = model_and_data
    train_dataloader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=test_case.batch_size
    )
    return influence_factory(
        model, test_case.loss, train_dataloader, test_case.hessian_reg
    )


@pytest.mark.torch
@pytest.mark.parametrize(
    "influence_factory",
    [
        lambda model, loss, train_dataLoader, hessian_reg: BatchCgInfluence(
            model, loss, hessian_reg
        ).fit(train_dataLoader),
        lambda model, loss, train_dataLoader, hessian_reg: DirectInfluence(
            model, loss, hessian_reg
        ).fit(train_dataLoader),
        lambda model, loss, train_dataLoader, hessian_reg: ArnoldiInfluence(
            model,
            loss,
            hessian_regularization=hessian_reg,
        ).fit(train_dataLoader),
    ],
    ids=["cg", "direct", "arnoldi"],
)
def test_dask_influence_factors(influence_factory, test_case, model_and_data):
    model, loss, x_train, y_train, x_test, y_test = model_and_data
    chunk_size = int(test_case.train_data_len / 4)
    da_x_train = da.from_array(
        x_train.numpy(), chunks=(chunk_size, *[-1 for _ in x_train.shape[1:]])
    )
    da_y_train = da.from_array(
        y_train.numpy(), chunks=(chunk_size, *[-1 for _ in y_train.shape[1:]])
    )
    da_x_test = da.from_array(
        x_test.numpy(), chunks=(chunk_size, *[-1 for _ in x_test.shape[1:]])
    )
    da_y_test = da.from_array(
        y_test.numpy(), chunks=(chunk_size, *[-1 for _ in y_test.shape[1:]])
    )
    train_dataloader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=test_case.batch_size
    )
    influence_model = influence_factory(
        model, test_case.loss, train_dataloader, test_case.hessian_reg
    )
    dask_inf = DaskInfluenceCalculator(
        influence_model, lambda t: t.numpy(), lambda t: torch.from_numpy(t)
    )
    dask_fac = dask_inf.influence_factors(da_x_train, da_y_train)
    dask_fac = dask_fac.compute(scheduler="synchronous")
    torch_fac = influence_model.influence_factors(x_train, y_train).numpy()
    assert np.allclose(dask_fac, torch_fac, atol=1e-5, rtol=1e-3)

    dask_val = dask_inf.influences(
        da_x_test,
        da_y_test,
        da_x_train,
        da_y_train,
        influence_type=test_case.influence_type,
    )
    dask_val = dask_val.compute(scheduler="synchronous")
    torch_val = influence_model.influences(
        x_test, y_test, x_train, y_train, influence_type=test_case.influence_type
    ).numpy()
    assert np.allclose(dask_val, torch_val, atol=1e-5, rtol=1e-3)


@pytest.mark.torch
def test_dask_influence_nn(model_and_data, test_case):
    model, loss, x_train, y_train, x_test, y_test = model_and_data
    chunk_size = int(test_case.train_data_len / 4)
    test_chunk_size = int(test_case.test_data_len / 4)
    da_x_train = da.from_array(
        x_train.numpy(), chunks=(chunk_size, *[-1 for _ in x_train.shape[1:]])
    )
    da_y_train = da.from_array(
        y_train.numpy(), chunks=(chunk_size, *[-1 for _ in y_train.shape[1:]])
    )
    da_x_test = da.from_array(
        x_test.numpy(), chunks=(test_chunk_size, *[-1 for _ in x_test.shape[1:]])
    )
    da_y_test = da.from_array(
        y_test.numpy(), chunks=(test_chunk_size, *[-1 for _ in y_test.shape[1:]])
    )

    train_dataloader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=test_case.batch_size
    )
    inf_model = ArnoldiInfluence(
        model,
        test_case.loss,
        hessian_regularization=test_case.hessian_reg,
    ).fit(train_dataloader)

    dask_influence = DaskInfluenceCalculator(
        inf_model, lambda x: x.numpy(), lambda x: torch.from_numpy(x)
    )

    da_factors = dask_influence.influence_factors(da_x_test, da_y_test)
    torch_factors = inf_model.influence_factors(x_test, y_test)
    assert np.allclose(
        da_factors.compute(scheduler="synchronous"),
        torch_factors.numpy(),
        atol=1e-6,
        rtol=1e-3,
    )

    torch_values_from_factors = inf_model.influences_from_factors(
        torch_factors, x_train, y_train, influence_type=test_case.influence_type
    )

    da_values_from_factors = dask_influence.influences_from_factors(
        da_factors, da_x_train, da_y_train, influence_type=test_case.influence_type
    )

    assert np.allclose(
        da_values_from_factors.compute(scheduler="synchronous"),
        torch_values_from_factors.numpy(),
        atol=1e-6,
        rtol=1e-3,
    )

    da_values = dask_influence.influences(
        da_x_test,
        da_y_test,
        da_x_train,
        da_y_train,
        influence_type=test_case.influence_type,
    ).compute(scheduler="synchronous")
    torch_values = inf_model.influences(
        x_test, y_test, x_train, y_train, influence_type=test_case.influence_type
    )
    assert np.allclose(da_values, torch_values.numpy(), atol=1e-6, rtol=1e-3)

    da_sym_values = dask_influence.influences(
        da_x_train, da_y_train, influence_type=test_case.influence_type
    ).compute(scheduler="synchronous")
    torch_sym_values = inf_model.influences(
        x_train, y_train, influence_type=test_case.influence_type
    )
    assert np.allclose(da_sym_values, torch_sym_values.numpy(), atol=1e-6, rtol=1e-3)

    with pytest.raises(UnSupportedInfluenceTypeException):
        dask_influence.influences(
            da_x_test,
            da_y_test,
            da_x_train,
            da_y_train,
            influence_type="fancy_influence",
        )

    with pytest.raises(ValueError):
        dask_influence.influences(da_x_test, da_y_test, da_x_train)

    with pytest.raises(ValueError):
        dask_influence.influences(da_x_test, da_y_test, x=None, y=da_y_train)

    # test distributed scheduler
    if test_case.influence_type == InfluenceType.Up:
        with Client(threads_per_worker=1):
            dask_influence_client = DaskInfluenceCalculator(
                inf_model, lambda x: x.numpy(), lambda x: torch.from_numpy(x)
            )
            da_factors_client = dask_influence_client.influence_factors(
                da_x_test, da_y_test
            )
            np.allclose(torch_factors.numpy(), da_factors_client.compute())

    with pytest.raises(DimensionChunksException):
        da_x_test_wrong_chunks = da.from_array(
            x_test.numpy(), chunks=(4, *[1 for _ in x_test.shape[1:]])
        )
        da_y_test_wrong_chunks = da.from_array(
            y_test.numpy(), chunks=(4, *[1 for _ in y_test.shape[1:]])
        )
        dask_influence.influence_factors(da_x_test_wrong_chunks, da_y_test_wrong_chunks)

    with pytest.raises(UnalignedChunksException):
        da_x_test_unaligned_chunks = da.from_array(
            x_test.numpy(), chunks=(4, *[-1 for _ in x_test.shape[1:]])
        )
        da_y_test_unaligned_chunks = da.from_array(
            y_test.numpy(), chunks=(3, *[-1 for _ in y_test.shape[1:]])
        )
        dask_influence.influence_factors(
            da_x_test_unaligned_chunks, da_y_test_unaligned_chunks
        )
