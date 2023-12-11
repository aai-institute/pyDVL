import shutil
import uuid

import dask.array as da
import numpy as np
import pytest
import torch
from distributed import Client
from torch.utils.data import DataLoader, TensorDataset

from pydvl.influence import DaskInfluenceCalculator, InfluenceType
from pydvl.influence.base_influence_model import UnsupportedInfluenceTypeException
from pydvl.influence.influence_calculator import (
    InvalidDimensionChunksError,
    SequentialInfluenceCalculator,
    UnalignedChunksError,
)
from pydvl.influence.torch import ArnoldiInfluence, CgInfluence, DirectInfluence
from pydvl.influence.torch.util import TorchCatAggregator, TorchNumpyConverter
from tests.influence.torch.test_influence_model import model_and_data, test_case


@pytest.fixture
@pytest.mark.parametrize(
    "influence_factory",
    [
        lambda model, loss, train_dataLoader, hessian_reg: CgInfluence(
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
        lambda model, loss, train_dataLoader, hessian_reg: CgInfluence(
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
    numpy_converter = TorchNumpyConverter()
    dask_inf = DaskInfluenceCalculator(influence_model, numpy_converter)
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

    numpy_converter = TorchNumpyConverter()
    dask_influence = DaskInfluenceCalculator(inf_model, numpy_converter)

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

    with pytest.raises(UnsupportedInfluenceTypeException):
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
            numpy_converter = TorchNumpyConverter()
            dask_influence_client = DaskInfluenceCalculator(inf_model, numpy_converter)
            da_factors_client = dask_influence_client.influence_factors(
                da_x_test, da_y_test
            )
            np.allclose(torch_factors.numpy(), da_factors_client.compute())

    with pytest.raises(InvalidDimensionChunksError):
        da_x_test_wrong_chunks = da.from_array(
            x_test.numpy(), chunks=(4, *[1 for _ in x_test.shape[1:]])
        )
        da_y_test_wrong_chunks = da.from_array(
            y_test.numpy(), chunks=(4, *[1 for _ in y_test.shape[1:]])
        )
        dask_influence.influence_factors(da_x_test_wrong_chunks, da_y_test_wrong_chunks)

    with pytest.raises(UnalignedChunksError):
        da_x_test_unaligned_chunks = da.from_array(
            x_test.numpy(), chunks=(4, *[-1 for _ in x_test.shape[1:]])
        )
        da_y_test_unaligned_chunks = da.from_array(
            y_test.numpy(), chunks=(3, *[-1 for _ in y_test.shape[1:]])
        )
        dask_influence.influence_factors(
            da_x_test_unaligned_chunks, da_y_test_unaligned_chunks
        )


def test_sequential_calculator(model_and_data, test_case):
    model, loss, x_train, y_train, x_test, y_test = model_and_data
    train_dataloader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=test_case.batch_size
    )
    test_dataloader = DataLoader(
        TensorDataset(x_test, y_test), batch_size=test_case.batch_size
    )

    inf_model = ArnoldiInfluence(
        model,
        test_case.loss,
        hessian_regularization=test_case.hessian_reg,
    ).fit(train_dataloader)

    block_aggregator = TorchCatAggregator()
    seq_calculator = SequentialInfluenceCalculator(inf_model)

    seq_factors_lazy_array = seq_calculator.influence_factors(test_dataloader)
    seq_factors = seq_factors_lazy_array.compute(block_aggregator)

    torch_factors = inf_model.influence_factors(x_test, y_test)
    zarr_factors_path = str(uuid.uuid4())
    seq_factors_from_zarr = seq_factors_lazy_array.to_zarr(
        zarr_factors_path, TorchNumpyConverter(), return_stored=True
    )
    assert torch.allclose(seq_factors, torch_factors, atol=1e-6)
    assert np.allclose(torch_factors.numpy(), seq_factors_from_zarr, atol=1e-6)
    shutil.rmtree(zarr_factors_path)

    torch_values_from_factors = inf_model.influences_from_factors(
        torch_factors, x_train, y_train, influence_type=test_case.influence_type
    )

    seq_factors_data_loader = DataLoader(
        TensorDataset(seq_factors), batch_size=test_case.batch_size
    )

    seq_values_from_factors_lazy_array = seq_calculator.influences_from_factors(
        seq_factors_data_loader,
        train_dataloader,
        influence_type=test_case.influence_type,
    )
    seq_values_from_factors = seq_values_from_factors_lazy_array.compute(
        block_aggregator
    )
    zarr_values_from_factors_path = str(uuid.uuid4())
    seq_values_from_factors_from_zarr = seq_values_from_factors_lazy_array.to_zarr(
        zarr_values_from_factors_path, TorchNumpyConverter(), return_stored=True
    )

    assert torch.allclose(seq_values_from_factors, torch_values_from_factors, atol=1e-6)
    assert np.allclose(
        seq_values_from_factors_from_zarr, torch_values_from_factors.numpy(), atol=1e-6
    )
    shutil.rmtree(zarr_values_from_factors_path)

    seq_values_lazy_array = seq_calculator.influences(
        test_dataloader, train_dataloader, influence_type=test_case.influence_type
    )
    seq_values = seq_values_lazy_array.compute(block_aggregator)
    zarr_values_path = str(uuid.uuid4())
    seq_values_from_zarr = seq_values_lazy_array.to_zarr(
        zarr_values_path, TorchNumpyConverter(), return_stored=True
    )

    torch_values = inf_model.influences(
        x_test, y_test, x_train, y_train, influence_type=test_case.influence_type
    )
    assert torch.allclose(seq_values, torch_values, atol=1e-6)
    assert np.allclose(seq_values_from_zarr, torch_values.numpy(), atol=1e-6)
    shutil.rmtree(zarr_values_path)
