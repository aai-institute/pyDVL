from time import time

from pydvl.influence import InfluenceType
from pydvl.influence.dask.influence_model import DaskInfluenceFunctionCalculator
from pydvl.influence.torch.util import torch_dataset_to_dask_array
#from dask_cuda import LocalCUDACluster

if __name__ == '__main__':
    from distributed import Client, LocalCluster
    import logging
    import torch
    import dask.array as da
    import dask
    from torch.utils.data import DataLoader, TensorDataset

    from pydvl.influence.torch.influence_model import BatchCgInfluence, DirectInfluence, ArnoldiInfluence, \
        LissaInfluence

    logging.basicConfig(level=logging.INFO)
    dimensions = (int(1e+5), 10)
    num_params = (dimensions[0] + 1) * dimensions[1]
    num_data = int(1e5)
    chunk_size = 50
    test_chunk_size = chunk_size
    t_x = torch.rand(num_data, dimensions[0])
    t_y = torch.rand(num_data, dimensions[1])
    t_x_test = torch.rand(int(num_data/10), dimensions[0])
    t_y_test = torch.rand(int(num_data/10), dimensions[1])

    data_loader = DataLoader(TensorDataset(t_x, t_y), batch_size=chunk_size)
    torch_model = torch.nn.Linear(*dimensions, bias=True)
    #da_x = da.from_array(t_x.numpy(), chunks=(chunk_size, -1))
    #da_y = da.from_array(t_y.numpy(), chunks=(chunk_size, -1))
    da_x, da_y = torch_dataset_to_dask_array(TensorDataset(t_x, t_y), chunk_size=chunk_size)
    da_x_test, da_y_test = torch_dataset_to_dask_array(TensorDataset(t_x_test, t_y_test), chunk_size=test_chunk_size)

    da_x_path = "da_x"
    da_y_path = "da_y"
    da_x_test_path = "da_x_test"
    da_y_test_path = "da_y_test"

    da_x.to_zarr(da_x_path, overwrite=True)
    da_y.to_zarr(da_y_path, overwrite=True)
    da_x_test.to_zarr(da_x_test_path, overwrite=True)
    da_y_test.to_zarr(da_y_test_path, overwrite=True)
    del da_x, da_y, da_x_test, da_y_test

    da_x = da.from_zarr(da_x_path)
    da_y = da.from_zarr(da_y_path)
    da_x_test = da.from_zarr(da_x_test_path)
    da_y_test = da.from_zarr(da_y_test_path)
    """
    """


    influence_model = ArnoldiInfluence(torch_model.eval(), torch.nn.functional.mse_loss, train_dataloader=data_loader, hessian_regularization=0.2)
    #influence_model = LissaInfluence(torch_model.eval(), torch.nn.functional.mse_loss, train_dataloader=data_loader, hessian_regularization=0.1, maxiter=10)

    print("Start dask")
    with Client(threads_per_worker=1, silence_logs=logging.CRITICAL) as client:
        start = time()
        dask_inf = DaskInfluenceFunctionCalculator(influence_model, lambda t: t.numpy(), lambda t: torch.from_numpy(t))
        all_fac = dask_inf.influence_on_parameters(da_x, da_y)
        #all_fac = all_fac.x.compute()
        all_fac.x.to_zarr("inf_values_zarr", overwrite=True)
        duration = time() - start
        print(f"{duration=}")


