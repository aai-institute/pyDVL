The implementations of [InfluenceFunctionModel][pydvl.influence.base_influence_model.InfluenceFunctionModel]
provide a convenient way to calculate influences for
in memory tensors. Nevertheless, there is a need for computing the influences on batches of data. This might
happen, if your input data does not fit into memory (e.g. it is very high-dimensional) or for large models
the derivative computations exceed your memory or any combinations of these.
For this scenario, we want to map our influence function model over collections of
batches (or chunks) of data.

## Sequential
The simplest way is to use a double for-loop
to iterate over the batches sequentially and collect them. pyDVL provides the simple convenience class
[SequentialInfluenceCalculator][pydvl.influence.influence_calculator] to do this. The
batch size should be chosen as large as possible, such that the corresponding batches fit
into memory.

```python
from pydvl.influence import SequentialInfluenceCalculator
from pydvl.influence.torch.util import TorchCatAggregator, TorchNumpyConverter
from pydvl.influence.torch import CgInfluence

batch_size = 10
train_dataloader = DataLoader(..., batch_size=batch_size)
test_dataloader = DataLoader(..., batch_size=batch_size)

if_model = CgInfluence(model, loss, hessian_regularization=0.01)
if_model = if_model.fit(train_dataloader)

seq_calc = SequentialInfluenceCalculator(if_model)

# this does not trigger the computation
lazy_influences = seq_calc.influences(test_dataloader, train_dataloader)

# trigger computation and pull the result into main memory, result is the full tensor for all combinations of the two loaders
influences = lazy_influences.compute(block_aggregator=TorchCatAggregator())
# or
# trigger computation and write results chunk-wise to disk using zarr in a sequential manner
lazy_influences.to_zarr("local_path/or/url", TorchNumpyConverter())
```

## Parallel
While the sequential calculation helps in the case the resulting tensors are too large to fit into memory, 
the batches are computed one after another. Because the influence computation itself is completely data parallel,
you may want to use a parallel processing framework. 

pyDVL provides an implementation of a parallel computation
model using [dask](https://docs.dask.org/en/stable/).
The wrapper class [DaskInfluenceCalculator][pydvl.influence.influence_calculator.DaskInfluenceCalculator]
has convenience methods to map the influence function computation over chunks of data in a parallel manner.

Again, choosing an appropriate chunk size can be crucial. For a better understanding see the
official 
[dask best practice documentation](https://docs.dask.org/en/latest/array-best-practices.html#select-a-good-chunk-size)
and the following [blog entry](https://blog.dask.org/2021/11/02/choosing-dask-chunk-sizes).

!!! Warning
    Make sure, you set `threads_per_worker=1`, when using the distributed scheduler for computing
    if your implementation of [InfluenceFunctionModel][pydvl.influence.base_influence_model.InfluenceFunctionModel]
    is not thread-safe. If you do not use the distributed scheduler,
    choose the `processes` single machine scheduler
    ```python
    client = Client(threads_per_worker=1)
    # or
    da_influences.compute(scheduler="processes")
    ```
    For details on dask schedulers see the [official documentation](https://docs.dask.org/en/stable/scheduling.html).

```python
from torch.utils.data import Dataset, DataLoader
from pydvl.influence import DaskInfluenceCalculator
from pydvl.influence.torch import CgInfluence
from pydvl.influence.torch.util import torch_dataset_to_dask_array, TorchNumpyConverter
from distributed import Client

train_data_set: Dataset = LargeDataSet(...) # Possible some out of memory large Dataset
test_data_set: Dataset = LargeDataSet(...) # Possible some out of memory large Dataset

train_dataloader = DataLoader(train_data_set)
if_model = CgInfluence(model, loss, hessian_regularization=0.01)
if_model = if_model.fit(train_dataloader)

# wrap your input data into dask arrays
chunk_size = 10
da_x, da_y = torch_dataset_to_dask_array(train_data_set, chunk_size=chunk_size)
da_x_test, da_y_test = torch_dataset_to_dask_array(test_data_set, chunk_size=chunk_size)

client = Client(n_workers=4, threads_per_worker=1)  # use only one thread for scheduling, due to non-thread safety of some torch operations

da_calc = DaskInfluenceCalculator(if_model, numpy_converter=TorchNumpyConverter())
da_influences = da_calc.influences(da_x_test, da_y_test, da_x, da_y)
# da_influences is a dask.array.Array

# trigger computation and write chunks to disk in parallel
da_influences.to_zarr("path/or/url")

```

