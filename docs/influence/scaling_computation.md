---
title: Scaling Influence Computation
---

# Scaling Influence Computation

The implementations of [InfluenceFunctionModel][pydvl.influence.base_influence_function_model.InfluenceFunctionModel]
provide a convenient way to calculate influences for
in memory tensors. 

Nevertheless, there is a need for computing the influences on batches of data. This might
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
from pydvl.influence.torch.util import (
    TorchNumpyConverter, NestedTorchCatAggregator,
)
from pydvl.influence.torch import CgInfluence

batch_size = 10
train_dataloader = DataLoader(..., batch_size=batch_size)
test_dataloader = DataLoader(..., batch_size=batch_size)

infl_model = CgInfluence(model, loss, hessian_regularization=0.01)
infl_model = infl_model.fit(train_dataloader)

infl_calc = SequentialInfluenceCalculator(infl_model)

# this does not trigger the computation
lazy_influences = infl_calc.influences(test_dataloader, train_dataloader)

# trigger computation and pull the result into main memory, 
# result is the full tensor for all combinations of the two loaders
influences = lazy_influences.compute(aggregator=NestedTorchCatAggregator())
# or
# trigger computation and write results chunk-wise to disk using zarr 
# in a sequential manner
lazy_influences.to_zarr("local_path/or/url", TorchNumpyConverter())
```
When invoking the `compute` method, you have the option to specify a custom aggregator 
by implementing [NestedSequenceAggregator][pydvl.influence.array.NestedSequenceAggregator]. 
This allows for the aggregation of computed chunks. 
Such an approach is particularly beneficial for straightforward aggregation tasks, 
commonly seen in sequential computation models. 
Examples include operations like concatenation, as implemented in 
[NestedTorchCatAggregator][pydvl.influence.torch.util.NestedTorchCatAggregator], 
or basic **min** and **max** operations. 

For more intricate aggregations, such as an **argmax** operation, 
it's advisable to use the 
[DaskInfluenceCalculator][pydvl.influence.influence_calculator.DaskInfluenceCalculator] 
(refer to [Parallel](#parallel) for more details). This is because it returns data structures in the 
form of [dask.array.Array][dask.array.Array] objects, which offer an API almost fully 
compatible with NumPy arrays.

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
    Make sure to set `threads_per_worker=1`, when using the distributed scheduler for computing,
    if your implementation of [InfluenceFunctionModel][pydvl.influence.base_influence_function_model.InfluenceFunctionModel]
    is not thread-safe.
    ```python
    client = Client(threads_per_worker=1)
    ```
    For details on dask schedulers see the [official documentation](https://docs.dask.org/en/stable/scheduling.html).

```python
import torch
from torch.utils.data import Dataset, DataLoader
from pydvl.influence import DaskInfluenceCalculator
from pydvl.influence.torch import CgInfluence
from pydvl.influence.torch.util import (
    torch_dataset_to_dask_array,
    TorchNumpyConverter,
)
from distributed import Client

train_data_set: Dataset = LargeDataSet(
    ...)  # Possible some out of memory large Dataset
test_data_set: Dataset = LargeDataSet(
    ...)  # Possible some out of memory large Dataset

train_dataloader = DataLoader(train_data_set)
infl_model = CgInfluence(model, loss, hessian_regularization=0.01)
infl_model = infl_model.fit(train_dataloader)

# wrap your input data into dask arrays
chunk_size = 10
da_x, da_y = torch_dataset_to_dask_array(train_data_set, chunk_size=chunk_size)
da_x_test, da_y_test = torch_dataset_to_dask_array(test_data_set,
                                                   chunk_size=chunk_size)

# use only one thread for scheduling, 
# due to non-thread safety of some torch operations
client = Client(n_workers=4, threads_per_worker=1)

infl_calc = DaskInfluenceCalculator(infl_model, 
                                  converter=TorchNumpyConverter(
                                      device=torch.device("cpu")
                                  ),
                                  client=client)
da_influences = infl_calc.influences(da_x_test, da_y_test, da_x, da_y)
# da_influences is a dask.array.Array
# trigger computation and write chunks to disk in parallel
da_influences.to_zarr("path/or/url")
```
During initialization of the 
[DaskInfluenceCalculator][pydvl.influence.influence_calculator.DaskInfluenceCalculator], 
the system verifies if all workers are operating in
single-threaded mode when the provided influence_function_model is
designated as not thread-safe (indicated by the `is_thread_safe` property).
If this condition is not met, the initialization will raise a specific
error, signaling a potential thread-safety conflict.

To intentionally skip this safety check
(e.g., for debugging purposes using the single machine synchronous
scheduler), you can supply the [DisableClientSingleThreadCheck]
[pydvl.influence.influence_calculator.DisableClientSingleThreadCheck] type.

```python
from pydvl.influence import DisableClientSingleThreadCheck

infl_calc = DaskInfluenceCalculator(infl_model,
                                    TorchNumpyConverter(device=torch.device("cpu")),
                                    DisableClientSingleThreadCheck)
da_influences = infl_calc.influences(da_x_test, da_y_test, da_x, da_y)
da_influences.compute(scheduler="synchronous")
```

