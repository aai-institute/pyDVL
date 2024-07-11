---
title: How-to scale up influence function with parallelization
alias:
    name: scale-up-if-with-parallel
---

Calculating influence functions on large datasets can be challenging due to
memory constraints (e.g. input data does not into memory or large models).
In these cases, we want to map our influence function model over
collections of batches (or chunks) of data.

This guide shows you how to parallelize influence function
computations using pyDVL to handle large batches of data efficiently.

!!! tip "Other guides"

    For speeding-up data valuation algorithms with parallelization
    see [[speed-up-value-with-parallel]].


### Sequential Computation

The simplest way to compute influence functions is sequentially using the
[SequentialInfluenceCalculator][pydvl.influence.influence_calculator].

This uses a double for-loop to iterate over the batches sequentially
and collects them.

We start by instantiating the dataloaders and the model and then fitting
the latter:

```python
from pydvl.influence.torch import CgInfluence
from torch.utils.data import DataLoader

batch_size = 10
train_dataloader = DataLoader(..., batch_size=batch_size)
test_dataloader = DataLoader(..., batch_size=batch_size)

infl_model = CgInfluence(model, loss, hessian_regularization=0.01)
infl_model.fit(train_dataloader)
```

We then compute influences sequentially:

```python
from pydvl.influence import SequentialInfluenceCalculator
from pydvl.influence.torch.util import NestedTorchCatAggregator

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

!!! tip "Batch size"

    The batch size should be chosen as large as possible,
    such that the corresponding batches fit into memory.

When invoking the `compute` method, you have the option to specify
a custom aggregator by implementing
[NestedSequenceAggregator][pydvl.influence.array.NestedSequenceAggregator]. 
This allows for the aggregation of computed chunks. 
Such an approach is particularly beneficial for straightforward
aggregation tasks, commonly seen in sequential computation models. 
Examples include operations like concatenation, as implemented in 
[NestedTorchCatAggregator][pydvl.influence.torch.util.NestedTorchCatAggregator], 
or basic **min** and **max** operations. 

For more complex aggregations, such as an **argmax** operation, 
consider using the parallel approach described below.

### Parallel Computation

While the sequential calculation helps in the case the resulting tensors
are too large to fit into memory, the batches are computed one after another.
Because the influence computation itself is completely data parallel,
you may want to use a parallel processing framework. 

For parallel processing, pyDVL uses [Dask](https://docs.dask.org/en/stable/).
This allows you to compute influences on data batches in parallel.

Again, choosing an appropriate chunk size can be crucial.
For a better understanding see the official 
[dask best practice documentation](https://docs.dask.org/en/latest/array-best-practices.html#select-a-good-chunk-size)
and the following [blog entry](https://blog.dask.org/2021/11/02/choosing-dask-chunk-sizes).

!!! Warning

    Make sure to set `threads_per_worker=1`, when using the distributed
    scheduler for computing, if your implementation of
    [InfluenceFunctionModel][pydvl.influence.base_influence_function_model.InfluenceFunctionModel]
    is not thread-safe.

    ```python
    client = Client(threads_per_worker=1)
    ```

    For details on dask schedulers see the
    [official documentation](https://docs.dask.org/en/stable/scheduling.html).

We start by instantiating the dataloaders and the model and then fitting
the latter:


```python
from torch.utils.data import Dataset, DataLoader
from pydvl.influence.torch import CgInfluence

train_data_set: Dataset = LargeDataSet(
    ...)  # Possible some out of memory large Dataset
test_data_set: Dataset = LargeDataSet(
    ...)  # Possible some out of memory large Dataset

train_dataloader = DataLoader(train_data_set)
infl_model = CgInfluence(model, loss, hessian_regularization=0.01)
infl_model = infl_model.fit(train_dataloader)
```

After that, we instantiate a Dask client and wrap the data into dask arrays:

```python
from pydvl.influence.torch.util import torch_dataset_to_dask_array
from distributed import Client

# use only one thread for scheduling, 
# due to non-thread safety of some torch operations
client = Client(n_workers=4, threads_per_worker=1)

# wrap your input data into dask arrays
chunk_size = 10
da_x, da_y = torch_dataset_to_dask_array(train_data_set, chunk_size=chunk_size)
da_x_test, da_y_test = torch_dataset_to_dask_array(
    test_data_set,
    chunk_size=chunk_size
)
```

Finally, we compute the influences and write them to disk:

```python
import torch
from pydvl.influence import DaskInfluenceCalculator
from pydvl.influence.torch.util import TorchNumpyConverter

infl_calc = DaskInfluenceCalculator(
    infl_model,
    converter=TorchNumpyConverter(device=torch.device("cpu")),
    client=client
)
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

#### Skipping Thread-Safety Check

To intentionally skip this safety check
(e.g., for debugging purposes using the single machine synchronous
scheduler), you can supply the [DisableClientSingleThreadCheck]
[pydvl.influence.influence_calculator.DisableClientSingleThreadCheck] type.

```python
from pydvl.influence import DisableClientSingleThreadCheck

infl_calc = DaskInfluenceCalculator(
    infl_model,
    TorchNumpyConverter(device=torch.device("cpu")),
    DisableClientSingleThreadCheck
)
da_influences = infl_calc.influences(da_x_test, da_y_test, da_x, da_y)
da_influences.compute(scheduler="synchronous")
```

## Conclusion

By following this guide, you can efficiently compute influence functions
on large datasets using both sequential and parallel methods.
