---
title: Advanced usage
alias: 
  name: advanced-usage
  text: Advanced usage
---

# Advanced usage

Besides the dos and don'ts of data valuation itself, which are the subject of
the examples and the documentation of each method, there are two main things to
keep in mind when using pyDVL namely Parallelization and Caching.

## Parallelization

pyDVL uses parallelization to scale and speed up computations. It does so
using one of Dask, Ray or Joblib. The first is used in
the [influence][pydvl.influence] package whereas the other two
are used in the [value][pydvl.value] package. 

### Data Valuation

For data valuation, pyDVL uses [joblib](https://joblib.readthedocs.io/en/latest/) for local
parallelization (within one machine) and supports using
[Ray](https://ray.io) for distributed parallelization (across multiple machines).

The former works out of the box but for the latter you will need to install
additional dependencies (see [[installation#extras|Extras]] )
and to provide a running cluster (or run ray in local mode).

!!! note

    As of v0.7.0 pyDVL does not allow requesting resources per task sent to the
    cluster, so you will need to make sure that each worker has enough resources to
    handle the tasks it receives. A data valuation task using game-theoretic methods
    will typically make a copy of the whole model and dataset to each worker, even
    if the re-training only happens on a subset of the data. This means that you
    should make sure that each worker has enough memory to handle the whole dataset.

#### Joblib

Please follow the instructions in Joblib's documentation
for all possible configuration options that you can pass to the
[parallel_config][joblib.parallel_config] context manager.

To use the joblib parallel backend with the `loky` backend and verbosity set to `100`
to compute exact shapley values you would use:

```python
import joblib
from pydvl.parallel import ParallelConfig
from pydvl.value.shapley import combinatorial_exact_shapley
from pydvl.utils.utility import Utility

config = ParallelConfig(backend="joblib") 
# Utility details are omitted here for the sake of brevity
u = Utility(...)

with joblib.parallel_config(backend="loky", verbose=100):
    combinatorial_exact_shapley(u, config=config)
```

#### Ray

Please follow the instructions in Ray's documentation to set up a cluster.
Once you have a running cluster, you can use it by passing the address

Before starting a computation, you should initialize ray by calling `ray.init()`
with the appropriate parameters:

For a local ray cluster you would use:

```python
import ray

ray.init()
```

Whereas for a remote ray cluster you would use:

```python
import ray

address = "<Hypothetical Ray Cluster IP Address>"
ray.init(address)
```

To use the ray parallel backend to compute exact shapley values you would use:

```python
import ray
from pydvl.parallel import ParallelConfig
from pydvl.value.shapley import combinatorial_exact_shapley
from pydvl.utils.utility import Utility

ray.init()
config = ParallelConfig(backend="ray")
# Utility details are omitted here for the sake of brevity
u = Utility(...)
combinatorial_exact_shapley(u, config=config)
```

### Influence Functions

#### Dask

For influence functions, pyDVL uses [Dask](https://docs.dask.org/en/stable/)
to scale out the computation.

```python
import torch
from pydvl.influence import DaskInfluenceCalculator
from pydvl.influence.torch import CgInfluence
from pydvl.influence.torch.util import (
    TorchNumpyConverter,
)
from distributed import Client

# We omit the details of CgInfluence for the sake of brevity
infl_model = CgInfluence(...)

client = Client(n_workers=4, threads_per_worker=1)
infl_calc = DaskInfluenceCalculator(
    infl_model,
    TorchNumpyConverter(device=torch.device("cpu")),
    client
)
# We omit the details of the computation for the sake of brevity
# da_influences is a dask.array.Array
da_influences = infl_calc.influences(...)

# trigger computation and write chunks to disk in parallel
da_influences.to_zarr("path/or/url")
```

Refer to the documentation of the [DaskInfluenceCalculator][pydvl.influence.influence_calculator.DaskInfluenceCalculator]
for more details.

## Caching

PyDVL can cache (memoize) the computation of the utility function
and speed up some computations for data valuation.
It is however disabled by default.
When it is enabled it takes into account the data indices passed as argument
and the utility function wrapped into the
[Utility][pydvl.utils.utility.Utility] object. This means that
care must be taken when reusing the same utility function with different data,
see the documentation for the [caching package][pydvl.utils.caching] for more
information.

In general, caching won't play a major role in the computation of Shapley values
because the probability of sampling the same subset twice, and hence needing
the same utility function computation, is very low. However, it can be very
useful when comparing methods that use the same utility function, or when
running multiple experiments with the same data.

pyDVL supports 3 different caching backends:

- [InMemoryCacheBackend][pydvl.utils.caching.memory.InMemoryCacheBackend]:
  an in-memory cache backend that uses a dictionary to store and retrieve
  cached values. This is used to share cached values between threads
  in a single process.

- [DiskCacheBackend][pydvl.utils.caching.disk.DiskCacheBackend]:
  a disk-based cache backend that uses pickled values written to and read from disk.  
  This is used to share cached values between processes in a single machine.
- [MemcachedCacheBackend][pydvl.utils.caching.memcached.MemcachedCacheBackend]:
  a [Memcached](https://memcached.org/)-based cache backend that uses pickled values written to
  and read from a Memcached server. This is used to share cached values
  between processes across multiple machines.

!!! note "Memcached extras"
  
    The Memcached backend requires optional dependencies.
    See [[installation#extras|Extras]] for more information.

As an example, here's how one would use the disk-based cached backend
with a utility:

```python
from pydvl.utils.caching.disk import DiskCacheBackend
from pydvl.utils.utility import Utility

cache_backend = DiskCacheBackend()
u = Utility(..., cache_backend=cache_backend)
```

Please refer to the documentation and examples of each backend class for more details.

!!! tip "When is the cache really necessary?"
    Crucially, semi-value computations with the
    [PermutationSampler][pydvl.value.sampler.PermutationSampler] require caching
    to be enabled, or they will take twice as long as the direct implementation
    in [compute_shapley_values][pydvl.value.shapley.compute_shapley_values].

!!! tip "Using the cache"
    Continue reading about the cache in the documentation
    for the [caching package][pydvl.utils.caching].

### Setting up the Memcached cache

[Memcached](https://memcached.org/) is an in-memory key-value store accessible
over the network. pyDVL can use it to cache the computation of the utility function
and speed up some computations (in particular, semi-value computations with the
[PermutationSampler][pydvl.value.sampler.PermutationSampler] but other methods
may benefit as well).

You can either install it as a package or run it inside a docker container (the
simplest). For installation instructions, refer to the [Getting
started](https://github.com/memcached/memcached/wiki#getting-started) section in
memcached's wiki. Then you can run it with:

```shell
memcached -u user
```

To run memcached inside a container in daemon mode instead, use:

```shell
docker container run -d --rm -p 11211:11211 memcached:latest
```
