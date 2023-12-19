---
title: Getting Started
alias: 
  name: getting-started
  text: Getting Started
---

# Getting started

!!! Warning
    Make sure you have read [[installation]] before using the library. 
    In particular read about which extra dependencies you may need.

## Main concepts

pyDVL aims to be a repository of production-ready, reference implementations of
algorithms for data valuation and influence functions. Even though we only
briefly introduce key concepts in the documentation, the following sections 
should be enough to get you started.

* [[data-valuation]] for key objects and usage patterns for Shapley value
  computation and related methods.
* [[influence-values]] for instructions on how to compute influence functions.

## Running the examples

If you are somewhat familiar with the concepts of data valuation, you can start
by browsing our worked-out examples illustrating pyDVL's capabilities either:

- In the examples under [[data-valuation]] and [[influence-values]].
- Using [binder](https://mybinder.org/) notebooks, deployed from each
  example's page.
- Locally, by starting a jupyter server at the root of the project. You will
  have to install jupyter first manually since it's not a dependency of the
  library.

## Advanced usage

Besides the dos and don'ts of data valuation itself, which are the subject of
the examples and the documentation of each method, there are two main things to
keep in mind when using pyDVL.

### Caching

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

  **Note** This specific backend requires optional dependencies.
  See [[installation#extras]] for more information)

!!! tip "When is the cache really necessary?"
    Crucially, semi-value computations with the
    [PermutationSampler][pydvl.value.sampler.PermutationSampler] require caching
    to be enabled, or they will take twice as long as the direct implementation
    in [compute_shapley_values][pydvl.value.shapley.compute_shapley_values].

!!! tip "Using the cache"
    Continue reading about the cache in the documentation
    for the [caching package][pydvl.utils.caching].

#### Setting up the Memcached cache

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

### Parallelization

pyDVL uses [joblib](https://joblib.readthedocs.io/en/latest/) for local
parallelization (within one machine) and supports using
[Ray](https://ray.io) for distributed parallelization (across multiple machines).

The former works out of the box but for the latter you will need to install
additional dependencies (see [[installation#extras]] )
and to provide a running cluster (or run ray in local mode).

As of v0.7.0 pyDVL does not allow requesting resources per task sent to the
cluster, so you will need to make sure that each worker has enough resources to
handle the tasks it receives. A data valuation task using game-theoretic methods
will typically make a copy of the whole model and dataset to each worker, even
if the re-training only happens on a subset of the data. This means that you
should make sure that each worker has enough memory to handle the whole dataset.

#### Ray

Please follow the instructions in Ray's documentation to set up a cluster.
Once you have a running cluster, you can use it by passing the address
of the head node to parallel methods via [ParallelConfig][pydvl.parallel.config.ParallelConfig].

For a local ray cluster you would use:

```python
from pydvl.parallel.config import ParallelConfig
config = ParallelConfig(backend="ray") 
```
