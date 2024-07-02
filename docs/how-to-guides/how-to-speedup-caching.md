---
title: How-to speed up computations with caching
alias:
    name: speed-up-caching
    text: How-to speed up computations with caching
---

pyDVL can cache (memoize) the computation of the utility function
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

    ??? info "Memcached extras"

         The Memcached backend requires optional dependencies.
         See [Extras][installation-extras] for more information.

As an example, here's how one would use the disk-based cached backend
with a utility:

```python
from pydvl.utils.caching.disk import DiskCacheBackend
from pydvl.valuation.utility import ModelUtility

cache_backend = DiskCacheBackend()
utility = ModelUtility(..., cache_backend=cache_backend)
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

### Setting up the Memcached cache { #setting-up-memcached }

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
