---
title: How-to speed up data valuation with caching
alias:
    name: speed-up-value-with-caching
---

Speeding up data valuation is crucial for handling complex and large datasets
efficiently. This guide explains how to leverage caching in pyDVL
to improve performance of data valuation algorithms.

While caching is disabled by default in pyDVL, it can be enabled
to speed up repeated computations.

Caching (memoization) can significantly reduce computation time by storing
the results of expensive function calls and reusing them when the same inputs
occur again. pyDVL supports three different caching backends:

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

Please refer to the documentation and examples of each backend class
for more details.

!!! tip "Other guides"

    For an alternative way to speed up data valuation,
    see [[speed-up-value-with-parallel]].

!!! tip "Using the cache"
    
    Continue reading about the cache in the documentation
    for the [caching package][pydvl.utils.caching].

## When to Use Caching

In general, caching won't play a major role in the computation of data values
because the probability of sampling the same subset twice, and hence needing
the same utility function computation, is very low. However, it can be very
useful for:

- Comparing methods using the same utility function.
- Running multiple experiments with the same data.


!!! warning "Reusing utility with caching"

    Care must be taken when reusing the same utility function with different data,
    see the documentation for the [caching package][pydvl.utils.caching] for more
    information.

## Example: Disk-Based Caching

To enable disk-based caching using:

```python
from pydvl.utils.caching.disk import DiskCacheBackend
from pydvl.valuation.utility import ModelUtility

cache_backend = DiskCacheBackend()
utility = ModelUtility(..., cache_backend=cache_backend)
```

## Example: Memcached-Based Caching

To enable memcached-based caching, you have to first install and run it:

- Follow the [Getting Started guide](https://github.com/memcached/memcached/wiki#getting-started)
  to install it locally. Then you can run it with:

  ```shell
  memcached -u user
  ```

- Or run it locally in a Docker container:
  
  ```shell
  docker container run -d --rm -p 11211:11211 memcached:latest
  ```
  
  This will run Memcached inside a container in daemon mode.

- Or run it remotely and get its address and port number.

If we assume that we chose the 2nd option and have memcached running
in a docker container locally, then we can enable memcached-based caching using:

```python
from pydvl.utils.caching.memcached import MemcachedCacheBackend, MemcachedClientConfig
from pydvl.valuation.utility import ModelUtility

cache_config = MemcachedClientConfig()
cache_backend = MemcachedCacheBackend(cache_config)
utility = ModelUtility(..., cache_backend=cache_backend)

```
