"""This module provides caching of functions.

PyDVL can cache (memoize) the computation of the utility function
and speed up some computations for data valuation.

!!! Warning
    Function evaluations are cached with a key based on the function's signature
    and code. This can lead to undesired cache hits, see [Cache reuse](#cache-reuse).

    Remember **not to reuse utility objects for different datasets**.

# Configuration

Caching is disabled by default but can be enabled easily,
see [Setting up the cache][getting-started-cache].
When enabled, it will be added to any callable used to construct a
[Utility][pydvl.utils.utility.Utility] (done with the wrap method of
[CacheBackend][pydvl.utils.caching.base.CacheBackend]).
Depending on the nature of the utility you might want to
enable the computation of a running average of function values, see
[Usage with stochastic functions](#usaage-with-stochastic-functions).
You can see all configuration options under
[CachedFuncConfig][pydvl.utils.caching.config.CachedFuncConfig].

# Supported Backends

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

    !!! Info
        This specific backend requires optional dependencies not installed by
        default. See [Extra dependencies][installation-extras] for more
        information.

# Usage with stochastic functions

In addition to standard memoization, the wrapped functions
can compute running average and standard error of repeated evaluations
for the same input. This can be useful for stochastic functions with high variance
(e.g. model training for small sample sizes), but drastically reduces
the speed benefits of memoization.

This behaviour can be activated with the option
[allow_repeated_evaluations][pydvl.utils.caching.config.CachedFuncConfig].

# Cache reuse

When working directly with [CachedFunc][pydvl.utils.caching.base.CachedFunc],  it is
essential to only cache pure functions. If they have any kind of state, either
internal or external (e.g. a closure over some data that may change), then the
cache will fail to notice this and the same value will be returned.

When a function is wrapped with [CachedFunc][pydvl.utils.caching.base.CachedFunc]
for memoization, its signature (input and output names) and code are used as a key
for the cache.

If you are running experiments with the same [Utility][pydvl.utils.utility.Utility]
but different datasets, this will lead to evaluations of the utility on new data
returning old values because utilities only use sample indices as arguments (so
there is no way to tell the difference between '1' for dataset A and '1' for
dataset 2 from the point of view of the cache). One solution is to empty the
cache between runs by calling the `clear` method of the cache backend instance,
but the preferred one is to **use a different Utility object for each dataset**.

# Unexpected cache misses

Because all arguments to a function are used as part of the key for the cache,
sometimes one must exclude some of them. For example, If a function is going to
run across multiple processes and some reporting arguments are added (like a
`job_id` for logging purposes), these will be part of the signature and make the
functions distinct to the eyes of the cache. This can be avoided with the use of
[ignore_args][pydvl.utils.caching.config.CachedFuncConfig] option in the configuration.

"""

from .base import *
from .config import *
from .disk import *
from .memory import *

try:
    from .memcached import *
except ModuleNotFoundError:
    pass
