"""Caching of functions.

pyDVL caches (memoizes) utility values to allow reusing previously computed evaluations.

!!! Warning
    Function evaluations are cached with a key based on the function's signature
    and code. This can lead to undesired cache hits, see [Cache reuse](#cache-reuse).

    Remember **not to reuse utility objects for different datasets**.

# Configuration

Memoization is disabled by default but can be enabled easily,
see [Setting up the cache](#setting-up-the-cache).
When enabled, it will be added to any callable used to construct a
[Utility][pydvl.utils.utility.Utility] (done with the decorator [@memcached][pydvl.utils.caching.memcached]).
Depending on the nature of the utility you might want to
enable the computation of a running average of function values, see
[Usage with stochastic functions](#usaage-with-stochastic-functions).
You can see all configuration options under [MemcachedConfig][pydvl.utils.config.MemcachedConfig].

## Default configuration

```python
default_config = dict(
   server=('localhost', 11211),
   connect_timeout=1.0,
   timeout=0.1,
   # IMPORTANT! Disable small packet consolidation:
   no_delay=True,
   serde=serde.PickleSerde(pickle_version=PICKLE_VERSION)
)
```

# Supported Backends

- [InMemoryCacheBackend][]
- [DiskCacheBackend][]
- [MemcachedCacheBackend][]

# Usage with stochastic functions

In addition to standard memoization, the decorator
[memcached()][pydvl.utils.caching.memcached] can compute running average and
standard error of repeated evaluations for the same input. This can be useful
for stochastic functions with high variance (e.g. model training for small
sample sizes), but drastically reduces the speed benefits of memoization.

This behaviour can be activated with the argument `allow_repeated_evaluations`
to [memcached()][pydvl.utils.caching.memcached].

# Cache reuse

When working directly with [memcached()][pydvl.utils.caching.memcached],  it is
essential to only cache pure functions. If they have any kind of state, either
internal or external (e.g. a closure over some data that may change), then the
cache will fail to notice this and the same value will be returned.

When a function is wrapped with [memcached()][pydvl.utils.caching.memcached] for
memoization, its signature (input and output names) and code are used as a key
for the cache. Alternatively you can pass a custom value to be used as key with

```python
cached_fun = memcached(**asdict(cache_options))(fun, signature=custom_signature)
```

If you are running experiments with the same [Utility][pydvl.utils.utility.Utility]
but different datasets, this will lead to evaluations of the utility on new data
returning old values because utilities only use sample indices as arguments (so
there is no way to tell the difference between '1' for dataset A and '1' for
dataset 2 from the point of view of the cache). One solution is to empty the
cache between runs, but the preferred one is to **use a different Utility
object for each dataset**.

# Unexpected cache misses

Because all arguments to a function are used as part of the key for the cache,
sometimes one must exclude some of them. For example, If a function is going to
run across multiple processes and some reporting arguments are added (like a
`job_id` for logging purposes), these will be part of the signature and make the
functions distinct to the eyes of the cache. This can be avoided with the use of
[ignore_args][pydvl.utils.config.MemcachedConfig] in the configuration.

"""
from .base import *
from .config import *
from .disk import *
from .memcached import *
from .memory import *
