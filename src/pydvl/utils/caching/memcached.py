""" Distributed caching of functions.

pyDVL uses [memcached](https://memcached.org) to cache utility values, through
[pymemcache](https://pypi.org/project/pymemcache). This allows sharing
evaluations across processes and nodes in a cluster. You can run memcached as a
service, locally or remotely, see [Setting up the cache](#setting-up-the-cache)

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
from __future__ import annotations

import logging
import socket
import uuid
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

from pymemcache import MemcacheUnexpectedCloseError
from pymemcache.client import Client, RetryingClient
from pymemcache.serde import PickleSerde

from .base import CacheBackendBase

__all__ = ["MemcachedClientConfig", "MemcachedCacheBackend"]

PICKLE_VERSION = 5  # python >= 3.8

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemcachedClientConfig:
    """Configuration of the memcached client.

    Args:
        server: A tuple of (IP|domain name, port).
        connect_timeout: How many seconds to wait before raising
            `ConnectionRefusedError` on failure to connect.
        timeout: seconds to wait for send or recv calls on the socket
            connected to memcached.
        no_delay: set the `TCP_NODELAY` flag, which may help with performance
            in some cases.
        serde: a serializer / deserializer ("serde"). The default `PickleSerde`
            should work in most cases. See [pymemcached's
            documentation](https://pymemcache.readthedocs.io/en/latest/apidoc/pymemcache.client.base.html#pymemcache.client.base.Client)
            for details.
    """

    server: Tuple[str, int] = ("localhost", 11211)
    connect_timeout: float = 1.0
    timeout: float = 1.0
    no_delay: bool = True
    serde: PickleSerde = PickleSerde(pickle_version=PICKLE_VERSION)


class MemcachedCacheBackend(CacheBackendBase):
    """Memcached cache backend.

    Implements CacheBackendBase using a memcached client.

    Attributes:
        config: Memcached client configuration.
        client: Memcached client instance.
    """

    def __init__(self, config: MemcachedClientConfig = MemcachedClientConfig()) -> None:
        """Initialize memcached cache backend.

        Args:
            config: Memcached client configuration.
        """
        super().__init__()
        self.config = config
        self.client = self._connect(self.config)

    def get(self, key: str) -> Optional[Any]:
        """Get value from memcached.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found or client disconnected.
        """
        result = None
        try:
            result = self.client.get(key)
        except socket.timeout as e:
            self.stats.timeouts += 1
            warnings.warn(f"{type(self).__name__}: {str(e)}", RuntimeWarning)
        except OSError as e:
            self.stats.errors += 1
            warnings.warn(f"{type(self).__name__}: {str(e)}", RuntimeWarning)
        except AttributeError as e:
            # FIXME: this depends on _recv() failing on invalid sockets
            # See pymemcache.base.py,
            self.stats.reconnects += 1
            warnings.warn(f"{type(self).__name__}: {str(e)}", RuntimeWarning)
            self.client = self._connect(self.config)
        if result is None:
            self.stats.misses += 1
        else:
            self.stats.hits += 1
        return result

    def set(self, key: str, value: Any) -> None:
        """Set value in memcached.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        self.client.set(key, value, noreply=True)
        self.stats.sets += 1

    def clear(self) -> None:
        """Flush all values from memcached."""
        self.client.flush_all(noreply=True)

    @staticmethod
    def _connect(config: MemcachedClientConfig) -> RetryingClient:
        """Connect to memcached server."""
        try:
            client = RetryingClient(
                Client(**asdict(config)),
                attempts=3,
                retry_delay=0.1,
                retry_for=[MemcacheUnexpectedCloseError],
            )

            temp_key = str(uuid.uuid4())
            client.set(temp_key, 7)
            assert client.get(temp_key) == 7
            client.delete(temp_key, 0)
            return client
        except ConnectionRefusedError as e:
            logger.error(  # type: ignore
                f"@memcached: Timeout connecting "
                f"to {config.server} after "
                f"{config.connect_timeout} seconds: {str(e)}. Did you start memcached?"
            )
            raise
        except AssertionError as e:
            logger.error(  # type: ignore
                f"@memcached: Failure saving dummy value "
                f"to {config.server}: {str(e)}"
            )
            raise

    def combine_hashes(self, *args: str) -> str:
        """Join cache key components for Memcached."""
        return ":".join(args)

    def __getstate__(self) -> Dict:
        """Enables pickling after a socket has been opened to the
        memcached server, by removing the client from the stored
        data."""
        odict = self.__dict__.copy()
        del odict["client"]
        return odict

    def __setstate__(self, d: Dict):
        """Restores a client connection after loading from a pickle."""
        self.config = d["config"]
        self.stats = d["stats"]
        self.client = self._connect(self.config)
