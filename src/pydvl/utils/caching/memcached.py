import logging
import socket
import uuid
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

try:
    from pymemcache import MemcacheUnexpectedCloseError
    from pymemcache.client import Client, RetryingClient
    from pymemcache.serde import PickleSerde
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"Cannot use MemcachedCacheBackend because pymemcache was not installed. "
        f"Make sure to install pyDVL using `pip install pyDVL[memcached]`. \n"
        f"Original error: {e}"
    )


from .base import CacheBackend

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
        timeout: Duration in seconds to wait for send or recv calls
            on the socket connected to memcached.
        no_delay: If True, set the `TCP_NODELAY` flag, which may help
            with performance in some cases.
        serde: Serializer / Deserializer ("serde"). The default `PickleSerde`
            should work in most cases. See [pymemcache.client.base.Client][]
            for details.
    """

    server: Tuple[str, int] = ("localhost", 11211)
    connect_timeout: float = 1.0
    timeout: float = 1.0
    no_delay: bool = True
    serde: PickleSerde = PickleSerde(pickle_version=PICKLE_VERSION)


class MemcachedCacheBackend(CacheBackend):
    """Memcached cache backend for the distributed caching of functions.

    Implements the [CacheBackend][pydvl.utils.caching.base.CacheBackend]
    interface for a memcached based cache. This allows sharing evaluations
    across processes and nodes in a cluster. You can run memcached as a service,
    locally or remotely, see [the caching documentation][getting-started-cache].

    Args:
        config: Memcached client configuration.

    Attributes:
        config: Memcached client configuration.
        client: Memcached client instance.

    Example:
        Basic usage:
        ```pycon
        >>> from pydvl.utils.caching.memcached import MemcachedCacheBackend
        >>> cache_backend = MemcachedCacheBackend()
        >>> cache_backend.clear()
        >>> value = 42
        >>> cache_backend.set("key", value)
        >>> cache_backend.get("key")
        42
        ```

        Callable wrapping:
        ```pycon
        >>> from pydvl.utils.caching.memcached import MemcachedCacheBackend
        >>> cache_backend = MemcachedCacheBackend()
        >>> cache_backend.clear()
        >>> value = 42
        >>> def foo(x: int):
        ...     return x + 1
        ...
        >>> wrapped_foo = cache_backend.wrap(foo)
        >>> wrapped_foo(value)
        43
        >>> wrapped_foo.stats.misses
        1
        >>> wrapped_foo.stats.hits
        0
        >>> wrapped_foo(value)
        43
        >>> wrapped_foo.stats.misses
        1
        >>> wrapped_foo.stats.hits
        1
        ```
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
                f"@memcached: Failure saving dummy value to {config.server}: {str(e)}"
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
