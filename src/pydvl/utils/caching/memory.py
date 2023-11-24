import os
from typing import Any, Dict, Optional

from pydvl.utils.caching.base import CacheBackend

__all__ = ["InMemoryCacheBackend"]


class InMemoryCacheBackend(CacheBackend):
    """In-memory cache backend that stores results in a dictionary.

    Implements the CacheBackend interface for an in-memory-based cache.
    Stores cache entries as values in a dictionary, keyed by cache key.
    This allows sharing evaluations across threads in a single process.

    The implementation is not thread-safe.

    Attributes:
        cached_values: Dictionary used to store cached values.

    ??? Examples
        ``` pycon
        >>> from pydvl.utils.caching.memory import InMemoryCacheBackend
        >>> cache_backend = InMemoryCacheBackend()
        >>> cache_backend.clear()
        >>> value = 42
        >>> cache_backend.set("key", value)
        >>> cache_backend.get("key")
        42
        ```

        ``` pycon
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

    def __init__(self) -> None:
        """Initialize the in-memory cache backend."""
        super().__init__()
        self.cached_values: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found.
        """
        value = self.cached_values.get(key, None)
        if value is not None:
            self.stats.hits += 1
        else:
            self.stats.misses += 1
        return value

    def set(self, key: str, value: Any) -> None:
        """Set a value in the cache.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        self.cached_values[key] = value
        self.stats.sets += 1

    def clear(self) -> None:
        """Deletes cache dictionary and recreates it."""
        del self.cached_values
        self.cached_values = {}

    def combine_hashes(self, *args: str) -> str:
        """Join cache key components."""
        return os.pathsep.join(args)
