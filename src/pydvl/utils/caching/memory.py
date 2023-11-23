import os
from typing import Any, Optional

from pydvl.utils.caching.base import CacheBackendBase

__all__ = ["InMemoryCacheBackend"]


class InMemoryCacheBackend(CacheBackendBase):
    """In-memory cache backend that stores results in a dictionary.

    Implements the CacheBackendBase interface for an in-memory-based cache.
    Stores cache entries as values in a dictionary, keyed by cache key.
    """

    def __init__(self) -> None:
        """Initialize the in-memory cache backend."""
        super().__init__()
        self.cached_values = {}

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
