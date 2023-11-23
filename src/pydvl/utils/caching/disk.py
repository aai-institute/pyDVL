import os
import shutil
from pathlib import Path
from typing import Any, Optional, Union

import cloudpickle

from pydvl.utils.caching.base import CacheBackend

__all__ = ["DiskCacheBackend"]

PICKLE_VERSION = 5  # python >= 3.8

DEFAULT_CACHE_DIR = Path().home() / ".pydvl_cache/disk"


class DiskCacheBackend(CacheBackend):
    """Disk cache backend that stores results in files.

    Implements the CacheBackend interface for a disk-based cache.
    Stores cache entries as pickled files on disk, keyed by cache key.
    This allows sharing evaluations across processes in a single node/computer.

    Args:
        cache_dir: Base directory for cache storage.

    Attributes:
        cache_dir: Base directory for cache storage.

    ??? Examples
        ``` pycon
        >>> from pydvl.utils.caching.disk import DiskCacheBackend
        >>> cache = DiskCacheBackend(cache_dir="/tmp/pydvl_disk_cache")
        >>> cache.clear()
        >>> value = 42
        >>> cache.set("key", value)
        >>> cache.get("key")
        42
        ```

        ``` pycon
        >>> from pydvl.utils.caching.disk import DiskCacheBackend
        >>> cache = DiskCacheBackend(cache_dir="/tmp/pydvl_disk_cache")
        >>> cache.clear()
        >>> value = 42
        >>> def foo(x: int):
        ...     return x + 1
        ...
        >>> wrapped_foo = cache.wrap(foo)
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

    def __init__(
        self,
        cache_dir: Union[os.PathLike, str] = DEFAULT_CACHE_DIR,
    ) -> None:
        """Initialize the disk cache backend.

        Args:
            cache_dir: Base directory for cache storage.
                By default, this is set to `~/.pydvl_cache/disk`
        """
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found.
        """
        cache_file = self.cache_dir / key
        if not cache_file.exists():
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        with cache_file.open("rb") as f:
            return cloudpickle.load(f)

    def set(self, key: str, value: Any) -> None:
        """Set a value in the cache.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        cache_file = self.cache_dir / key
        self.stats.sets += 1
        with cache_file.open("wb") as f:
            cloudpickle.dump(value, f, protocol=PICKLE_VERSION)

    def clear(self) -> None:
        """Deletes cache directory and recreates it."""
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def combine_hashes(self, *args: str) -> str:
        """Join cache key components."""
        return os.pathsep.join(args)
