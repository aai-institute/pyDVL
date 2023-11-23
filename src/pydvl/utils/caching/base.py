import inspect
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Collection, Dict, Optional, Tuple, TypeVar, cast

from joblib import hashing
from joblib.func_inspect import filter_args

from ..numeric import running_moments
from .config import CachedFuncConfig

__all__ = ["CacheStats", "CacheBackend", "CachedFunc"]

T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Class used to store statistics gathered by cached functions.

    Attributes:
        sets: Number of times a value was set in the cache.
        misses: Number of times a value was not found in the cache.
        hits: Number of times a value was found in the cache.
        timeouts: Number of times a timeout occurred.
        errors: Number of times an error occurred.
        reconnects: Number of times the client reconnected to the server.
    """

    sets: int = 0
    misses: int = 0
    hits: int = 0
    timeouts: int = 0
    errors: int = 0
    reconnects: int = 0


@dataclass
class CacheResult:
    """A class used to store the cached result of a computation
    as well as count and variance when using repeated evaluation.

    Attributes:
        value: Cached value.
        count: Number of times this value has been computed.
        variance: Variance associated with the cached value.
    """

    value: float
    count: int = 1
    variance: float = 0.0


class CacheBackend(ABC):
    """Abstract base class for cache backends.

    Defines interface for cache access including wrapping callables,
    getting/setting results, clearing cache, and combining cache keys.

    Attributes:
        stats: Cache statistics tracker.
    """

    def __init__(self) -> None:
        self.stats = CacheStats()

    def wrap(
        self,
        func: Callable,
        *,
        cached_func_config: CachedFuncConfig = CachedFuncConfig(),
    ) -> "CachedFunc":
        """Wraps a function to cache its results.

        Args:
            func: The function to wrap.
            cached_func_config: Optional caching options for the wrapped function.

        Returns:
            The wrapped cached function.
        """
        return CachedFunc(
            func,
            cache_backend=self,
            cached_func_options=cached_func_config,
        )

    @abstractmethod
    def get(self, key: str) -> Optional[CacheResult]:
        """Abstract method to retrieve a cached result.

        Implemented by subclasses.

        Args:
            key: The cache key.

        Returns:
            The cached result or None if not found.
        """
        pass

    @abstractmethod
    def set(self, key: str, value: CacheResult) -> None:
        """Abstract method to set a cached result.

        Implemented by subclasses.

        Args:
            key: The cache key.
            value: The result to cache.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Abstract method to clear the entire cache."""
        pass

    @abstractmethod
    def combine_hashes(self, *args: str) -> str:
        """Abstract method to combine cache keys."""
        pass


class CachedFunc:
    """Caches callable function results with a provided cache backend.

    Wraps a callable function to cache its results using a provided
    an instance of a subclass of [CacheBackend][pydvl.utils.caching.base.CacheBackend].

    This class is heavily inspired from that of [joblib.memory.MemorizedFunc][].

    Args:
        func: Callable to wrap.
        cache_backend: Instance of CacheBackendBase that handles
            setting and getting values.
        cached_func_options: Configuration for wrapped function.
    """

    def __init__(
        self,
        func: Callable[..., T],
        *,
        cache_backend: CacheBackend,
        cached_func_options: CachedFuncConfig = CachedFuncConfig(),
    ) -> None:
        self.func = func
        self.cache_backend = cache_backend
        self.cached_func_options = cached_func_options

        self.__doc__ = f"A wrapper around {func.__name__}() with caching enabled.\n" + (
            CachedFunc.__doc__ or ""
        )
        self.__name__ = f"cached_{func.__name__}"
        path = list(reversed(func.__qualname__.split(".")))
        patched = [f"cached_{path[0]}"] + path[1:]
        self.__qualname__ = ".".join(reversed(patched))

    def __call__(self, *args, **kwargs) -> T:
        """Call the wrapped cached function.

        Executes the wrapped function, caching and returning the result.
        """
        return self._cached_call(args, kwargs)

    def _force_call(self, args, kwargs) -> Tuple[T, float]:
        """Force re-evaluation of the wrapped function.

        Executes the wrapped function without caching.

        Returns:
            Function result and execution duration.
        """
        start = time.monotonic()
        value = self.func(*args, **kwargs)
        end = time.monotonic()
        duration = end - start
        return value, duration

    def _cached_call(self, args, kwargs) -> T:
        """Cached wrapped function call.

        Executes the wrapped function with cache checking/setting.

        Returns:
            Cached result of the wrapped function.
        """
        key = self._get_cache_key(*args, **kwargs)
        cached_result = self.cache_backend.get(key)
        if cached_result is None:
            value, duration = self._force_call(args, kwargs)
            result = CacheResult(value)
            if (
                duration >= self.cached_func_options.time_threshold
                or self.cached_func_options.allow_repeated_evaluations
            ):
                self.cache_backend.set(key, result)
        else:
            result = cached_result
            if self.cached_func_options.allow_repeated_evaluations:
                error_on_average = (result.variance / result.count) ** (1 / 2)
                if (
                    error_on_average
                    > self.cached_func_options.rtol_stderr * result.value
                    or result.count <= self.cached_func_options.min_repetitions
                ):
                    new_value, _ = self._force_call(args, kwargs)
                    new_avg, new_var = running_moments(
                        result.value,
                        result.variance,
                        result.count,
                        cast(float, new_value),
                    )
                    result.value = new_avg
                    result.count += 1
                    result.variance = new_var
                    self.cache_backend.set(key, result)
        return result.value

    def _get_cache_key(self, *args, **kwargs) -> str:
        """Returns a string key used to identify the function and input parameter hash."""
        func_hash = self._hash_function(self.func)
        argument_hash = self._hash_arguments(
            self.func, self.cached_func_options.ignore_args, args, kwargs
        )
        key = self.cache_backend.combine_hashes(func_hash, argument_hash)
        return key

    @staticmethod
    def _hash_function(func: Callable) -> str:
        """Create hash for wrapped function."""
        func_hash = hashing.hash((func.__code__.co_code, func.__code__.co_consts))
        return func_hash

    @staticmethod
    def _hash_arguments(
        func: Callable,
        ignore_args: Collection[str],
        args: Tuple[Any],
        kwargs: Dict[str, Any],
    ) -> str:
        """Create hash for function arguments."""
        return hashing.hash(
            CachedFunc._filter_args(func, ignore_args, args, kwargs),
        )

    @staticmethod
    def _filter_args(
        func: Callable,
        ignore_args: Collection[str],
        args: Tuple[Any],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Filter arguments to exclude from cache keys."""
        # Remove kwargs before calling filter_args
        # Because some of them might not be explicitly in the function's signature
        # and that would raise an error when calling filter_args
        kwargs = {k: v for k, v in kwargs.items() if k not in ignore_args}  # type: ignore
        # Update ignore_args
        func_signature = inspect.signature(func)
        arg_names = []
        for param in func_signature.parameters.values():
            if param.kind in [
                param.POSITIONAL_ONLY,
                param.POSITIONAL_OR_KEYWORD,
                param.KEYWORD_ONLY,
            ]:
                arg_names.append(param.name)
        ignore_args = [x for x in ignore_args if x in arg_names]
        filtered_args: Dict[str, Any] = filter_args(func, ignore_args, args, kwargs)  # type: ignore
        # We ignore 'self' because for our use case we only care about the method.
        # We don't want a cache if another attribute changes in the instance.
        try:
            filtered_args.pop("self")
        except KeyError:
            pass
        return filtered_args  # type: ignore

    @property
    def stats(self) -> CacheStats:
        """Cache backend statistics."""
        return self.cache_backend.stats
