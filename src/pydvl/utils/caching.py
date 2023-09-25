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
from functools import wraps
from hashlib import blake2b
from io import BytesIO
from time import time
from typing import Any, Callable, Dict, Iterable, Optional, TypeVar, cast

from cloudpickle import Pickler
from pymemcache import MemcacheUnexpectedCloseError
from pymemcache.client import Client, RetryingClient

from .config import MemcachedClientConfig
from .numeric import running_moments

PICKLE_VERSION = 5  # python >= 3.8

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheStats:
    """Statistics gathered by cached functions.

    Attributes:
        sets: number of times a value was set in the cache
        misses: number of times a value was not found in the cache
        hits: number of times a value was found in the cache
        timeouts: number of times a timeout occurred
        errors: number of times an error occurred
        reconnects: number of times the client reconnected to the server
    """

    sets: int = 0
    misses: int = 0
    hits: int = 0
    timeouts: int = 0
    errors: int = 0
    reconnects: int = 0


def serialize(x: Any) -> bytes:
    """Serialize an object to bytes.
    Args:
        x: object to serialize.

    Returns:
        serialized object.
    """
    pickled_output = BytesIO()
    pickler = Pickler(pickled_output, PICKLE_VERSION)
    pickler.dump(x)
    return pickled_output.getvalue()


def memcached(
    client_config: Optional[MemcachedClientConfig] = None,
    time_threshold: float = 0.3,
    allow_repeated_evaluations: bool = False,
    rtol_stderr: float = 0.1,
    min_repetitions: int = 3,
    ignore_args: Optional[Iterable[str]] = None,
) -> Callable[[Callable[..., T], bytes | None], Callable[..., T]]:
    """
    Transparent, distributed memoization of function calls.

    Given a function and its signature, memcached uses a distributed cache
    that, for each set of inputs, keeps track of the average returned value,
    with variance and number of times it was calculated.

    If the function is deterministic, i.e. same input corresponds to the same
    exact output, set `allow_repeated_evaluations` to `False`. If instead the
    function is stochastic (like the training of a model depending on random
    initializations), memcached() allows to set a minimum number of evaluations
    to compute a running average, and a tolerance after which the function will
    not be called anymore. In other words, the function will be recomputed
    until the value has stabilized with a standard error smaller than
    `rtol_stderr * running average`.

    !!! Warning
        Do not cache functions with state! See [Cache reuse](cache-reuse)

    ??? Example
        ```python
        cached_fun = memcached(**asdict(cache_options))(heavy_computation)
        ```

    Args:
        client_config: configuration for pymemcache's
            [Client][pymemcache.client.base.Client].
            Will be merged on top of the default configuration (see below).
        time_threshold: computations taking less time than this many seconds are
            not cached.
        allow_repeated_evaluations: If `True`, repeated calls to a function
            with the same arguments will be allowed and outputs averaged until the
            running standard deviation of the mean stabilizes below
            `rtol_stderr * mean`.
        rtol_stderr: relative tolerance for repeated evaluations. More precisely,
            [memcached()][pydvl.utils.caching.memcached] will stop evaluating the function once the
            standard deviation of the mean is smaller than `rtol_stderr * mean`.
        min_repetitions: minimum number of times that a function evaluation
            on the same arguments is repeated before returning cached values. Useful
            for stochastic functions only. If the model training is very noisy, set
            this number to higher values to reduce variance.
        ignore_args: Do not take these keyword arguments into account when
            hashing the wrapped function for usage as key in memcached. This allows
            sharing the cache among different jobs for the same experiment run if
            the callable happens to have "nuisance" parameters like `job_id` which
            do not affect the result of the computation.

    Returns:
        A wrapped function

    """
    if ignore_args is None:
        ignore_args = []

    # Do I really need this?
    def connect(config: MemcachedClientConfig):
        """First tries to establish a connection, then tries setting and
        getting a value."""
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
            raise e
        except AssertionError as e:
            logger.error(  # type: ignore
                f"@memcached: Failure saving dummy value "
                f"to {config.server}: {str(e)}"
            )

    def wrapper(fun: Callable[..., T], signature: Optional[bytes] = None):
        if signature is None:
            signature = serialize((fun.__code__.co_code, fun.__code__.co_consts))

        @wraps(fun, updated=[])  # don't try to use update() for a class
        class Wrapped:
            config: MemcachedClientConfig
            stats: CacheStats
            client: RetryingClient

            def __init__(self, config: MemcachedClientConfig):
                self.config = config
                self.stats = CacheStats()
                self.client = connect(self.config)
                self._signature = signature

            def __call__(self, *args, **kwargs) -> T:
                key_kwargs = {k: v for k, v in kwargs.items() if k not in ignore_args}  # type: ignore
                arg_signature: bytes = serialize((args, list(key_kwargs.items())))

                key = blake2b(self._signature + arg_signature).hexdigest().encode("ASCII")  # type: ignore

                result_dict: Dict[str, float] = self.get_key_value(key)
                if result_dict is None:
                    result_dict = {}
                    start = time()
                    value = fun(*args, **kwargs)
                    end = time()
                    result_dict["value"] = value
                    if end - start >= time_threshold or allow_repeated_evaluations:
                        result_dict["count"] = 1
                        result_dict["variance"] = 0
                        self.client.set(key, result_dict, noreply=True)
                        self.stats.sets += 1
                    self.stats.misses += 1
                elif allow_repeated_evaluations:
                    self.stats.hits += 1
                    value = result_dict["value"]
                    count = result_dict["count"]
                    variance = result_dict["variance"]
                    error_on_average = (variance / count) ** (1 / 2)
                    if (
                        error_on_average > rtol_stderr * value
                        or count <= min_repetitions
                    ):
                        new_value = fun(*args, **kwargs)
                        new_avg, new_var = running_moments(
                            value, variance, int(count), cast(float, new_value)
                        )
                        result_dict["value"] = new_avg
                        result_dict["count"] = count + 1
                        result_dict["variance"] = new_var
                        self.client.set(key, result_dict, noreply=True)
                        self.stats.sets += 1
                else:
                    self.stats.hits += 1
                return result_dict["value"]  # type: ignore

            def __getstate__(self):
                """Enables pickling after a socket has been opened to the
                memcached server, by removing the client from the stored
                data."""
                odict = self.__dict__.copy()
                del odict["client"]
                return odict

            def __setstate__(self, d: dict):
                """Restores a client connection after loading from a pickle."""
                self.config = d["config"]
                self.stats = d["stats"]
                self.client = Client(**asdict(self.config))
                self._signature = signature

            def get_key_value(self, key: bytes):
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
                    self.client = connect(self.config)
                return result

        Wrapped.__doc__ = (
            f"A wrapper around {fun.__name__}() with remote caching enabled.\n"
            + (Wrapped.__doc__ or "")
        )
        Wrapped.__name__ = f"memcached_{fun.__name__}"
        path = list(reversed(fun.__qualname__.split(".")))
        patched = [f"memcached_{path[0]}"] + path[1:]
        Wrapped.__qualname__ = ".".join(reversed(patched))

        # TODO: pick from some config file or something
        return Wrapped(client_config or MemcachedClientConfig())

    return wrapper
