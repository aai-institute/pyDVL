""" Distributed caching of functions, using memcached.

pyDVL uses [Memcached](https://memcached.org/) to cache utility values.
You can run it either locally or, using [Docker](https://www.docker.com/):

```shell
docker container run --rm -p 11211:11211 --name pydvl-cache -d memcached:latest
```

Caching is enabled by default but can be disabled if not needed or desired. It
happens transparently within :class:`pydvl.utils.utility.Utility`. Just pass any
configuration options when constructing it, using `
"""

import logging
import socket
import uuid
import warnings
from dataclasses import dataclass
from functools import wraps
from hashlib import blake2b
from io import BytesIO
from time import time
from typing import Callable, Dict, Iterable, Optional

from cloudpickle import Pickler
from pymemcache import MemcacheUnexpectedCloseError
from pymemcache.client import Client, RetryingClient

from .config import MemcachedClientConfig
from .numeric import get_running_avg_variance

PICKLE_VERSION = 5  # python >= 3.8

logger = logging.getLogger(__name__)


@dataclass
class CacheInfo:
    sets: int = 0
    misses: int = 0
    hits: int = 0
    timeouts: int = 0
    errors: int = 0
    reconnects: int = 0


def serialize(x):
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
):
    """Transparent, distributed memoization of function calls.

    Given a function and its signature, memcached creates a distributed cache
    that, for each set of inputs, keeps track of the average returned value,
    with variance and number of times it was calculated.

    If the function is deterministic, i.e. same input corresponds to the same
    exact output, set `allow_repeated_evaluations` to `False`. If instead the
    function is stochastic (like the training of a model depending on random
    initializations), memcached allows to set a minimum number of evaluations
    to compute a running average, and a tolerance after which the function will
    not be called anymore. In other words, the function will be recomputed
    until the value has stabilized with a standard error smaller than
    `rtol_stderr * running average`.

    :param client_config: configuration for
        `pymemcache's Client()
        <https://pymemcache.readthedocs.io/en/stable/apidoc/pymemcache.client
        .base.html>`_.
        Will be merged on top of the default configuration, which is:

        ```
        default_config = dict(
            server=('localhost', 11211),
            connect_timeout=1.0,
            timeout=0.1,
            # IMPORTANT! Disable small packet consolidation:
            no_delay=True,
            serde=serde.PickleSerde(pickle_version=PICKLE_VERSION)
        )
        ```
    :param time_threshold: computations taking less time than this many seconds
        are not cached.
    :param allow_repeated_evaluations: If `True`, repeated calls to a function
        with the same arguments will be allowed and outputs averaged until the
        running standard deviation of the mean stabilises below
        `rtol_stderr * mean`.
    :param rtol_stderr: relative tolerance for repeated evaluations. More
        precisely, :func:`memcached` will stop evaluating the function once the
        standard deviation of the mean is smaller than `rtol_stderr * mean`.
    :param min_repetitions: minimum number of times that a function evaluation
        on the same arguments is repeated before returning cached values. Useful
        for stochastic functions only. If the model training is very noisy, set
        this number to higher values to reduce variance.
    :param ignore_args: Do not take these keyword arguments into account when
        hashing the wrapped function for usage as key in memcached

    :return: A wrapped function

    """
    if ignore_args is None:
        ignore_args = []

    # Do I really need this?
    def connect(config: MemcachedClientConfig):
        """First tries to establish a connection, then tries setting and
        getting a value."""
        try:
            test_config: Dict = dict(**config)
            client = RetryingClient(
                Client(**test_config),
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

    def wrapper(fun: Callable[..., float], signature: Optional[bytes] = None):
        if signature is None:
            signature = serialize((fun.__code__.co_code, fun.__code__.co_consts))

        @wraps(fun, updated=[])  # don't try to use update() for a class
        class Wrapped:
            def __init__(self, config: MemcachedClientConfig):
                self.config = config
                self.cache_info = CacheInfo()
                self.client = connect(self.config)
                self._signature = signature

            def __call__(self, *args, **kwargs) -> float:
                key_kwargs = {k: v for k, v in kwargs.items() if k not in ignore_args}  # type: ignore
                arg_signature: bytes = serialize((args, list(key_kwargs.items())))

                key = blake2b(self._signature + arg_signature).hexdigest().encode("ASCII")  # type: ignore

                result_dict: Dict = self.get_key_value(key)
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
                        self.cache_info.sets += 1
                    self.cache_info.misses += 1
                elif allow_repeated_evaluations:
                    self.cache_info.hits += 1
                    value = result_dict["value"]
                    count = result_dict["count"]
                    variance = result_dict["variance"]
                    error_on_average = (variance / count) ** (1 / 2)
                    if (
                        error_on_average > rtol_stderr * value
                        or count <= min_repetitions
                    ):
                        new_value = fun(*args, **kwargs)
                        new_avg, new_var = get_running_avg_variance(
                            value, variance, new_value, count
                        )
                        result_dict["value"] = new_avg
                        result_dict["count"] = count + 1
                        result_dict["variance"] = new_var
                        self.client.set(key, result_dict, noreply=True)
                        self.cache_info.sets += 1
                else:
                    self.cache_info.hits += 1
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
                self.cache_info = d["cache_info"]
                self.client = Client(**self.config)
                self._signature = signature

            def get_key_value(self, key: bytes):
                result = None
                try:
                    result = self.client.get(key)
                except socket.timeout as e:
                    self.cache_info.timeouts += 1
                    warnings.warn(f"{type(self).__name__}: {str(e)}", RuntimeWarning)
                except OSError as e:
                    self.cache_info.errors += 1
                    warnings.warn(f"{type(self).__name__}: {str(e)}", RuntimeWarning)
                except AttributeError as e:
                    # FIXME: this depends on _recv() failing on invalid sockets
                    # See pymemcache.base.py,
                    self.cache_info.reconnects += 1
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
        config = MemcachedClientConfig()
        if client_config is not None:
            config.update(client_config)  # type: ignore
        return Wrapped(config)

    return wrapper
