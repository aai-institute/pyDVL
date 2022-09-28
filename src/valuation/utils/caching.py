"""
Distributed caching of functions, using memcached.
"""
import logging
import socket
import uuid
from dataclasses import dataclass
from functools import wraps
from hashlib import blake2b
from io import BytesIO
from time import time
from typing import Callable, Dict, Iterable, Optional

from cloudpickle import Pickler
from pymemcache import MemcacheUnexpectedCloseError
from pymemcache.client import Client, RetryingClient

from valuation.utils.numeric import get_running_avg_variance

from .config import MemcachedClientConfig

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
    cache_threshold: float = 0.3,
    allow_repeated_training: bool = False,
    rtol_threshold: float = 0.1,
    min_repetitions: int = 3,
    ignore_args: Optional[Iterable[str]] = None,
):
    """Wrap a callable with this in order to have transparent caching.
    Given a function and a signature, memcached creates a distributed cache
    that, for each set of inputs, keeps track of the average returned value,
    with variance and number of times it was calculated.
    If the function is deterministic, i.e. same input corresponds to the same
    exact output, set allow_repeated_training to False.
    If instead the function is noisy, memcache allows to set the minimum number
    of repetitions and the relative tolerance on the average output after which
    the cache will not be updated anymore. In other words, the function computation will
    be repeated until the average has stabilized.

    :param client_config: config for pymemcache.client.Client().
        Will be merged on top of the default configuration.
    :param cache_threshold: computations taking below this value (in seconds) are not
        cached
    :param allow_repeated_training: If True, models with same data are re-trained and
        results cached until rtol_threshold precision is reached
    :para rtol_threshold: relative tolerance for repeated training. More precisely,
        memcache will stop retraining the models once std/mean of the scores is smaller than
        the given rtol_threshold
    :param min_repetitions: minimum number of repetitions
    :param ignore_args: Do not take these keyword arguments into account when
        hashing the wrapped function for usage as key in memcached
    :return: A wrapped function

    The default configuration is::

        default_config = dict(
            server=('localhost', 11211),
            connect_timeout=1.0,
            timeout=0.1,
            # IMPORTANT! Disable small packet consolidation:
            no_delay=True,
            serde=serde.PickleSerde(pickle_version=PICKLE_VERSION)
        )
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
        except Exception as e:
            logger.error(  # type: ignore
                f"@memcached: Timeout connecting "
                f"to {config.server} after "
                f"{config.connect_timeout} seconds: {str(e)}"
            )
            raise e
        else:
            try:
                temp_key = str(uuid.uuid4())
                client.set(temp_key, 7)
                assert client.get(temp_key) == 7
                client.delete(temp_key, 0)
                return client
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
                    if end - start >= cache_threshold or allow_repeated_training:
                        result_dict["count"] = 1
                        result_dict["variance"] = 0
                        self.client.set(key, result_dict, noreply=True)
                        self.cache_info.sets += 1
                    self.cache_info.misses += 1
                elif allow_repeated_training:
                    self.cache_info.hits += 1
                    value = result_dict["value"]
                    count = result_dict["count"]
                    variance = result_dict["variance"]
                    error_on_average = (variance / count) ** (1 / 2)
                    if (
                        error_on_average > rtol_threshold * value
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
                memacached server, by removing the client from the stored data.
                """
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
                    logger.warning(f"{type(self).__name__}: {str(e)}")  # type: ignore
                except OSError as e:
                    self.cache_info.errors += 1
                    logger.warning(f"{type(self).__name__}: {str(e)}")  # type: ignore
                except AttributeError as e:
                    # FIXME: this depends on _recv() failing on invalid sockets
                    # See pymemcache.base.py,
                    self.cache_info.reconnects += 1
                    logger.warning(f"{type(self).__name__}: {str(e)}")  # type: ignore
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
