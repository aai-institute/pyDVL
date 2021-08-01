"""
Distributed caching of functions, using memcached.
TODO: wrap this to allow for different backends
"""
from functools import wraps
from dataclasses import dataclass, make_dataclass
from time import time
from typing import Callable, Iterable
from pymemcache.client import Client
from pymemcache.serde import PickleSerde
from pyhash import spooky_64
from cloudpickle import Pickler
from io import BytesIO


from valuation.utils.types import unpackable
from valuation.utils.logging import logger

PICKLE_VERSION = 5  # python >= 3.8


@unpackable
@dataclass
class ClientConfig:
    server: str = 'localhost:11211'
    connect_timeout: float = 1.0
    timeout: float = 1.0
    no_delay: bool = True
    serde: PickleSerde = PickleSerde(pickle_version=PICKLE_VERSION)


@unpackable
@dataclass
class MemcachedConfig:
    client = ClientConfig()
    threshold: float = 0.3
    ignore_args: Iterable[str] = None


def _serialize(x):
    pickled_output = BytesIO()
    pickler = Pickler(pickled_output, PICKLE_VERSION)
    pickler.dump(x)
    return pickled_output.getvalue()


def memcached(client_config: ClientConfig = None,
              threshold: float = 0.3,
              ignore_args: Iterable[str] = None):
    """ Decorate a callable with this in order to have transparent caching.

    The function's code, constants and all arguments (except for those in \
    `ignore_args` are used to generate the key for the remote cache.

    :param client_config: config for pymemcache.client.Client(). Will be merged
        on top of:

        default_config = dict(server='localhost:11211',
                              connect_timeout=1.0,
                               timeout=0.1,
                              # IMPORTANT! Disable small packet consolidation:
                              no_delay=True,
                              serde=serde.PickleSerde(
                                            pickle_version=PICKLE_VERSION))

    :param threshold: computations taking below this value (in seconds) are not
        cached
    :param ignore_args: Do not take these keyword arguments into account when
        hashing the wrapped function for usage as key in memcached
    :return: A wrapped function
    """
    if ignore_args is None:
        ignore_args = []

    # TODO: pick from some config file or something
    config = ClientConfig()

    if client_config is not None:
        config.update(client_config)
    try:
        test = Client(**config)
        test.set('dummy_key', 7)
        assert test.get('dummy_key') == 7
        test.delete('dummy_key', 0)
    except Exception as e:
        logger.error(f'@memcached: Timeout connecting '
                     f'to {config["server"]}')
        raise e

    def wrapper(fun: Callable):
        cache = Client(**config)
        # noinspection PyUnresolvedReferences
        signature: bytes = _serialize((fun.__code__.co_code,
                                       fun.__code__.co_consts))

        @wraps(fun)
        def wrapped(*args, **kwargs):
            key_kwargs = {k: v for k, v in kwargs.items() if k not in ignore_args}
            arg_signature: bytes = _serialize((args, list(key_kwargs.items())))

            # FIXME: do I really need to hash this?
            # FIXME: ensure that the hashing algorithm is portable
            # FIXME: determine right bit size
            # NB: I need to create the spooky_64 object here because it can't be
            #  pickled
            hasher = spooky_64(seed=0)
            key = str(hasher(signature + arg_signature)).encode('ASCII')
            result = cache.get(key)
            start = time()
            if result is None:
                result = fun(*args, **kwargs)
                end = time()
                # TODO: make the threshold adaptive
                if end - start >= threshold:
                    cache.set(key, result, noreply=True)
                    wrapped.cache_info.sets += 1
                wrapped.cache_info.misses += 1
            else:
                wrapped.cache_info.hits += 1
            return result

        wrapped.cache_info = \
            make_dataclass('CacheInfo', ['sets', 'misses', 'hits'])(0, 0, 0)
        return wrapped

    return wrapper
