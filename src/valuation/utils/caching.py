"""
Distributed caching of functions, using memcached.
TODO: wrap this to allow for different backends
"""
from functools import wraps
from time import time
from typing import Callable, Iterable
from pymemcache.client import Client
from pymemcache import serde
from pyhash import spooky_64
from pickle import Pickler
from io import BytesIO

from valuation.utils.logging import logger

PICKLE_VERSION = 5  # python >= 3.8


def _serialize(x):
    # I could use pymemcache's serializer, but it expects a key,val pair,
    # although the key is actually never used. Relying on this seems brittle and
    # I might be better off using pickle
    # serializer = serde.PickleSerde(pickle_version=PICKLE_VERSION)
    # return serializer.serialize(None, (args, list(kwargs.items())))
    pickled_output = BytesIO()
    pickler = Pickler(pickled_output, PICKLE_VERSION)
    pickler.dump(x)
    return pickled_output.getvalue()


def memcached(config: dict = None,
              threshold: float = 0,
              ignore_args: Iterable[str] = None):
    """ Decorate a callable with this in order to have transparent caching.

    The function's code, constants and all arguments (except for those in \
    `ignore_args` are used to generate the key for the remote cache.

    :param config: kwargs for pymemcache.client.Client(). Will be merged on top
    of:

        default_config = dict(server='localhost:11211',
                              connect_timeout=1.0, timeout=0.1,
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
    default_config = dict(server='localhost:11211',
                          connect_timeout=1.0, timeout=0.1,
                          # IMPORTANT! Disable small packet consolidation:
                          no_delay=True,
                          serde=serde.PickleSerde(pickle_version=PICKLE_VERSION))
    if config is not None:
        default_config.update(config)
    try:
        test = Client(**default_config)
        test.set('dummy_key', 0)
        test.delete('dummy_key', 0)
    except Exception as e:
        logger.error(f'@memcached: Timeout connecting '
                     f'to {default_config["server"]}')
        raise e

    def wrapper(fun: Callable):
        cache = Client(**default_config)
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
            logger.info(f"wrapped: {list(kwargs.items())}, key = {key}")
            result = cache.get(key)
            start = time()
            if result is None:
                result = fun(*args, **kwargs)
                end = time()
                # TODO: make the threshold adaptive
                if end - start >= threshold:
                    cache.set(key, result, noreply=True)
            return result
        return wrapped

    return wrapper
