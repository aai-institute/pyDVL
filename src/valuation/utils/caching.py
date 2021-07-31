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


def memcached(server: str = 'localhost:11211',
              threshold: float = 0,
              ignore_args: Iterable[str] = None):
    """ Decorate a callable with this in order to have transparent caching.

    The function's code, constants and all arguments (except for those in \
    `ignore_args` are used to generate the key for the remote cache.

    :param server: server and port or unix socket
    :param threshold: computations taking below this value (in seconds) are not
        cached
    :param ignore_args: Do not take these keyword arguments into account when
        hashing the wrapped function for usage as key in memcached
    :return: A wrapped function
    """
    if ignore_args is None:
        ignore_args = []

    def wrapper(fun: Callable):
        cache = Client(server=server, connect_timeout=1.0, timeout=0.1,
                       serde=serde.PickleSerde(pickle_version=PICKLE_VERSION))
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
            # FIXME: I need to create the spooky_64 object here because it
            #  can't be pickled
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
                    cache.set(key, result)  # , noreply=True)
            return result
        return wrapped

    return wrapper
