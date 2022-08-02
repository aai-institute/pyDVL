"""
Distributed caching of functions, using memcached.
TODO: wrap this to allow for different backends
"""
import socket
from dataclasses import dataclass, field, make_dataclass
from functools import wraps
from hashlib import blake2b
from io import BytesIO
from time import time
from typing import Callable, Iterable

from cloudpickle import Pickler
from pymemcache import MemcacheUnexpectedCloseError
from pymemcache.client import Client, RetryingClient
from pymemcache.serde import PickleSerde

from valuation.utils.logging import logger
from valuation.utils.types import unpackable

PICKLE_VERSION = 5  # python >= 3.8


@unpackable
@dataclass
class ClientConfig:
    server: str = ("localhost", 11211)
    connect_timeout: float = 1.0
    timeout: float = 1.0
    no_delay: bool = True
    serde: PickleSerde = PickleSerde(pickle_version=PICKLE_VERSION)


@unpackable
@dataclass
class MemcachedConfig:
    """Configuration for memcache.

    - cache_threshold determines the minimum number of seconds a model training needs
    to take to cache its scores. If a model is super fast to train, you may just want
    to re-train it every time without saving the score. In most cases, caching the model,
    even when it takes very little to train, is preferable.
    The default to cache_threshold is 0.3 seconds.

    - if allow_repeated_training is set to true, instead of storing just a single score of a model,
    the cache will store a running average of its score until a certain relative tolerance
    (set by the rtol_threshold argument) is achieved. More precisely, since most machine learning
    model-trainings are non-deterministic, depending on the starting weights or on randomness in
    the training process, the trained model can have very different scores.
    In your workflow, if you observe that the training process is very noisy even relative to the
    same training set, then we recommend to set allow_repeated_training to True.
    If instead the score is not impacted too much by non-deterministic training, setting allow_repeated_training
    to false will speed up the shapley_dval calculation substantially.

    - As mentioned above, the rtol_threshold argument regulates the relative tolerance for returning the running
    average of a model instead of re-training it. If allow_repeated_training is True, set rtol_threshold to
    small values and the shapley coefficients will have higher precision.

    - Similarly to rtol_threshold, min_repetitions regulates repeated trainings by setting the minimum number of
    repeated training a model has to go through before the cache can return its average score.
    If the model training is very noisy, set min_repetitions to higher values and the scores will be more
    reflective of the real average performance of the trained models.
    """

    client_config: ClientConfig = field(default_factory=ClientConfig)
    cache_threshold: float = 0.3
    allow_repeated_training: bool = True
    rtol_threshold: float = 0.1
    min_repetitions: int = 3
    ignore_args: Iterable[str] = None


def _serialize(x):
    pickled_output = BytesIO()
    pickler = Pickler(pickled_output, PICKLE_VERSION)
    pickler.dump(x)
    return pickled_output.getvalue()


def get_running_avg_variance(
    previous_avg: float, previous_variance: float, new_value: float, count: int
):
    """The method uses Welford's algorithm to calculate the running average and variance of
    a set of numbers.

    :param previous_avg: average value at previous step
    :param previous_variance: variance at previous step
    :param new_value: new value in the series of numbers
    :param count: number of points seen so far
    :return: new_average, new_variance, calculated with the new number
    """
    new_average = (new_value + count * previous_avg) / (count + 1)
    new_variance = previous_variance + (
        (new_value - previous_avg) * (new_value - new_average) - previous_variance
    ) / (count + 1)
    return new_average, new_variance


def memcached(
    client_config: ClientConfig = None,
    cache_threshold: float = 0.3,
    allow_repeated_training: bool = False,
    rtol_threshold: float = 0.1,
    min_repetitions: int = 3,
    ignore_args: Iterable[str] = None,
):
    """Decorate a callable with this in order to have transparent caching.

    The function's code, constants and all arguments (except for those in
    `ignore_args` are used to generate the key for the remote cache.

    **FIXME?**:
        Due to the need to pickle memcached functions, this returns a class
        instead of a function. This has the drawback of a messy docstring.

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
    def connect(config: ClientConfig):
        """First tries to establish a connection, then tries setting and
        getting a value."""
        try:
            test_config = dict(**config)
            # test_config.update(timeout=config.connect_timeout)  # allow longer delays
            client = RetryingClient(
                Client(**test_config),
                attempts=3,
                retry_delay=0.1,
                retry_for=[MemcacheUnexpectedCloseError],
            )
        except Exception as e:
            logger.error(
                f"@memcached: Timeout connecting "
                f'to {config["server"]} after '
                f"{config.connect_timeout} seconds: {str(e)}"
            )
            raise e
        else:
            try:
                import uuid

                temp_key = str(uuid.uuid4())
                client.set(temp_key, 7)
                assert client.get(temp_key) == 7
                client.delete(temp_key, 0)
                return client
            except AssertionError as e:
                logger.error(
                    f"@memcached: Failure saving dummy value "
                    f'to {config["server"]}: {str(e)}'
                )

    def wrapper(fun: Callable):
        # noinspection PyUnresolvedReferences
        signature: bytes = _serialize((fun.__code__.co_code, fun.__code__.co_consts))

        @wraps(fun, updated=[])  # don't try to use update() for a class
        class Wrapped:
            def __init__(self, config: ClientConfig):
                self.config = config
                self.cache_info = make_dataclass(
                    "CacheInfo",
                    ["sets", "misses", "hits", "timeouts", "errors", "reconnects"],
                )(0, 0, 0, 0, 0, 0)
                self.client = connect(self.config)

            def __call__(self, *args, **kwargs):
                key_kwargs = {k: v for k, v in kwargs.items() if k not in ignore_args}
                arg_signature: bytes = _serialize((args, list(key_kwargs.items())))

                # FIXME: do I really need to hash this?
                # FIXME: ensure that the hashing algorithm is portable
                # FIXME: determine right bit size
                # NB: I need to create the hasher object here because it can't be
                #  pickled
                key = blake2b(signature + arg_signature).hexdigest().encode("ASCII")
                key_count = (
                    blake2b(signature + arg_signature + _serialize("count"))
                    .hexdigest()
                    .encode("ASCII")
                )
                key_variance = (
                    blake2b(signature + arg_signature + _serialize("variance"))
                    .hexdigest()
                    .encode("ASCII")
                )

                result = self.get_key_value(key)
                if result is None:
                    start = time()
                    result = fun(*args, **kwargs)
                    end = time()
                    if end - start >= cache_threshold or allow_repeated_training:
                        self.client.set(key, result, noreply=True)
                        self.client.set(key_count, 1, noreply=True)
                        self.client.set(key_variance, 0, noreply=True)
                        self.cache_info.sets += 1
                    self.cache_info.misses += 1
                elif allow_repeated_training:
                    self.cache_info.hits += 1
                    count = self.get_key_value(key_count)
                    variance = self.get_key_value(key_variance)
                    error_on_average = (variance / (count)) ** (1 / 2)
                    if (
                        error_on_average > rtol_threshold * result
                        or count <= min_repetitions
                    ):
                        new_value = fun(*args, **kwargs)
                        new_avg, new_var = get_running_avg_variance(
                            result, variance, new_value, count
                        )
                        self.client.set(key, new_avg, noreply=True)
                        self.client.set(key_count, count + 1, noreply=True)
                        self.client.set(key_variance, new_var, noreply=True)
                        self.cache_info.sets += 1
                        result = new_avg
                else:
                    self.cache_info.hits += 1
                return result

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

            def get_key_value(self, key: str):
                result = None
                try:
                    result = self.client.get(key)
                except socket.timeout as e:
                    self.cache_info.timeouts += 1
                    logger.warning(f"{type(self).__name__}: {str(e)}")
                except OSError as e:
                    self.cache_info.errors += 1
                    logger.warning(f"{type(self).__name__}: {str(e)}")
                except AttributeError as e:
                    # FIXME: this depends on _recv() failing on invalid sockets
                    # See pymemcache.base.py,
                    self.cache_info.reconnects += 1
                    logger.warning(f"{type(self).__name__}: {str(e)}")
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
        config = ClientConfig()
        if client_config is not None:
            config.update(client_config)
        return Wrapped(config)

    return wrapper
