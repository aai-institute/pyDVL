import logging
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional, Tuple, Union

from pymemcache.serde import PickleSerde

PICKLE_VERSION = 5  # python >= 3.8

__all__ = ["ParallelConfig", "MemcachedClientConfig", "MemcachedConfig"]


@dataclass
class ParallelConfig:
    """Configuration for parallel computation backend.

    backend: Type of backend to use.
    Defaults to 'ray'
    address: Address of existing remote or local cluster to use.
    n_cpus_local: Number of CPUs to use when creating a local ray cluster.
    This has no effect when using an existing ray cluster.
    logging_level: Logging level for the parallel backend's worker.
    """

    backend: Literal["sequential", "ray"] = "ray"
    address: Optional[Union[str, Tuple[str, int]]] = None
    n_cpus_local: Optional[int] = None
    logging_level: int = logging.WARNING

    def __post_init__(self) -> None:
        if self.address is not None and self.n_cpus_local is not None:
            raise ValueError("When `address` is set, `n_cpus_local` should be None.")


@dataclass
class MemcachedClientConfig:
    """Configuration of the memcached client.

    server: A tuple of (IP|domain name, port).
    connect_timeout: How many seconds to wait before raising
    `ConnectionRefusedError` on failure to connect.
    timeout: seconds to wait for send or recv calls on the socket
    connected to memcached.
    no_delay: set the `TCP_NODELAY` flag, which may help with performance
    in some cases.
    serde: a serializer / deserializer ("serde"). The default
    `PickleSerde` should work in most cases. See `pymemcached's
    documentation
    <https://pymemcache.readthedocs.io/en/latest/apidoc/pymemcache.client.base.html#pymemcache.client.base.Client>`_
    for details.
    """

    server: Tuple[str, int] = ("localhost", 11211)
    connect_timeout: float = 1.0
    timeout: float = 1.0
    no_delay: bool = True
    serde: PickleSerde = PickleSerde(pickle_version=PICKLE_VERSION)


@dataclass
class MemcachedConfig:
    """Configuration for [memcached()][pydvl.utils.caching.memcached], providing
    memoization of function calls.

    Instances of this class are typically used as arguments for the construction
    of a :class:`~pydvl.utils.utility.Utility`.

        client_config: Configuration for the connection to the memcached
        server.
        time_threshold: computations taking less time than this many seconds
        are not cached.
        allow_repeated_evaluations: If `True`, repeated calls to a function
        with the same arguments will be allowed and outputs averaged until the
        running standard deviation of the mean stabilises below
        `rtol_stderr * mean`.
        rtol_stderr: relative tolerance for repeated evaluations. More
        precisely, [memcached()][pydvl.utils.caching.memcached] will stop evaluating
        the function once the standard deviation of the mean is smaller than
        `rtol_stderr * mean`.
        min_repetitions: minimum number of times that a function evaluation
        on the same arguments is repeated before returning cached values. Useful
        for stochastic functions only. If the model training is very noisy, set
        this number to higher values to reduce variance.
        ignore_args: Do not take these keyword arguments into account when
        hashing the wrapped function for usage as key in memcached.
    """

    client_config: MemcachedClientConfig = field(default_factory=MemcachedClientConfig)
    time_threshold: float = 0.3
    allow_repeated_evaluations: bool = False
    rtol_stderr: float = 0.1
    min_repetitions: int = 3
    ignore_args: Optional[Iterable[str]] = None
