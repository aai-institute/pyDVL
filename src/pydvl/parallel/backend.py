from __future__ import annotations

import logging
import os
import warnings
from abc import abstractmethod
from concurrent.futures import Executor
from enum import Flag, auto
from typing import Any, Callable, Optional, Type, Union

from deprecate import deprecated

from .config import ParallelConfig

__all__ = [
    "init_parallel_backend",
    "_maybe_init_parallel_backend",
    "available_cpus",
    "ParallelBackend",
    "CancellationPolicy",
]


log = logging.getLogger(__name__)


class CancellationPolicy(Flag):
    """Policy to use when cancelling futures after exiting an Executor.

    !!! Note
        Not all backends support all policies.

    Attributes:
        NONE: Do not cancel any futures.
        PENDING: Cancel all pending futures, but not running ones.
        RUNNING: Cancel all running futures, but not pending ones.
        ALL: Cancel all pending and running futures.
    """

    NONE = 0
    PENDING = auto()
    RUNNING = auto()
    ALL = PENDING | RUNNING


class ParallelBackend:
    """Abstract base class for all parallel backends."""

    config: dict[str, Any] = {}
    BACKENDS: dict[str, "Type[ParallelBackend]"] = {}

    def __init_subclass__(cls, *, backend_name: str, **kwargs):
        super().__init_subclass__(**kwargs)
        ParallelBackend.BACKENDS[backend_name] = cls

    @classmethod
    @abstractmethod
    def executor(
        cls,
        max_workers: Optional[int] = None,
        *,
        config: Optional[ParallelConfig] = None,
        cancel_futures: Union[CancellationPolicy, bool] = CancellationPolicy.PENDING,
    ) -> Executor:
        """Returns a futures executor for the parallel backend."""
        ...

    @abstractmethod
    def get(self, v: Any, *args, **kwargs):
        ...

    @abstractmethod
    def put(self, v: Any, *args, **kwargs) -> Any:
        ...

    @abstractmethod
    def wrap(self, fun: Callable, **kwargs) -> Callable:
        ...

    @abstractmethod
    def wait(self, v: Any, *args, **kwargs) -> Any:
        ...

    @abstractmethod
    def _effective_n_jobs(self, n_jobs: int) -> int:
        ...

    def effective_n_jobs(self, n_jobs: int = -1) -> int:
        if n_jobs == 0:
            raise ValueError("n_jobs == 0 in Parallel has no meaning")
        n_jobs = self._effective_n_jobs(n_jobs)
        return n_jobs

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.config}>"


@deprecated(
    target=None,
    remove_in="0.10.0",
    deprecated_in="0.9.0",
)
def init_parallel_backend(config: ParallelConfig) -> ParallelBackend:
    """Initializes the parallel backend and returns an instance of it.

    The following example creates a parallel backend instance with the default
    configuration, which is a local joblib backend.

    ??? Example
        ``` python
        config = ParallelConfig()
        parallel_backend = init_parallel_backend(config)
        ```

    To create a parallel backend instance with a different backend, e.g. ray,
    you can pass the backend name as a string to the constructor of
    [ParallelConfig][pydvl.utils.config.ParallelConfig].

    ??? Example
        ```python
        config = ParallelConfig(backend="ray")
        parallel_backend = init_parallel_backend(config)
        ```

    Args:
        config: instance of [ParallelConfig][pydvl.utils.config.ParallelConfig]
            with cluster address, number of cpus, etc.


    """
    try:
        parallel_backend_cls = ParallelBackend.BACKENDS[config.backend]
    except KeyError:
        raise NotImplementedError(f"Unexpected parallel backend {config.backend}")
    return parallel_backend_cls(config)  # type: ignore


# TODO: delete this class once it's made redundant in v0.10.0
# This string for the benefit of deprecation searches:
# remove_in="0.10.0"
def _maybe_init_parallel_backend(
    parallel_backend: Optional[ParallelBackend] = None,
    config: Optional[ParallelConfig] = None,
) -> ParallelBackend:
    """Helper function inside during the deprecation period of
    [][pydvl.parallel.backend.init_parallel_backend] and should be removed in v0.10.0
    """
    if parallel_backend is not None:
        if config is not None:
            warnings.warn(
                "You should not set both `config` and `parallel_backend`. The former will be ignored.",
                UserWarning,
            )
    else:
        if config is not None:
            parallel_backend = init_parallel_backend(config)
        else:
            from pydvl.parallel.backends import JoblibParallelBackend

            parallel_backend = JoblibParallelBackend()
    return parallel_backend


def available_cpus() -> int:
    """Platform-independent count of available cores.

    FIXME: do we really need this or is `os.cpu_count` enough? Is this portable?

    Returns:
        Number of cores, or 1 if it is not possible to determine.
    """
    from platform import system

    if system() != "Linux":
        return os.cpu_count() or 1
    return len(os.sched_getaffinity(0))  # type: ignore
