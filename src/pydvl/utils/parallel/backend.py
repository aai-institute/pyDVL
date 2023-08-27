from __future__ import annotations

import logging
import os
from abc import abstractmethod
from concurrent.futures import Executor
from enum import Flag, auto
from typing import Any, Callable, Type, TypeVar

from ..config import ParallelConfig
from ..types import NoPublicConstructor

__all__ = [
    "init_parallel_backend",
    "effective_n_jobs",
    "available_cpus",
    "BaseParallelBackend",
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


class BaseParallelBackend(metaclass=NoPublicConstructor):
    """Abstract base class for all parallel backends."""

    config: dict[str, Any] = {}
    BACKENDS: dict[str, "Type[BaseParallelBackend]"] = {}

    def __init_subclass__(cls, *, backend_name: str, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseParallelBackend.BACKENDS[backend_name] = cls

    @classmethod
    @abstractmethod
    def executor(
        cls,
        max_workers: int | None = None,
        config: ParallelConfig = ParallelConfig(),
        cancel_futures: CancellationPolicy = CancellationPolicy.PENDING,
    ) -> Executor:
        """Returns an executor for the parallel backend."""
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


def init_parallel_backend(config: ParallelConfig) -> BaseParallelBackend:
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
        parallel_backend_cls = BaseParallelBackend.BACKENDS[config.backend]
    except KeyError:
        raise NotImplementedError(f"Unexpected parallel backend {config.backend}")
    return parallel_backend_cls.create(config)  # type: ignore


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


def effective_n_jobs(n_jobs: int, config: ParallelConfig = ParallelConfig()) -> int:
    """Returns the effective number of jobs.

    This number may vary depending on the parallel backend and the resources
    available.

    Args:
        n_jobs: the number of jobs requested. If -1, the number of available
            CPUs is returned.
        config: instance of [ParallelConfig][pydvl.utils.config.ParallelConfig] with
            cluster address, number of cpus, etc.

    Returns:
        The effective number of jobs, guaranteed to be >= 1.

    Raises:
        RuntimeError: if the effective number of jobs returned by the backend
            is < 1.
    """
    parallel_backend = init_parallel_backend(config)
    if (eff_n_jobs := parallel_backend.effective_n_jobs(n_jobs)) < 1:
        raise RuntimeError(
            f"Invalid number of jobs {eff_n_jobs} obtained from parallel backend {config.backend}"
        )
    return eff_n_jobs
