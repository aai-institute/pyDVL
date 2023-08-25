from __future__ import annotations

import logging
import os
from abc import abstractmethod
from concurrent.futures import Executor
from enum import Enum
from typing import Any, Callable, Iterable, Type, TypeVar, cast

import joblib
import ray
from joblib import delayed
from joblib.externals.loky import get_reusable_executor
from ray import ObjectRef
from ray.util.joblib import register_ray

from ..config import ParallelConfig
from ..types import NoPublicConstructor

__all__ = ["init_parallel_backend", "effective_n_jobs", "available_cpus"]


T = TypeVar("T")

log = logging.getLogger(__name__)


class CancellationPolicy(Enum):
    """Policy to use when cancelling futures after exiting an Executor.

    .. note:
       Not all backends support all policies.

    :cvar NONE: Do not cancel any futures.
    :cvar PENDING: Cancel all pending futures, but not running ones.
    :cvar RUNNING: Cancel all running futures, but not pending ones.
    :cvar ALL: Cancel all pending and running futures.
    """

    NONE = 0
    PENDING = 1
    RUNNING = 2
    ALL = 3

    def __and__(self, other: CancellationPolicy) -> bool:
        return int(self.value) & int(other.value) > 0


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


class JoblibParallelBackend(BaseParallelBackend, backend_name="joblib"):
    """Class used to wrap joblib to make it transparent to algorithms.

    It shouldn't be initialized directly. You should instead call
    [init_parallel_backend()][pydvl.utils.parallel.backend.init_parallel_backend].

    Args:
        config: instance of [ParallelConfig][pydvl.utils.config.ParallelConfig]
            with cluster address, number of cpus, etc.
    """

    def __init__(self, config: ParallelConfig):
        self.config = {
            "logging_level": config.logging_level,
            "n_jobs": config.n_cpus_local,
        }

    @classmethod
    def executor(
        cls,
        max_workers: int | None = None,
        config: ParallelConfig = ParallelConfig(),
        cancel_futures: CancellationPolicy = CancellationPolicy.NONE,
    ) -> Executor:
        if cancel_futures not in (CancellationPolicy.NONE, False):
            log.warning(
                "Cancellation of futures is not supported by the joblib backend"
            )
        return cast(Executor, get_reusable_executor(max_workers=max_workers))

    def get(self, v: T, *args, **kwargs) -> T:
        return v

    def put(self, v: T, *args, **kwargs) -> T:
        return v

    def wrap(self, fun: Callable, **kwargs) -> Callable:
        """Wraps a function as a joblib delayed.

        Args:
            fun: the function to wrap

        Returns:
            The delayed function.
        """
        return delayed(fun)  # type: ignore

    def wait(self, v: list[T], *args, **kwargs) -> tuple[list[T], list[T]]:
        return v, []

    def _effective_n_jobs(self, n_jobs: int) -> int:
        if self.config["n_jobs"] is None:
            maximum_n_jobs = joblib.effective_n_jobs()
        else:
            maximum_n_jobs = self.config["n_jobs"]
        eff_n_jobs: int = min(joblib.effective_n_jobs(n_jobs), maximum_n_jobs)
        return eff_n_jobs


class RayParallelBackend(BaseParallelBackend, backend_name="ray"):
    """Class used to wrap ray to make it transparent to algorithms.

    It shouldn't be initialized directly. You should instead call
    [init_parallel_backend()][pydvl.utils.parallel.backend.init_parallel_backend].

    Args:
        config: instance of [ParallelConfig][pydvl.utils.config.ParallelConfig]
            with cluster address, number of cpus, etc.
    """

    def __init__(self, config: ParallelConfig):
        self.config = {"address": config.address, "logging_level": config.logging_level}
        if self.config["address"] is None:
            self.config["num_cpus"] = config.n_cpus_local
        if not ray.is_initialized():
            ray.init(**self.config)
        # Register ray joblib backend
        register_ray()

    @classmethod
    def executor(
        cls,
        max_workers: int | None = None,
        config: ParallelConfig = ParallelConfig(),
        cancel_futures: CancellationPolicy = CancellationPolicy.PENDING,
    ) -> Executor:
        from pydvl.utils.parallel.futures.ray import RayExecutor

        return RayExecutor(max_workers, config=config, cancel_futures=cancel_futures)  # type: ignore

    def get(self, v: ObjectRef | Iterable[ObjectRef] | T, *args, **kwargs) -> T | Any:
        timeout: float | None = kwargs.get("timeout", None)
        if isinstance(v, ObjectRef):
            return ray.get(v, timeout=timeout)
        elif isinstance(v, Iterable):
            return [self.get(x, timeout=timeout) for x in v]
        else:
            return v

    def put(self, v: T, *args, **kwargs) -> ObjectRef[T] | T:
        try:
            return ray.put(v, **kwargs)  # type: ignore
        except TypeError:
            return v  # type: ignore

    def wrap(self, fun: Callable, **kwargs: dict) -> Callable:
        """Wraps a function as a ray remote.

        Args:
            fun: the function to wrap
            kwargs: keyword arguments to pass to @ray.remote

        Returns:
            The `.remote` method of the ray `RemoteFunction`.
        """
        if len(kwargs) > 0:
            return ray.remote(**kwargs)(fun).remote  # type: ignore
        return ray.remote(fun).remote  # type: ignore

    def wait(
        self, v: list[ObjectRef], *args, **kwargs: dict
    ) -> tuple[list[ObjectRef], list[ObjectRef]]:
        num_returns: int = kwargs.get("num_returns", 1)
        timeout: float | None = kwargs.get("timeout", None)
        return ray.wait(v, num_returns=num_returns, timeout=timeout)  # type: ignore

    def _effective_n_jobs(self, n_jobs: int) -> int:
        ray_cpus = int(ray._private.state.cluster_resources()["CPU"])  # type: ignore
        if n_jobs < 0:
            eff_n_jobs = ray_cpus
        else:
            eff_n_jobs = min(n_jobs, ray_cpus)
        return eff_n_jobs


def init_parallel_backend(config: ParallelConfig) -> BaseParallelBackend:
    """Initializes the parallel backend and returns an instance of it.

    Args:
        config: instance of [ParallelConfig][pydvl.utils.config.ParallelConfig]
            with cluster address, number of cpus, etc.

    Example:

        >>> from pydvl.utils.parallel.backend import init_parallel_backend
        >>> from pydvl.utils.config import ParallelConfig
        >>> config = ParallelConfig()
        >>> parallel_backend = init_parallel_backend(config)
        >>> parallel_backend
        <JoblibParallelBackend: {'logging_level': 30, 'n_jobs': None}>

        >>> from pydvl.utils.parallel.backend import init_parallel_backend
        >>> from pydvl.utils.config import ParallelConfig
        >>> config = ParallelConfig(backend="ray")
        >>> parallel_backend = init_parallel_backend(config)
        >>> parallel_backend
        <RayParallelBackend: {'address': None, 'logging_level': 30, 'num_cpus': None}>

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
