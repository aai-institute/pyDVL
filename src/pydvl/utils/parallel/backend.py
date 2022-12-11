import functools
import os
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import ray
from ray import ObjectRef
from ray.remote_function import RemoteFunction

from ..config import ParallelConfig

__all__ = [
    "init_parallel_backend",
    "available_cpus",
]

T = TypeVar("T")

_PARALLEL_BACKENDS: Dict[str, "BaseParallelBackend"] = {}


class BaseParallelBackend(ABC):
    """Abstract base class for all parallel backends"""

    config: Dict[str, Any] = {}

    @abstractmethod
    def get(self, v: Any, *args, **kwargs):
        ...

    @abstractmethod
    def put(self, v: Any, *args, **kwargs) -> Any:
        ...

    @abstractmethod
    def wrap(self, *args, **kwargs) -> Any:
        ...

    @abstractmethod
    def wait(self, v: Any, *args, **kwargs) -> Any:
        ...

    @abstractmethod
    def effective_n_jobs(self, n_jobs: Optional[int]) -> int:
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.config}>"


class SequentialParallelBackend(BaseParallelBackend):
    """Class used to run jobs sequentiall and locally. It shouldn't
    be initialized directly. You should instead call `init_parallel_backend`.

    :param config: instance of :class:`~pydvl.utils.config.ParallelConfig` with number of cpus

    :Example:

    >>> from pydvl.utils.parallel.backend import SequentialParallelBackend
    >>> from pydvl.utils.config import ParallelConfig
    >>> config = ParallelConfig(backend="sequential")
    >>> parallel_backend = SequentialParallelBackend(config)
    >>> parallel_backend
    <SequentialParallelBackend: {'num_cpus': None}>

    """

    def __init__(self, config: ParallelConfig):
        config_dict = asdict(config)
        config_dict.pop("backend")
        config_dict.pop("address")
        config_dict["num_cpus"] = config_dict.pop("n_local_workers")
        self.config = config_dict

    def get(self, v: Any, *args, **kwargs):
        return v

    def put(self, v: Any, *args, **kwargs) -> Any:
        pass

    def wrap(self, *args, **kwargs) -> Any:
        assert len(args) == 1
        return functools.partial(args[0], **kwargs)

    def wait(self, v: Any, *args, **kwargs) -> Tuple[list, list]:
        return v, []

    def effective_n_jobs(self, n_jobs: Optional[int]) -> int:
        if n_jobs == 0:
            raise ValueError("n_jobs == 0 in Parallel has no meaning")
        elif n_jobs is None or n_jobs < 0:
            if self.config["num_cpus"]:
                eff_n_jobs = self.config["num_cpus"]
            else:
                eff_n_jobs = available_cpus()
        else:
            eff_n_jobs = n_jobs
        return eff_n_jobs


class RayParallelBackend(BaseParallelBackend):
    """Class used to wrap ray to make it transparent to algorithms. It shouldn't
    be initialized directly. You should instead call `init_parallel_backend`.

    :param config: instance of :class:`~pydvl.utils.config.ParallelConfig` with
        cluster address, number of cpus, etc.

    :Example:

    >>> from pydvl.utils.parallel.backend import RayParallelBackend
    >>> from pydvl.utils.config import ParallelConfig
    >>> config = ParallelConfig(backend="ray")
    >>> parallel_backend = RayParallelBackend(config)
    >>> parallel_backend
    <RayParallelBackend: {'address': None, 'num_cpus': None}>

    """

    def __init__(self, config: ParallelConfig):
        config_dict = asdict(config)
        config_dict.pop("backend")
        config_dict["num_cpus"] = config_dict.pop("n_local_workers")
        self.config = config_dict
        ray.init(**self.config)

    def get(
        self,
        v: Union[ObjectRef, Iterable[ObjectRef], T],
        *,
        timeout: Optional[float] = None,
    ) -> Union[T, Any]:
        if isinstance(v, ObjectRef):
            return ray.get(v, timeout=timeout)
        elif isinstance(v, Iterable):
            return [self.get(x, timeout=timeout) for x in v]
        else:
            return v

    def put(self, v: Any, **kwargs) -> ObjectRef:
        return ray.put(v, **kwargs)  # type: ignore

    def wrap(self, *args, **kwargs) -> RemoteFunction:
        return ray.remote(*args, **kwargs)  # type: ignore

    def wait(
        self,
        v: List["ray.ObjectRef"],
        *,
        num_returns: int = 1,
        timeout: Optional[float] = None,
    ) -> Tuple[List[ObjectRef], List[ObjectRef]]:
        return ray.wait(  # type: ignore
            v,
            num_returns=num_returns,
            timeout=timeout,
        )

    def effective_n_jobs(self, n_jobs: Optional[int]) -> int:
        if n_jobs == 0:
            raise ValueError("n_jobs == 0 in Parallel has no meaning")
        elif n_jobs is None or n_jobs < 0:
            ray_cpus = int(ray._private.state.cluster_resources()["CPU"])  # type: ignore
            eff_n_jobs = ray_cpus
        else:
            eff_n_jobs = n_jobs
        return eff_n_jobs


def init_parallel_backend(config: ParallelConfig) -> "BaseParallelBackend":
    """Initializes the parallel backend and returns an instance of it.

    :param config: instance of :class:`~pydvl.utils.config.ParallelConfig` with cluster address, number of cpus, etc.

    :Example:

    >>> from pydvl.utils.parallel.backend import init_parallel_backend
    >>> from pydvl.utils.config import ParallelConfig
    >>> config = ParallelConfig(backend="ray")
    >>> parallel_backend = init_parallel_backend(config)
    >>> parallel_backend
    <RayParallelBackend: {'address': None, 'num_cpus': None}>

    """
    global _PARALLEL_BACKENDS
    if config.backend not in ["sequential", "ray"]:
        raise NotImplementedError(f"Unexpected parallel type {config.backend}")
    if config.backend not in _PARALLEL_BACKENDS:
        if config.backend == "ray":
            _PARALLEL_BACKENDS["ray"] = RayParallelBackend(config)
        else:
            _PARALLEL_BACKENDS["sequential"] = SequentialParallelBackend(config)
    return _PARALLEL_BACKENDS[config.backend]


def available_cpus() -> int:
    """Platform-independent count of available cores.

    FIXME: do we really need this or is `os.cpu_count` enough? Is this portable?
    :return: Number of cores, or 1 if it is not possible to determine.
    """
    from platform import system

    if system() != "Linux":
        return os.cpu_count() or 1
    return len(os.sched_getaffinity(0))
