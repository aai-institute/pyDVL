import os
from dataclasses import asdict
from typing import Any, Iterable, List, Optional, Tuple, TypeVar, Union

import ray
from ray import ObjectRef
from ray.remote_function import RemoteFunction

from ..config import ParallelConfig

__all__ = [
    "init_parallel_backend",
    "available_cpus",
]

T = TypeVar("T")

_PARALLEL_BACKED: Optional["RayParallelBackend"] = None


class RayParallelBackend:
    """Class used to wrap ray to make it transparent to algorithms. It shouldn't be initialized directly.
    You should instead call `init_parallel_backend`.

    :param config: instance of :class:`~valuation.utils.config.ParallelConfig` with cluster address, number of cpus, etc.

    :Example:

    >>> from valuation.utils.parallel.backend import RayParallelBackend
    >>> from valuation.utils.config import ParallelConfig
    >>> config = ParallelConfig(backend="ray")
    >>> parallel_backend = RayParallelBackend(config)
    >>> parallel_backend
    <RayParallelBackend: {'address': None, 'num_cpus': None}>
    """

    def __init__(self, config: ParallelConfig):
        config_dict = asdict(config)
        config_dict.pop("backend")
        config_dict["num_cpus"] = config_dict.pop("num_workers")
        self.config = config_dict
        ray.init(**self.config)

    def get(
        self, v: Union[ObjectRef, Iterable[ObjectRef], T], *, timeout: int = 300
    ) -> Union[T, Any]:
        if isinstance(v, ObjectRef):
            return ray.get(v, timeout=timeout)
        elif isinstance(v, Iterable):
            return [self.get(x, timeout=timeout) for x in v]
        else:
            return v

    def put(self, x: Any, **kwargs) -> ObjectRef:
        return ray.put(x, **kwargs)  # type: ignore

    def wrap(self, *args, **kwargs) -> RemoteFunction:
        return ray.remote(*args, **kwargs)  # type: ignore

    def wait(
        self,
        object_refs: List["ray.ObjectRef"],
        *,
        num_returns: int = 1,
        timeout: Optional[float] = None,
    ) -> Tuple[List[ObjectRef], List[ObjectRef]]:
        return ray.wait(  # type: ignore
            object_refs,
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

    def __repr__(self) -> str:
        return f"<RayParallelBackend: {self.config}>"


def init_parallel_backend(config: ParallelConfig) -> "RayParallelBackend":
    """Initializes the parallel backend and returns an instance of it.

    :param config: instance of :class:`~valuation.utils.config.ParallelConfig` with cluster address, number of cpus, etc.

    :Example:

    >>> from valuation.utils.parallel.backend import init_parallel_backend
    >>> from valuation.utils.config import ParallelConfig
    >>> config = ParallelConfig(backend="ray")
    >>> parallel_backend = init_parallel_backend(config)
    >>> parallel_backend
    <RayParallelBackend: {'address': None, 'num_cpus': None}>
    """
    global _PARALLEL_BACKED
    if _PARALLEL_BACKED is None:
        if config.backend == "ray":
            _PARALLEL_BACKED = RayParallelBackend(config)
        else:
            raise NotImplementedError(f"Unexpected parallel type {config.backend}")
    return _PARALLEL_BACKED


def available_cpus():
    from platform import system

    if system() != "Linux":
        return os.cpu_count()
    return len(os.sched_getaffinity(0))
