import os
from dataclasses import asdict
from typing import Any, Iterable, Optional, TypeVar, Union

import ray
from ray import ObjectRef

from ..config import ParallelConfig

__all__ = [
    "init_parallel_backend",
    "available_cpus",
]

T = TypeVar("T")

_PARALLEL_BACKED: Optional["RayParallelBackend"] = None


class RayParallelBackend:
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

    def wrap(self, x):
        return ray.remote(x)

    def effective_n_jobs(self, n_jobs: Optional[int]) -> int:
        if n_jobs == 0:
            raise ValueError("n_jobs == 0 in Parallel has no meaning")
        elif n_jobs is None or n_jobs < 0:
            ray_cpus = int(ray._private.state.cluster_resources()["CPU"])  # type: ignore
            eff_n_jobs = ray_cpus
        else:
            eff_n_jobs = n_jobs
        return eff_n_jobs


def init_parallel_backend(config: ParallelConfig) -> "RayParallelBackend":
    global _PARALLEL_BACKED
    if _PARALLEL_BACKED is None:
        _PARALLEL_BACKED = RayParallelBackend(config)
    return _PARALLEL_BACKED


def available_cpus():
    from platform import system

    if system() != "Linux":
        return os.cpu_count()
    return len(os.sched_getaffinity(0))
