from __future__ import annotations

from concurrent.futures import Executor
from typing import Any, Callable, Iterable, TypeVar

import ray
from ray import ObjectRef
from ray.util.joblib import register_ray

from pydvl.utils import ParallelConfig
from pydvl.utils.parallel.backend import BaseParallelBackend, CancellationPolicy

__all__ = ["RayParallelBackend"]


T = TypeVar("T")


class RayParallelBackend(BaseParallelBackend, backend_name="ray"):
    """Class used to wrap ray to make it transparent to algorithms.

    It shouldn't be initialized directly. You should instead call
    :func:`~pydvl.utils.parallel.backend.init_parallel_backend`.

    :param config: instance of :class:`~pydvl.utils.config.ParallelConfig` with
        cluster address, number of cpus, etc.
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

    def wrap(self, fun: Callable, **kwargs) -> Callable:
        """Wraps a function as a ray remote.

        :param fun: the function to wrap
        :param kwargs: keyword arguments to pass to @ray.remote

        :return: The `.remote` method of the ray `RemoteFunction`.
        """
        if len(kwargs) > 0:
            return ray.remote(**kwargs)(fun).remote  # type: ignore
        return ray.remote(fun).remote  # type: ignore

    def wait(
        self, v: list[ObjectRef], *args, **kwargs
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
