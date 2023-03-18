from concurrent.futures import Executor, Future
from contextlib import contextmanager
from dataclasses import asdict
from typing import Callable, Generator, TypeVar
from weakref import WeakSet

import ray

from ..config import ParallelConfig
from .backend import effective_n_jobs

__all__ = ["init_executor", "RayExecutor"]

T = TypeVar("T")


@contextmanager
def init_executor(
    max_workers: int, config: ParallelConfig
) -> Generator[Executor, None, None]:
    if config.backend == "ray":
        max_workers = effective_n_jobs(max_workers, config=config)
        executor = RayExecutor(max_workers, config)
    else:
        raise NotImplementedError(f"Unexpected parallel type {config.backend}")
    yield executor


class RayExecutor(Executor):
    def __init__(self, max_workers: int, config: ParallelConfig):
        if config.backend != "ray":
            raise ValueError(
                f"Parallel backend must be set to 'ray' and not {config.backend}"
            )
        self.max_workers = max_workers
        self.futures: "WeakSet[Future]" = WeakSet()
        config_dict = asdict(config)
        config_dict.pop("backend")
        config_dict["num_cpus"] = config_dict.pop("n_local_workers")
        self.config = config_dict
        if not ray.is_initialized():
            ray.init(**self.config)

    def submit(self, fn: Callable[..., T], *args, **kwargs) -> "Future[T]":
        remote_fn = ray.remote(fn)
        ref = remote_fn.remote(*args, **kwargs)
        future: "Future[T]" = ref.future()
        self.futures.add(future)
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        if cancel_futures:
            for future in self.futures:
                future.cancel()
        if wait:
            for future in self.futures:
                future.result()
