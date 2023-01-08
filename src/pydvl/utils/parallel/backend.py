import functools
import logging
import os
from abc import ABCMeta, abstractmethod
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union

import ray
from ray import ObjectRef
from ray.remote_function import RemoteFunction

from ..config import ParallelConfig

__all__ = [
    "init_parallel_backend",
]

T = TypeVar("T")

_PARALLEL_BACKENDS: Dict[str, "Type[BaseParallelBackend]"] = {}


class NoPublicConstructor(ABCMeta):
    """Metaclass that ensures a private constructor

    If a class uses this metaclass like this:

        class SomeClass(metaclass=NoPublicConstructor):
            pass

    If you try to instantiate your class (`SomeClass()`),
    a `TypeError` will be thrown.

    Taken almost verbatim from:
    https://stackoverflow.com/a/64682734
    """

    def __call__(cls, *args, **kwargs):
        raise TypeError(
            f"{cls.__module__}.{cls.__qualname__} cannot be initialized directly. "
            "Use init_parallel_backend() instead."
        )

    def _create(cls, *args: Any, **kwargs: Any):
        return super().__call__(*args, **kwargs)


class BaseParallelBackend(metaclass=NoPublicConstructor):
    """Abstract base class for all parallel backends"""

    config: Dict[str, Any] = {}

    def __init_subclass__(cls, *, backend_name: str, **kwargs):
        global _PARALLEL_BACKENDS
        _PARALLEL_BACKENDS[backend_name] = cls
        super().__init_subclass__(**kwargs)

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
    def _effective_n_jobs(self, n_jobs: int) -> int:
        ...

    def effective_n_jobs(self, n_jobs: int = -1) -> int:
        if n_jobs == 0:
            raise ValueError("n_jobs == 0 in Parallel has no meaning")
        n_jobs = self._effective_n_jobs(n_jobs)
        return n_jobs

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.config}>"


class SequentialParallelBackend(BaseParallelBackend, backend_name="sequential"):
    """Class used to run jobs sequentially and locally. It shouldn't
    be initialized directly. You should instead call `init_parallel_backend`.

    :param config: instance of :class:`~pydvl.utils.config.ParallelConfig` with number of cpus
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
        return v

    def wrap(self, *args, **kwargs) -> Any:
        assert len(args) == 1
        return functools.partial(args[0], **kwargs)

    def wait(self, v: Any, *args, **kwargs) -> Tuple[list, list]:
        return v, []

    def _effective_n_jobs(self, n_jobs: int) -> int:
        if n_jobs < 0:
            if self.config["num_cpus"]:
                eff_n_jobs: int = self.config["num_cpus"]
            else:
                eff_n_jobs = available_cpus()
        else:
            eff_n_jobs = n_jobs
        return eff_n_jobs


class RayParallelBackend(BaseParallelBackend, backend_name="ray"):
    """Class used to wrap ray to make it transparent to algorithms. It shouldn't
    be initialized directly. You should instead call `init_parallel_backend`.

    :param config: instance of :class:`~pydvl.utils.config.ParallelConfig` with
        cluster address, number of cpus, etc.
    """

    def __init__(self, config: ParallelConfig):
        config_dict = asdict(config)
        config_dict.pop("backend")
        config_dict["num_cpus"] = config_dict.pop("n_local_workers")
        self.config = config_dict
        if not ray.is_initialized():
            ray.init(**self.config)

    def get(
        self,
        v: Union[ObjectRef, Iterable[ObjectRef], T],
        *args,
        **kwargs,
    ) -> Union[T, Any]:
        timeout: Optional[float] = kwargs.get("timeout", None)
        if isinstance(v, ObjectRef):
            return ray.get(v, timeout=timeout)
        elif isinstance(v, Iterable):
            return [self.get(x, timeout=timeout) for x in v]
        else:
            return v

    def put(self, v: T, *args, **kwargs) -> Union["ObjectRef[T]", T]:
        try:
            return ray.put(v, **kwargs)  # type: ignore
        except TypeError:
            return v  # type: ignore

    def wrap(self, *args, **kwargs) -> RemoteFunction:
        return ray.remote(*args, **kwargs)  # type: ignore

    def wait(
        self,
        v: List["ObjectRef"],
        *args,
        **kwargs,
    ) -> Tuple[List[ObjectRef], List[ObjectRef]]:
        num_returns: int = kwargs.get("num_returns", 1)
        timeout: Optional[float] = kwargs.get("timeout", None)
        return ray.wait(  # type: ignore
            v,
            num_returns=num_returns,
            timeout=timeout,
        )

    def _effective_n_jobs(self, n_jobs: int) -> int:
        if n_jobs < 0:
            ray_cpus = int(ray._private.state.cluster_resources()["CPU"])  # type: ignore
            eff_n_jobs = ray_cpus
        else:
            eff_n_jobs = n_jobs
        return eff_n_jobs


def init_parallel_backend(
    config: ParallelConfig,
) -> BaseParallelBackend:
    """Initializes the parallel backend and returns an instance of it.

    :param config: instance of :class:`~pydvl.utils.config.ParallelConfig` with cluster address, number of cpus, etc.

    :Example:

    >>> from pydvl.utils.parallel.backend import init_parallel_backend
    >>> from pydvl.utils.config import ParallelConfig
    >>> config = ParallelConfig(backend="ray")
    >>> parallel_backend = init_parallel_backend(config)
    >>> parallel_backend
    <RayParallelBackend: {'address': None, 'logging_level': 30, 'num_cpus': None}>

    """
    try:
        parallel_backend_cls = _PARALLEL_BACKENDS[config.backend]
    except KeyError:
        raise NotImplementedError(f"Unexpected parallel backend {config.backend}")
    parallel_backend = parallel_backend_cls._create(config)
    return parallel_backend  # type: ignore


def available_cpus() -> int:
    """Platform-independent count of available cores.

    FIXME: do we really need this or is `os.cpu_count` enough? Is this portable?
    :return: Number of cores, or 1 if it is not possible to determine.
    """
    from platform import system

    if system() != "Linux":
        return os.cpu_count() or 1
    return len(os.sched_getaffinity(0))
