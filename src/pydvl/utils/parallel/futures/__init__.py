from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Generator

from pydvl.utils.config import ParallelConfig
from pydvl.utils.parallel.futures.ray import RayExecutor

__all__ = ["init_executor", "RayExecutor"]


@contextmanager
def init_executor(
    config: ParallelConfig = ParallelConfig(),
) -> Generator[Executor, None, None]:
    """Initializes a futures executor based on the passed parallel configuration object.

    :param max_workers: Maximum number of concurrent tasks.
    :param config: instance of :class:`~pydvl.utils.config.ParallelConfig` with cluster address, number of cpus, etc.

    :Example:

    >>> from pydvl.utils.parallel.futures import init_executor
    >>> from pydvl.utils.config import ParallelConfig
    >>> config = ParallelConfig(backend="ray")
    >>> with init_executor(config=config) as executor:
    ...     pass

    """
    if config.backend == "ray":
        max_workers = config.n_workers
        with RayExecutor(max_workers, config=config) as executor:
            yield executor
    elif config.backend == "sequential":
        max_workers = 1
        with ThreadPoolExecutor(max_workers) as executor:
            yield executor
    else:
        raise NotImplementedError(f"Unexpected parallel type {config.backend}")
