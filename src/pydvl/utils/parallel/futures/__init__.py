from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Generator, Optional

from pydvl.utils.config import ParallelConfig
from pydvl.utils.parallel.futures.ray import RayExecutor

__all__ = ["init_executor"]


@contextmanager
def init_executor(
    max_workers: Optional[int] = None,
    config: ParallelConfig = ParallelConfig(),
) -> Generator[Executor, None, None]:
    """Initializes a futures executor based on the passed parallel configuration object.

    :param max_workers: Maximum number of concurrent tasks.
    :param config: instance of :class:`~pydvl.utils.config.ParallelConfig` with cluster address, number of cpus, etc.

    :Example:

    >>> from pydvl.utils.parallel.futures import init_executor
    >>> from pydvl.utils.config import ParallelConfig
    >>> config = ParallelConfig(backend="ray")
    >>> with init_executor(max_workers=3, config=config) as executor:
    ...     pass

    >>> from pydvl.utils.parallel.futures import init_executor
    >>> with init_executor() as executor:
    ...     future = executor.submit(lambda x: x + 1, 1)
    ...     result = future.result()
    ...
    >>> print(result)
    2

    >>> from pydvl.utils.parallel.futures import init_executor
    >>> with init_executor() as executor:
    ...     results = list(executor.map(lambda x: x + 1, range(5)))
    ...
    >>> print(results)
    [1, 2, 3, 4, 5]

    """
    if config.backend == "ray":
        with RayExecutor(max_workers, config=config) as executor:
            yield executor
    elif config.backend == "sequential":
        max_workers = 1
        with ThreadPoolExecutor(max_workers) as executor:
            yield executor
    else:
        raise NotImplementedError(f"Unexpected parallel type {config.backend}")
