from concurrent.futures import Executor
from contextlib import contextmanager
from typing import Generator, Optional

from pydvl.utils.config import ParallelConfig
from pydvl.utils.parallel.backend import BaseParallelBackend
from pydvl.utils.parallel.futures.ray import RayExecutor

__all__ = ["init_executor"]


@contextmanager
def init_executor(
    max_workers: Optional[int] = None,
    config: ParallelConfig = ParallelConfig(),
    **kwargs: dict,
) -> Generator[Executor, None, None]:
    """Initializes a futures executor for the given parallel configuration.

    Args:
        max_workers: Maximum number of concurrent tasks.
        config: instance of [ParallelConfig][pydvl.utils.config.ParallelConfig]
            with cluster address, number of cpus, etc.
        kwargs: Other optional parameter that will be passed to the executor.

    Examples:

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
    try:
        cls = BaseParallelBackend.BACKENDS[config.backend]
        with cls.executor(max_workers=max_workers, config=config, **kwargs) as e:
            yield e
    except KeyError:
        raise NotImplementedError(f"Unexpected parallel backend {config.backend}")
