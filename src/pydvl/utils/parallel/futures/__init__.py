from concurrent.futures import Executor
from contextlib import contextmanager
from typing import Generator, Optional

from joblib.externals.loky import get_reusable_executor

from pydvl.utils.config import ParallelConfig
from pydvl.utils.parallel.futures.ray import RayExecutor

__all__ = ["init_executor"]


@contextmanager
def init_executor(
    max_workers: Optional[int] = None,
    config: ParallelConfig = ParallelConfig(),
    **kwargs,
) -> Generator[Executor, None, None]:
    """Initializes a futures executor based on the passed parallel configuration object.

    :param max_workers: Maximum number of concurrent tasks.
    :param config: instance of :class:`~pydvl.utils.config.ParallelConfig` with cluster address, number of cpus, etc.
    :param kwargs: Other optional parameter that will be passed to the executor.

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
        with RayExecutor(max_workers, config=config, **kwargs) as executor:
            yield executor
    elif config.backend == "joblib":
        with get_reusable_executor(max_workers=max_workers, **kwargs) as executor:
            yield executor
    else:
        raise NotImplementedError(f"Unexpected parallel type {config.backend}")
