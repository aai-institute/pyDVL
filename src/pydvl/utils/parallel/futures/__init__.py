import logging
from concurrent.futures import Executor
from contextlib import contextmanager
from typing import Generator, Optional

from pydvl.utils.config import ParallelConfig
from pydvl.utils.parallel.backend import BaseParallelBackend

try:
    from pydvl.utils.parallel.futures.ray import RayExecutor
except ImportError:
    pass

__all__ = ["init_executor"]

logger = logging.getLogger(__name__)


@contextmanager
def init_executor(
    max_workers: Optional[int] = None,
    config: ParallelConfig = ParallelConfig(),
    **kwargs,
) -> Generator[Executor, None, None]:
    """Initializes a futures executor based on the passed parallel configuration object.

    :param max_workers: Maximum number of concurrent tasks.
    :param config: instance of :class:`~pydvl.utils.config.ParallelConfig` with
        cluster address, number of cpus, etc.
    :param kwargs: Optional parameters that will be passed to the executor,
        e.g. ``cancel_futures`` for executors that support it, like
        :class:`~pydvl.utils.parallel.futures.ray.RayExecutor`.

    :Examples:

    .. code-block:: python

       from pydvl.utils.parallel.futures import init_executor
       from pydvl.utils.config import ParallelConfig
       config = ParallelConfig(backend="ray")
       with init_executor(max_workers=1, config=config) as executor:
           future = executor.submit(lambda x: x + 1, 1)
           result = future.result()
       assert result == 2

    .. code-block:: python

       from pydvl.utils.parallel.futures import init_executor
       with init_executor() as executor:
           results = list(executor.map(lambda x: x + 1, range(5)))
       assert results == [1, 2, 3, 4, 5]

    """
    try:
        cls = BaseParallelBackend.BACKENDS[config.backend]
        with cls.executor(max_workers=max_workers, config=config, **kwargs) as e:
            yield e
    except KeyError:
        raise NotImplementedError(f"Unexpected parallel backend {config.backend}")
