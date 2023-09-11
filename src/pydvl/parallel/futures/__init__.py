from concurrent.futures import Executor
from contextlib import contextmanager
from typing import Generator, Optional

from pydvl.parallel.backend import BaseParallelBackend
from pydvl.parallel.config import ParallelConfig

try:
    from pydvl.parallel.futures.ray import RayExecutor
except ImportError:
    pass

__all__ = ["init_executor"]


@contextmanager
def init_executor(
    max_workers: Optional[int] = None,
    config: ParallelConfig = ParallelConfig(),
    **kwargs,
) -> Generator[Executor, None, None]:
    """Initializes a futures executor for the given parallel configuration.

    Args:
        max_workers: Maximum number of concurrent tasks.
        config: instance of [ParallelConfig][pydvl.utils.config.ParallelConfig]
            with cluster address, number of cpus, etc.
        kwargs: Other optional parameter that will be passed to the executor.


    ??? Examples
        ``` python
        from pydvl.parallel import init_executor, ParallelConfig

        config = ParallelConfig(backend="ray")
        with init_executor(max_workers=1, config=config) as executor:
            future = executor.submit(lambda x: x + 1, 1)
            result = future.result()
        assert result == 2
        ```
        ``` python
        from pydvl.parallel.futures import init_executor
        with init_executor() as executor:
            results = list(executor.map(lambda x: x + 1, range(5)))
        assert results == [1, 2, 3, 4, 5]
        ```
    """
    try:
        cls = BaseParallelBackend.BACKENDS[config.backend]
        with cls.executor(max_workers=max_workers, config=config, **kwargs) as e:
            yield e
    except KeyError:
        raise NotImplementedError(f"Unexpected parallel backend {config.backend}")
