from __future__ import annotations

import logging
import warnings
from concurrent.futures import Executor
from typing import Callable, TypeVar, cast

import joblib
from joblib import delayed
from joblib.externals.loky import get_reusable_executor

from pydvl.parallel.backend import BaseParallelBackend, CancellationPolicy
from pydvl.parallel.config import ParallelConfig

__all__ = ["JoblibParallelBackend"]

T = TypeVar("T")

logger = logging.getLogger(__name__)


class JoblibParallelBackend(BaseParallelBackend, backend_name="joblib"):
    """Class used to wrap joblib to make it transparent to algorithms.

    ??? Example
        ``` python
        from pydvl.parallel import init_parallel_backend, ParallelConfig
        config = ParallelConfig(backend="joblib")
        parallel_backend = init_parallel_backend(config)
        ```

    ??? Example
        ``` python
        import joblib
        from pydvl.parallel import init_paralle_backend, ParallelConfig
        with joblib.parallel_config(verbose=100):
            config = ParallelConfig(backend="joblib")
            parallel_backend = init_parallel_backend(config)
        ```

    Args:
        config: instance of [ParallelConfig][pydvl.utils.config.ParallelConfig]
            with cluster address, number of cpus, etc.
    """

    def __init__(self, config: ParallelConfig):
        self.config = {
            "n_jobs": config.n_cpus_local,
        }

    @classmethod
    def executor(
        cls,
        max_workers: int | None = None,
        config: ParallelConfig = ParallelConfig(),
        cancel_futures: CancellationPolicy = CancellationPolicy.NONE,
    ) -> Executor:
        if cancel_futures not in (CancellationPolicy.NONE, False):
            warnings.warn(
                "Cancellation of futures is not supported by the joblib backend"
            )
        return cast(Executor, get_reusable_executor(max_workers=max_workers))

    def get(self, v: T, *args, **kwargs) -> T:
        return v

    def put(self, v: T, *args, **kwargs) -> T:
        return v

    def wrap(self, fun: Callable, **kwargs) -> Callable:
        """Wraps a function as a joblib delayed.

        Args:
            fun: the function to wrap

        Returns:
            The delayed function.
        """
        return delayed(fun)  # type: ignore

    def wait(self, v: list[T], *args, **kwargs) -> tuple[list[T], list[T]]:
        return v, []

    def _effective_n_jobs(self, n_jobs: int) -> int:
        eff_n_jobs: int = joblib.effective_n_jobs(n_jobs)
        if self.config["n_jobs"] is not None:
            maximum_n_jobs = self.config["n_jobs"]
            eff_n_jobs = min(eff_n_jobs, maximum_n_jobs)
        return eff_n_jobs
