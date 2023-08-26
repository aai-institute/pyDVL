from __future__ import annotations

from concurrent.futures import Executor
from typing import Callable, cast

import joblib
from joblib import delayed
from joblib.externals.loky import get_reusable_executor

from pydvl.utils import ParallelConfig
from pydvl.utils.parallel.backend import BaseParallelBackend, CancellationPolicy, T, log


class JoblibParallelBackend(BaseParallelBackend, backend_name="joblib"):
    """Class used to wrap joblib to make it transparent to algorithms.

    It shouldn't be initialized directly. You should instead call
    :func:`~pydvl.utils.parallel.backend.init_parallel_backend`.

    :param config: instance of :class:`~pydvl.utils.config.ParallelConfig` with
        cluster address, number of cpus, etc.
    """

    def __init__(self, config: ParallelConfig):
        self.config = {
            "logging_level": config.logging_level,
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
            log.warning(
                "Cancellation of futures is not supported by the joblib backend"
            )
        return cast(Executor, get_reusable_executor(max_workers=max_workers))

    def get(self, v: T, *args, **kwargs) -> T:
        return v

    def put(self, v: T, *args, **kwargs) -> T:
        return v

    def wrap(self, fun: Callable, **kwargs) -> Callable:
        """Wraps a function as a joblib delayed.

        :param fun: the function to wrap

        :return: The delayed function.
        """
        return delayed(fun)  # type: ignore

    def wait(self, v: list[T], *args, **kwargs) -> tuple[list[T], list[T]]:
        return v, []

    def _effective_n_jobs(self, n_jobs: int) -> int:
        if self.config["n_jobs"] is None:
            maximum_n_jobs = joblib.effective_n_jobs()
        else:
            maximum_n_jobs = self.config["n_jobs"]
        eff_n_jobs: int = min(joblib.effective_n_jobs(n_jobs), maximum_n_jobs)
        return eff_n_jobs
