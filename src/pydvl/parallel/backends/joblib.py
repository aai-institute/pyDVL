from __future__ import annotations

import logging
import warnings
from concurrent.futures import Executor
from typing import Callable, Optional, TypeVar, Union, cast

import joblib
from deprecate import deprecated
from joblib import delayed
from joblib.externals.loky import get_reusable_executor

from pydvl.parallel.backend import CancellationPolicy, ParallelBackend
from pydvl.parallel.config import ParallelConfig

__all__ = ["JoblibParallelBackend"]

T = TypeVar("T")

logger = logging.getLogger(__name__)


class JoblibParallelBackend(ParallelBackend, backend_name="joblib"):
    """Class used to wrap joblib to make it transparent to algorithms.

    ??? Example
        ``` python
        from pydvl.parallel import JoblibParallelBackend
        parallel_backend = JoblibParallelBackend()
        ```

    ??? Example
        ``` python
        import joblib
        from pydvl.parallel import JoblibParallelBackend
        with joblib.parallel_config(verbose=100):
            parallel_backend = JoblibParallelBackend()
        ```
    """

    @deprecated(
        target=True,
        args_mapping={"config": None},
        deprecated_in="0.9.0",
        remove_in="0.10.0",
    )
    def __init__(self, config: Optional[ParallelConfig] = None) -> None:
        n_jobs: Optional[int] = None
        if config is not None:
            n_jobs = config.n_cpus_local
        self.config = {
            "n_jobs": n_jobs,
        }

    @classmethod
    def executor(
        cls,
        max_workers: Optional[int] = None,
        *,
        config: Optional[ParallelConfig] = None,
        cancel_futures: Union[CancellationPolicy, bool] = CancellationPolicy.NONE,
    ) -> Executor:
        if config is not None:
            warnings.warn(
                "The `JoblibParallelBackend` uses deprecated arguments: "
                "`config` -> `None`. They were deprecated since v0.9.0 "
                "and will be removed in v0.10.0.",
                FutureWarning,
            )

        if cancel_futures not in (CancellationPolicy.NONE, False):
            warnings.warn(
                "Cancellation of futures is not supported by the joblib backend",
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
