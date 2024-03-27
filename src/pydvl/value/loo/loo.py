from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, wait
from typing import Optional

from deprecate import deprecated
from tqdm import tqdm

from pydvl.parallel import ParallelBackend, ParallelConfig, maybe_init_parallel_backend
from pydvl.utils import Utility
from pydvl.value.result import ValuationResult

__all__ = ["compute_loo"]


@deprecated(
    target=True,
    args_mapping={"config": None},
    deprecated_in="0.9.0",
    remove_in="0.10.0",
)
def compute_loo(
    u: Utility,
    *,
    n_jobs: int = 1,
    parallel_backend: Optional[ParallelBackend] = None,
    config: Optional[ParallelConfig] = None,
    progress: bool = True,
) -> ValuationResult:
    r"""Computes leave one out value:

    $$v(i) = u(D) - u(D \setminus \{i\}) $$

    Args:
        u: Utility object with model, data, and scoring function
        progress: If True, display a progress bar
        n_jobs: Number of parallel jobs to use
        parallel_backend: Parallel backend instance to use
            for parallelizing computations. If `None`,
            use [JoblibParallelBackend][pydvl.parallel.backends.JoblibParallelBackend] backend.
            See the [Parallel Backends][pydvl.parallel.backends] package
            for available options.
        config: (**DEPRECATED**) Object configuring parallel computation,
            with cluster address, number of cpus, etc.
        progress: If True, display a progress bar

    Returns:
        Object with the data values.

    !!! tip "New in version 0.7.0"
        Renamed from `naive_loo` and added parallel computation.

    !!! note "Changed in version 0.9.0"
        Deprecated `config` argument and added a `parallel_backend`
        argument to allow users to pass the Parallel Backend configuration.
    """
    if len(u.data) < 3:
        raise ValueError("Dataset must have at least 2 elements")

    result = ValuationResult.zeros(
        algorithm="loo",
        indices=u.data.indices,
        data_names=u.data.data_names,
    )

    all_indices = set(u.data.indices)
    total_utility = u(u.data.indices)

    def fun(idx: int) -> tuple[int, float]:
        return idx, total_utility - u(all_indices.difference({idx}))

    parallel_backend = maybe_init_parallel_backend(parallel_backend, config)
    max_workers = parallel_backend.effective_n_jobs(n_jobs)
    n_submitted_jobs = 2 * max_workers  # number of jobs in the queue

    # NOTE: this could be done with a simple executor.map(), but we want to
    # display a progress bar

    with parallel_backend.executor(
        max_workers=max_workers, cancel_futures=True
    ) as executor:
        pending: set[Future] = set()
        index_it = iter(u.data.indices)

        pbar = tqdm(disable=not progress, total=100, unit="%")
        while True:
            pbar.n = 100 * sum(result.counts) / len(u.data)
            pbar.refresh()
            completed, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
            for future in completed:
                idx, marginal = future.result()
                result.update(idx, marginal)

            # Ensure that we always have n_submitted_jobs running
            try:
                for _ in range(n_submitted_jobs - len(pending)):
                    pending.add(executor.submit(fun, next(index_it)))
            except StopIteration:
                if len(pending) == 0:
                    return result
