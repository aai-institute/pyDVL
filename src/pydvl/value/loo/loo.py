from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, wait

from tqdm import tqdm

from pydvl.utils import ParallelConfig, Utility, effective_n_jobs, init_executor
from pydvl.value.result import ValuationResult

__all__ = ["compute_loo"]


def compute_loo(
    u: Utility,
    *,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    progress: bool = True,
) -> ValuationResult:
    r"""Computes leave one out value:

    $$v(i) = u(D) - u(D \setminus \{i\}) $$

    Args:
        u: Utility object with model, data, and scoring function
        progress: If True, display a progress bar
        n_jobs: Number of parallel jobs to use
        config: Object configuring parallel computation, with cluster
            address, number of cpus, etc.
        progress: If True, display a progress bar

    Returns:
        Object with the data values.

    !!! tip "New in version 0.8.0"
        Renamed from `naive_loo` and added parallel computation.
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

    max_workers = effective_n_jobs(n_jobs, config)
    n_submitted_jobs = 2 * max_workers  # number of jobs in the queue

    # NOTE: this could be done with a simple executor.map(), but we want to
    # display a progress bar

    with init_executor(
        max_workers=max_workers, config=config, cancel_futures=True
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
