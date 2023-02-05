import logging
import warnings
from typing import Iterable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from pydvl.utils.config import ParallelConfig
from pydvl.utils.numeric import random_powerset
from pydvl.utils.parallel import MapReduceJob
from pydvl.utils.progress import maybe_progress
from pydvl.utils.utility import Utility
from pydvl.value.least_core._common import LeastCoreProblem, lc_solve_problem
from pydvl.value.results import ValuationResult

logger = logging.getLogger(__name__)


__all__ = [
    "montecarlo_least_core",
    "mclc_prepare_problem",
]


def _montecarlo_least_core(
    u: Utility,
    n_iterations: int,
    *,
    progress: bool = False,
    job_id: int = 1,
    **kwargs,
) -> LeastCoreProblem:
    """Computes utility values and the Least Core upper bound matrix for a given number of iterations.

    :param u: Utility object with model, data, and scoring function
    :param n_iterations: total number of iterations to use
    :param progress: If True, shows a tqdm progress bar
    :param job_id: Integer id used to determine the position of the progress bar
    :return:
    """
    n = len(u.data)

    utility_values = np.zeros(n_iterations)

    # Randomly sample subsets of full dataset
    power_set = random_powerset(
        u.data.indices,
        max_subsets=n_iterations,
    )

    A_lb = np.zeros((n_iterations, n))

    for i, subset in enumerate(
        maybe_progress(
            power_set,
            progress,
            total=n_iterations,
            position=job_id,
        )
    ):
        indices = np.zeros(n, dtype=bool)
        indices[list(subset)] = True
        A_lb[i, indices] = 1
        utility_values[i] = u(subset)

    return LeastCoreProblem(utility_values, A_lb)


def _reduce_func(
    results: Iterable[Tuple[NDArray[np.float_], NDArray[np.float_]]]
) -> LeastCoreProblem:
    """Combines the results from different parallel runs of
    :func:`_montecarlo_least_core`"""
    utility_values_list, A_lb_list = zip(*results)
    utility_values = np.concatenate(utility_values_list)
    A_lb = np.concatenate(A_lb_list)
    return LeastCoreProblem(utility_values, A_lb)


def montecarlo_least_core(
    u: Utility,
    n_iterations: int,
    *,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    options: Optional[dict] = None,
    progress: bool = False,
) -> ValuationResult:
    r"""Computes approximate Least Core values using a Monte Carlo approach.

    $$
    \begin{array}{lll}
    \text{minimize} & \displaystyle{e} & \\
    \text{subject to} & \displaystyle\sum_{i\in N} x_{i} = v(N) & \\
    & \displaystyle\sum_{i\in S} x_{i} + e \geq v(S) & ,
    \forall S \in \{S_1, S_2, \dots, S_m \overset{\mathrm{iid}}{\sim} U(2^N) \}
    \end{array}
    $$

    Where:

    * $U(2^N)$ is the uniform distribution over the powerset of $N$.
    * $m$ is the number of subsets that will be sampled and whose utility will
      be computed and used to compute the data values.

    :param u: Utility object with model, data, and scoring function
    :param n_iterations: total number of iterations to use
    :param n_jobs: number of jobs across which to distribute the computation
    :param config: Object configuring parallel computation, with cluster
        address, number of cpus, etc.
    :param options: Keyword arguments that will be used to select a solver
        and to configure it. Refer to the following page for all possible options:
        https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options
    :param progress: If True, shows a tqdm progress bar
    :return: Object with the data values and the least core value.
    """
    problem = mclc_prepare_problem(
        u, n_iterations, n_jobs=n_jobs, config=config, progress=progress
    )
    return lc_solve_problem(
        u, problem, algorithm="montecarlo_least_core", **(options or {})
    )


def mclc_prepare_problem(
    u: Utility,
    n_iterations: int,
    *,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    progress: bool = False,
) -> LeastCoreProblem:
    """Prepares a linear problem by sampling subsets of the data.
    Use this to separate the problem preparation from the solving with
    :func:`~pydvl.value.least_core._common.lc_solve_problem`. Useful for
    parallel execution of multiple experiments.

    See :func:`montecarlo_least_core` for argument descriptions.
    """
    n = len(u.data)

    if n_iterations < n:
        raise ValueError(
            "Number of iterations should be greater than the size of the " "dataset"
        )

    if n_iterations > 2**n:
        warnings.warn(
            f"Passed n_iterations is greater than the number subsets! "
            f"Setting it to 2^{n}",
            RuntimeWarning,
        )
        n_iterations = 2**n

    iterations_per_job = max(1, n_iterations // n_jobs)

    map_reduce_job: MapReduceJob["Utility", "LeastCoreProblem"] = MapReduceJob(
        inputs=u,
        map_func=_montecarlo_least_core,
        reduce_func=_reduce_func,
        map_kwargs=dict(
            n_iterations=iterations_per_job,
            progress=progress,
        ),
        n_jobs=n_jobs,
        config=config,
    )

    return map_reduce_job()
