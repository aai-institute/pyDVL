import logging
import warnings
from typing import Iterable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from pydvl.utils import Utility, maybe_progress
from pydvl.utils.config import ParallelConfig
from pydvl.utils.numeric import random_powerset
from pydvl.utils.parallel import MapReduceJob, init_parallel_backend
from pydvl.value.least_core._common import _solve_linear_programming
from pydvl.value.results import ValuationResult, ValuationStatus

logger = logging.getLogger(__name__)


__all__ = ["montecarlo_least_core"]


def _montecarlo_least_core(
    u: Utility,
    max_iterations: int,
    *,
    progress: bool = False,
    job_id: int = 1,
    **kwargs,
) -> Tuple[NDArray, NDArray]:
    """Computes utility values and the Least Core upper bound matrix for a given number of iterations.

    :param u: Utility object with model, data, and scoring function
    :param max_iterations: total number of iterations to use
    :param progress: If True, shows a tqdm progress bar
    :param job_id: Integer id used to determine the position of the progress bar
    :return:
    """
    n = len(u.data)

    utility_values = np.zeros(max_iterations)

    # Randomly sample subsets of full dataset
    power_set = random_powerset(
        u.data.indices,
        max_subsets=max_iterations,
    )

    A_ub = np.zeros((max_iterations, n + 1))
    A_ub[:, -1] = -1

    for i, subset in enumerate(
        maybe_progress(
            power_set,
            progress,
            total=max_iterations,
            position=job_id,
        )
    ):
        indices = np.zeros(n + 1, dtype=bool)
        indices[list(subset)] = True
        A_ub[i, indices] = -1
        utility_values[i] = u(subset)

    return utility_values, A_ub


def _reduce_func(
    results: Iterable[Tuple["NDArray", "NDArray"]]
) -> Tuple["NDArray", "NDArray"]:
    """Combines the results from different parallel runs of the `_montecarlo_least_core` function"""
    utility_values_list, A_ub_list = zip(*results)
    utility_values = np.concatenate(utility_values_list)
    A_ub = np.concatenate(A_ub_list)
    return utility_values, A_ub


def montecarlo_least_core(
    u: Utility,
    max_iterations: int,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    *,
    epsilon: float = 0.01,
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
    * $m$ is the number of subsets that will be sampled and whose utility will be computed
      and used to compute the Least Core values.

    :param u: Utility object with model, data, and scoring function
    :param max_iterations: total number of iterations to use
    :param n_jobs: number of jobs across which to distribute the computation
    :param config: Object configuring parallel computation, with cluster
        address, number of cpus, etc.
    :param epsilon: Relaxation value by which the subset utility is decreased.
    :param options: LP Solver options. \
        Refer to this page for more information https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html
    :param progress: If True, shows a tqdm progress bar
    :return: Dictionary of {"index or label": exact_value}, sorted by decreasing
        value.
    """
    n = len(u.data)

    max_iterations = min(max_iterations, 2**n)

    if max_iterations < n:
        raise ValueError(
            "Number of iterations should be greater than the size of the dataset"
        )

    if options is None:
        options = {}

    parallel_backend = init_parallel_backend(config)

    u_id = parallel_backend.put(u)

    iterations_per_job = max(1, max_iterations // n_jobs)

    logger.debug("Instantiating MapReduceJob")
    map_reduce_job: MapReduceJob["Utility", Tuple["NDArray", "NDArray"]] = MapReduceJob(
        map_func=_montecarlo_least_core,
        reduce_func=_reduce_func,
        map_kwargs=dict(
            max_iterations=iterations_per_job,
            progress=progress,
        ),
        config=config,
    )
    logger.debug("Calling MapReduceJob instance")
    utility_values, A_ub = map_reduce_job(u_id, chunkify_inputs=False, n_jobs=n_jobs)[0]

    if np.any(np.isnan(utility_values)):
        warnings.warn(
            f"Calculation returned {np.sum(np.isnan(utility_values))} nan values out of {utility_values.size}",
            RuntimeWarning,
        )

    logger.debug("Building vectors and matrices for linear programming problem")
    c = np.zeros(n + 1)
    c[-1] = 1
    b_ub = -(utility_values - epsilon)
    # We explicitly add the utility for the empty set
    A_ub = np.concatenate([np.zeros((1, n + 1)), A_ub], axis=0)
    A_ub[0, -1] = -1
    b_ub = np.concatenate([[u([])], b_ub])
    A_eq = np.ones((1, n + 1))
    A_eq[:, -1] = 0
    # We explicitly add the utility value for the entire dataset
    b_eq = np.array([u(u.data.indices)])

    logger.debug("Removing possible duplicate values in upper bound array")
    A_ub, unique_indices = np.unique(A_ub, return_index=True, axis=0)
    b_ub = b_ub[unique_indices]

    logger.debug(f"{unique_indices=}")

    values = _solve_linear_programming(
        c, A_eq, b_eq, A_ub, b_ub, bounds=[(None, None)] * n + [(0.0, None)], **options
    )

    if values is None:
        logger.debug("No values were found")
        status = ValuationStatus.Failed
        values = np.empty(n)
        values[:] = np.nan
    else:
        status = ValuationStatus.Converged

    return ValuationResult(
        algorithm="exact_least_core",
        status=status,
        values=values,
        stderr=None,
        data_names=u.data.data_names,
    )
