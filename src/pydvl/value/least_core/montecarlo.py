import logging
import warnings
from typing import Iterable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from pydvl.utils import Utility, maybe_progress
from pydvl.utils.config import ParallelConfig
from pydvl.utils.numeric import random_powerset
from pydvl.utils.parallel import MapReduceJob
from pydvl.value.least_core._common import (
    _solve_egalitarian_least_core_quadratic_program,
    _solve_least_core_linear_program,
)
from pydvl.value.results import ValuationResult, ValuationStatus

logger = logging.getLogger(__name__)


__all__ = ["montecarlo_least_core"]


def _montecarlo_least_core(
    u: Utility,
    n_iterations: int,
    *,
    progress: bool = False,
    job_id: int = 1,
    **kwargs,
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
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

    return utility_values, A_lb


def _reduce_func(
    results: Iterable[Tuple[NDArray[np.float_], NDArray[np.float_]]]
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    """Combines the results from different parallel runs of the `_montecarlo_least_core` function"""
    utility_values_list, A_lb_list = zip(*results)
    utility_values = np.concatenate(utility_values_list)
    A_lb = np.concatenate(A_lb_list)
    return utility_values, A_lb


def montecarlo_least_core(
    u: Utility,
    n_iterations: int,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    *,
    epsilon: float = 0.0,
    options: Optional[dict] = None,
    progress: bool = False,
) -> ValuationResult:
    r"""Computes approximate Least Core values using a Monte Carlo approach.

    $$
    \begin{array}{lll}
    \text{minimize} & \displaystyle{e} & \\
    \text{subject to} & \displaystyle\sum_{i\in N} x_{i} = v(N) & \\
    & \displaystyle\sum_{i\in S} x_{i} + e + \epsilon \geq v(S) & ,
    \forall S \in \{S_1, S_2, \dots, S_m \overset{\mathrm{iid}}{\sim} U(2^N) \}
    \end{array}
    $$

    Where:

    * $U(2^N)$ is the uniform distribution over the powerset of $N$.
    * $m$ is the number of subsets that will be sampled and whose utility will be computed
      and used to compute the data values.
    * $\epsilon \ge 0$ is an optional relaxation value.

    :param u: Utility object with model, data, and scoring function
    :param n_iterations: total number of iterations to use
    :param n_jobs: number of jobs across which to distribute the computation
    :param config: Object configuring parallel computation, with cluster
        address, number of cpus, etc.
    :param epsilon: Relaxation value by which the subset utility is decreased.
    :param options: Keyword arguments that will be used to select a solver
        and to configure it. Refer to the following page for all possible options:
        https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options
    :param progress: If True, shows a tqdm progress bar
    :return: Object with the data values and the least core value.
    """
    n = len(u.data)

    if n_iterations < n:
        raise ValueError(
            "Number of iterations should be greater than the size of the dataset"
        )

    if n_iterations > 2**n:
        warnings.warn(
            f"Passed n_iterations is greater than the number subsets! Setting it to 2^{n}",
            RuntimeWarning,
        )
        n_iterations = 2**n

    if options is None:
        options = {}

    iterations_per_job = max(1, n_iterations // n_jobs)

    logger.debug("Instantiating MapReduceJob")
    map_reduce_job: MapReduceJob["Utility", Tuple["NDArray", "NDArray"]] = MapReduceJob(
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
    logger.debug("Calling MapReduceJob instance")
    utility_values, A_lb = map_reduce_job()

    if np.any(np.isnan(utility_values)):
        warnings.warn(
            f"Calculation returned {np.sum(np.isnan(utility_values))} nan values out of {utility_values.size}",
            RuntimeWarning,
        )

    logger.debug("Building vectors and matrices for linear programming problem")
    b_lb = utility_values
    A_eq = np.ones((1, n))
    # We explicitly add the utility value for the entire dataset
    b_eq = np.array([u(u.data.indices)])

    logger.debug("Removing possible duplicate values in lower bound array")
    A_lb, unique_indices = np.unique(A_lb, return_index=True, axis=0)
    b_lb = b_lb[unique_indices]

    _, subsidy = _solve_least_core_linear_program(
        A_eq=A_eq, b_eq=b_eq, A_lb=A_lb, b_lb=b_lb, epsilon=epsilon, **options
    )

    values: Optional[NDArray[np.float_]]

    if subsidy is None:
        logger.debug("No values were found")
        status = ValuationStatus.Failed
        values = np.empty(n)
        values[:] = np.nan
        subsidy = np.nan

        return ValuationResult(
            algorithm="montecarlo_least_core",
            status=status,
            values=values,
            subsidy=subsidy,
            stderr=None,
            data_names=u.data.data_names,
        )

    values = _solve_egalitarian_least_core_quadratic_program(
        subsidy,
        A_eq=A_eq,
        b_eq=b_eq,
        A_lb=A_lb,
        b_lb=b_lb,
        **options,
    )

    if values is None:
        logger.debug("No values were found")
        status = ValuationStatus.Failed
        values = np.empty(n)
        values[:] = np.nan
        subsidy = np.nan
    else:
        status = ValuationStatus.Converged

    return ValuationResult(
        algorithm="montecarlo_least_core",
        status=status,
        values=values,
        subsidy=subsidy,
        stderr=None,
        data_names=u.data.data_names,
    )
