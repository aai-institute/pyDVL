import logging
import warnings
from typing import TYPE_CHECKING, Dict, Iterable, Tuple

import numpy as np
import scipy

from ..reporting.scores import sort_values
from ..utils import Utility, maybe_progress
from ..utils.config import ParallelConfig
from ..utils.numeric import random_powerset
from ..utils.parallel import MapReduceJob, init_parallel_backend

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


__all__ = ["montecarlo_least_core"]


def _montecarlo_least_core(
    u: Utility,
    max_iterations: int,
    *,
    progress: bool = False,
    job_id: int = 1,
    **kwargs,
) -> Tuple["NDArray", "NDArray"]:
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

    A_ub = np.zeros((max_iterations, n + 1), dtype=np.int8)
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
    progress: bool = False,
) -> Dict[str, float]:
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
    :param progress: If True, shows a tqdm progress bar
    :return: Dictionary of {"index or label": exact_value}, sorted by decreasing
        value.
    """
    n = len(u.data)

    max_iterations = min(max_iterations, 2**n)

    parallel_backend = init_parallel_backend(config)

    u_id = parallel_backend.put(u)

    iterations_per_job = max_iterations // n_jobs

    map_reduce_job: MapReduceJob["Utility", Tuple["NDArray", "NDArray"]] = MapReduceJob(
        map_func=_montecarlo_least_core,
        reduce_func=_reduce_func,
        map_kwargs=dict(
            max_iterations=iterations_per_job,
            progress=progress,
        ),
        config=config,
    )
    utility_values, A_ub = map_reduce_job(u_id, chunkify_inputs=False, n_jobs=n_jobs)[0]

    if np.any(np.isnan(utility_values)):
        warnings.warn(
            f"Calculation returned {np.sum(np.isnan(utility_values))} nan values out of {utility_values.size}",
            RuntimeWarning,
        )

    c = np.zeros(n + 1, dtype=np.int8)
    c[-1] = 1
    A_eq = np.ones((1, n + 1), dtype=np.int8)
    A_eq[:, -1] = 0
    b_ub = -utility_values
    b_eq = np.array([u(u.data.indices)])

    result = scipy.optimize.linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        method="highs",
        bounds=(None, None),
    )

    if not result.success:
        warnings.warn(
            "Could not find optimal solution. Consider increasing 'max_iterations'",
            RuntimeWarning,
        )
        return {}

    values = result.x[:-1]
    sorted_values = sort_values({u.data.data_names[i]: v for i, v in enumerate(values)})
    return sorted_values
