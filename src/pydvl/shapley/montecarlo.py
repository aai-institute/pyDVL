r"""
Monte Carlo approximations to Shapley Data values.

**Note:** You probably want to use the common interface provided by
:func:`~pydvl.shapley.compute_shapley_values` instead of using the functions in
this module.

Exact computation of Shapley value requires $\mathcal{O}(2^n)$ retrainings of
the model. Recall the definition of the value of sample $i$:

$$v_i = \frac{1}{N}  \sum_{S \subseteq D_{\backslash \{ i \}}}
\frac{1}{\binom{N - 1}{ | S | }} [U (S_{\cup \{ i \}}) - U (S)] ,$$

where $D$ is the set of indices in the training set, which we identify with the
data itself.

To overcome this limitation, it is possible to only sample some subsets of the
training set (or permutations thereof) to obtain a Monte Carlo approximation to
the true value. This is done in
:func:`~pydvl.shapley.montecarlo.combinatorial_montecarlo_shapley`. Alternatively,
employing the reformulation of the expression above as a sum over permutations,
one has the implementation in
:func:`~pydvl.shapley.montecarlo.permutation_montecarlo_shapley`.

Additionally, one can implement an early stopping strategy to
adapt computation time. This is done in
:func:`~pydvl.shapley.montecarlo.truncated_montecarlo_shapley`.

Finally, you can consider grouping your data points using
:class:`~pydvl.utils.dataset.GroupedDataset` and computing the values of the
groups instead.
"""

import logging
import math
import warnings
from collections import OrderedDict
from time import sleep, time
from typing import TYPE_CHECKING, Dict, Iterable, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from ..reporting.scores import sort_values
from ..utils import Utility, maybe_progress
from ..utils.config import ParallelConfig
from ..utils.numeric import (
    PowerSetDistribution,
    get_running_avg_variance,
    random_powerset,
)
from ..utils.parallel import MapReduceJob, init_parallel_backend
from .actor import get_shapley_coordinator, get_shapley_worker

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MonteCarloResults(NamedTuple):
    values: "NDArray"
    stderr: "NDArray"


logger = logging.getLogger(__name__)

__all__ = [
    "truncated_montecarlo_shapley",
    "permutation_montecarlo_shapley",
    "combinatorial_montecarlo_shapley",
]


def truncated_montecarlo_shapley(
    u: Utility,
    value_tolerance: Optional[float] = None,
    max_iterations: Optional[int] = None,
    n_jobs: Optional[int] = None,
    config: ParallelConfig = ParallelConfig(),
    *,
    progress: bool = False,
    coordinator_update_frequency: int = 10,
    worker_update_frequency: int = 5,
) -> Tuple["OrderedDict[str, float]", Dict[str, float]]:
    """MonteCarlo approximation to the Shapley value of data points.

    This implements the method described in:

    Ghorbani, Amirata, and James Zou. ‘Data Shapley: Equitable Valuation of Data
    for Machine Learning’. In International Conference on Machine Learning,
    2242–51. PMLR, 2019. http://proceedings.mlr.press/v97/ghorbani19c.html.

    Instead of naively implementing the expectation, we sequentially add points
    to a dataset from a permutation. We keep sampling permutations and updating
    all shapley values until the std/value score in
    the moving average falls below a given threshold (value_tolerance)
    or when the number of iterations exceeds a certain number (max_iterations).

    :param u: Utility object with model, data, and scoring function
    :param value_tolerance: Terminate if the standard deviation of the
        average value for every sample has dropped below this value
    :param max_iterations: Terminate if the total number of permutations exceeds
        this number.
    :param n_jobs: number of jobs processing permutations. If None, it will be
        set to :func:`available_cpus`.
    :param config: Object configuring parallel computation, with cluster address,
        number of cpus, etc.
    :param progress: set to `True` to use tqdm progress bars.
    :param coordinator_update_frequency: in seconds. Check status with the job
        coordinator every so often.
    :param worker_update_frequency: interval in seconds between different updates to
        and from the coordinator
    :return: Tuple with the first element being an :obj:`collections.OrderedDict`
        of approximate Shapley values for the indices, and the second being the
        estimated standard error of each value.
    """
    parallel_backend = init_parallel_backend(config)

    n_jobs = parallel_backend.effective_n_jobs(n_jobs)

    u_id = parallel_backend.put(u)

    coordinator = get_shapley_coordinator(  # type: ignore
        value_tolerance, max_iterations, progress, config=config
    )
    workers = [
        get_shapley_worker(  # type: ignore
            u=u_id,
            coordinator=coordinator,
            worker_id=worker_id,
            progress=progress,
            update_frequency=worker_update_frequency,
            config=config,
        )
        for worker_id in range(n_jobs)
    ]
    for worker_id in range(n_jobs):
        workers[worker_id].run(block=False)
    last_update_time = time()
    is_done = False
    while not is_done:
        sleep(0.01)
        if time() - last_update_time > coordinator_update_frequency:
            is_done = coordinator.check_done()
            last_update_time = time()
    dvl_values, dvl_std = coordinator.get_results()
    sorted_shapley_values = sort_values(
        {u.data.data_names[i]: v for i, v in enumerate(dvl_values)}
    )
    montecarlo_error = {u.data.data_names[i]: v for i, v in enumerate(dvl_std)}
    return sorted_shapley_values, montecarlo_error


def _permutation_montecarlo_shapley(
    u: Utility, max_permutations: int, progress: bool = False, job_id: int = 1, **kwargs
) -> "NDArray":
    """Helper function for :func:`permutation_montecarlo_shapley`.

    Computes the marginal utility of each training sample in
    :obj:`pydvl.utils.utility.Utility.data`

    :param u: Utility object with model, data, and scoring function
    :param max_permutations: total number of permutations to try
    :param progress: true to plot progress bar
    :param job_id: id to use for reporting progress
    :return: a matrix with each row being a different permutation
        and each column being the score of a different data point
    """
    n = len(u.data)
    values = np.zeros(shape=(max_permutations, n))
    pbar = maybe_progress(max_permutations, progress, position=job_id)
    for iter_idx in pbar:
        prev_score = 0.0
        permutation = np.random.permutation(u.data.indices)
        marginals = np.zeros(shape=n)
        for i, idx in enumerate(permutation):
            score = u(permutation[: i + 1])
            marginals[idx] = score - prev_score
            prev_score = score
        values[iter_idx] = marginals
    return values


def permutation_montecarlo_shapley(
    u: Utility,
    max_iterations: int,
    n_jobs: int,
    config: ParallelConfig = ParallelConfig(),
    *,
    progress: bool = False,
) -> Tuple["OrderedDict[str, float]", Dict[str, float]]:
    """Computes an approximate Shapley value using independent index permutations.

    :param u: Utility object with model, data, and scoring function
    :param max_iterations: total number of iterations (permutations) to use
    :param n_jobs: number of jobs across which to distribute the computation.
    :param config: Object configuring parallel computation, with cluster address,
        number of cpus, etc.
    :param progress: Set to True to print a progress bar.
    :return: Tuple with the first element being an ordered Dict of approximate
        Shapley values for the indices, the second being their standard error
    """
    parallel_backend = init_parallel_backend(config)

    u_id = parallel_backend.put(u)

    iterations_per_job = max_iterations // n_jobs

    map_reduce_job: MapReduceJob["NDArray", "NDArray"] = MapReduceJob(
        map_func=_permutation_montecarlo_shapley,
        reduce_func=np.concatenate,  # type: ignore
        map_kwargs=dict(max_permutations=iterations_per_job, progress=progress),
        reduce_kwargs=dict(axis=0),
        config=config,
        chunkify_inputs=False,
        n_jobs=n_jobs,
    )
    full_results = map_reduce_job(u_id)[0]

    # Careful: for some models there might be nans, e.g. for i=0 or i=1!
    if np.any(np.isnan(full_results)):
        warnings.warn(
            f"Calculation returned {np.sum(np.isnan(full_results))} nan values "
            f"out of {full_results.size}",
            RuntimeWarning,
        )
    acc = np.nanmean(full_results, axis=0)
    acc_std = np.nanstd(full_results, axis=0) / np.sqrt(full_results.shape[0])
    sorted_shapley_values = sort_values(
        {u.data.data_names[i]: v for i, v in enumerate(acc)}
    )
    montecarlo_error = {u.data.data_names[i]: v for i, v in enumerate(acc_std)}
    return sorted_shapley_values, montecarlo_error


def _combinatorial_montecarlo_shapley(
    indices: Sequence[int],
    u: Utility,
    max_iterations: int,
    dist: PowerSetDistribution,
    *,
    progress: bool = False,
    job_id: int = 1,
    **kwargs,
) -> MonteCarloResults:
    """Helper function for :func:`combinatorial_montecarlo_shapley`.

    This is the code that is sent to workers to compute values using the
    combinatorial definition.

    :param u: Utility object with model, data, and scoring function
    :param max_iterations: total number of subsets to sample.
    :param dist: Distribution to use of sets over the power set.
    :param progress: true to plot progress bar
    :param job_id: id to use for reporting progress
    :return: A tuple of ndarrays with estimated values and standard errors
    """
    n = len(u.data)

    if len(np.unique(indices)) != len(indices):
        raise ValueError("Repeated indices passed")

    # FIXME: is this ok?
    if dist is PowerSetDistribution.WEIGHTED:
        correction = 2 ** (n - 1) / n
    else:
        raise NotImplementedError(
            f"Correction for sampling distribution {dist=} not implemented"
        )

    values = np.zeros(n)
    variances = np.zeros(n)
    counts = np.zeros(n)
    pbar = maybe_progress(indices, progress, position=job_id)
    for idx in pbar:
        # Randomly sample subsets of full dataset without idx
        subset = np.setxor1d(u.data.indices, [idx], assume_unique=True)
        power_set = random_powerset(
            subset,
            dist=dist,
            max_subsets=max_iterations,
        )
        for s in maybe_progress(
            power_set,
            progress,
            desc=f"Index {idx}",
            total=max_iterations,
            position=job_id,
        ):
            new_marginal = (u({idx}.union(s)) - u(s)) / math.comb(n - 1, len(s))
            if np.isnan(new_marginal):
                continue
            values[idx], variances[idx] = get_running_avg_variance(
                values[idx], variances[idx], new_marginal, counts[idx]
            )
            counts[idx] += 1

    return MonteCarloResults(
        values=correction * values,
        stderr=correction**2 * variances / np.sqrt(np.maximum(1, counts)),
    )


def combinatorial_montecarlo_shapley(
    u: Utility,
    max_iterations: int,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    *,
    dist: PowerSetDistribution = PowerSetDistribution.WEIGHTED,
    progress: bool = False,
) -> Tuple["OrderedDict[str, float]", Dict[str, float]]:
    """Computes an approximate Shapley value using the combinatorial definition.

    :param u: utility
    :param max_iterations: total number of iterations (permutations) to use
    :param n_jobs: number of jobs across which to distribute the computation
    :param config: Object configuring parallel computation, with cluster
        address, number of cpus, etc.
    :param dist: Distribution to use of sets over the power set.
    :param progress: true to plot progress bar
    :return: Tuple with the first element being an ordered Dict of approximate
        Shapley values for the indices, the second being their standard error
    """
    parallel_backend = init_parallel_backend(config)
    u_id = parallel_backend.put(u)
    iterations_per_job = max_iterations // n_jobs

    def reducer(results_it: Iterable[MonteCarloResults]) -> MonteCarloResults:
        values = np.zeros(len(u.data))
        stderr = np.zeros_like(values)

        # non-zero indices in results are disjoint by construction, so it is ok
        # to add them
        for val, std in results_it:
            values += val
            stderr += std
        return MonteCarloResults(values=values, stderr=stderr)

    map_reduce_job: MapReduceJob["NDArray", MonteCarloResults] = MapReduceJob(
        map_func=_combinatorial_montecarlo_shapley,
        reduce_func=reducer,
        map_kwargs=dict(
            u=u_id,
            dist=dist,
            max_iterations=iterations_per_job,
            progress=progress,
        ),
        chunkify_inputs=True,
        n_jobs=n_jobs,
        config=config,
    )
    results = map_reduce_job(u.data.indices)[0]
    sorted_shapley_values = sort_values(
        {u.data.data_names[i]: v for i, v in enumerate(results.values)}
    )
    montecarlo_errors = {u.data.data_names[i]: v for i, v in enumerate(results.stderr)}

    return sorted_shapley_values, montecarlo_errors
