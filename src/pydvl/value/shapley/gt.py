"""
This module implements Group Testing for the approximation of Shapley values, as
introduced in :footcite:t:`jia_efficient_2019`. The sampling of index subsets is
done in such a way that an approximation to the true Shapley values can be
computed with guarantees.

.. warning::
   This method is extremely inefficient. Potential improvements to the
   implementation notwithstanding, convergence seems to be very slow (in terms
   of evaluations of the utility required). We recommend other Monte Carlo
   methods instead.

You can read more :ref:`in the documentation<data valuation>`.

.. versionadded:: 0.4.0

"""
import logging
from collections import namedtuple
from typing import Iterable, Tuple, TypeVar, cast

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from pydvl.utils import MapReduceJob, ParallelConfig, Utility, maybe_progress
from pydvl.utils.numeric import random_subset_of_size
from pydvl.utils.parallel.backend import effective_n_jobs
from pydvl.utils.status import Status
from pydvl.value import ValuationResult

__all__ = ["group_testing_shapley", "num_samples_eps_delta"]

log = logging.getLogger(__name__)

T = TypeVar("T", NDArray[np.float_], float)
GTConstants = namedtuple("GTConstants", ["kk", "Z", "q", "q_tot", "T"])


def _constants(
    n: int, epsilon: float, delta: float, utility_range: float
) -> GTConstants:
    """A helper function returning the constants for the algorithm. Pretty ugly,
    yes.

    """
    r = utility_range

    kk = np.arange(1, n)  # sample sizes
    Z = 2 * (1.0 / kk).sum()
    q = (1 / kk + 1 / (n - kk)) / Z
    q_tot = (n - 2) / n * q[0] + np.inner(
        q[1:], 1 + 2 * kk[1:] * (kk[1:] - n) / (n * (n - 1))
    )

    def h(u: T) -> T:
        return cast(T, (1 + u) * np.log(1 + u) - u)

    # The implementation in GitHub defines a different bound:
    # T_code = int( 4
    #     / (1 - q_tot**2)
    #     / h(2 * epsilon / Z / r / (1 - q_tot**2))
    #     * np.log(n * (n - 1) / (2 * delta))
    # )
    if r == np.inf:
        log.warning(
            "Group Testing: cannot estimate minimum number of iterations for "
            "unbounded utilities. Please specify a range in the Scorer if "
            "you need this estimate."
        )
        min_iter = -1
    else:
        min_iter = 8 * np.log(n * (n - 1) / (2 * delta)) / (1 - q_tot**2)
        min_iter /= h(2 * epsilon / (np.sqrt(n) * Z * r * (1 - q_tot**2)))

    return GTConstants(kk=kk, Z=Z, q=q, q_tot=q_tot, T=int(min_iter))


def num_samples_eps_delta(
    eps: float, delta: float, n: int, utility_range: float
) -> int:
    r"""Implements the formula in Theorem 3 of :footcite:t:`jia_efficient_2019`
    which gives a lower bound on the number of samples required to obtain an
    (ε/√n,δ/(N(N-1))-approximation to all pair-wise differences of Shapley
    values, wrt. $\ell_2$ norm.

    :param eps: ε
    :param delta: δ
    :param n: Number of samples
    :param utility_range: Range of the :class:`~pydvl.utils.utility.Utility`
        function
    :return: Number of samples from $2^{[n]}$ guaranteeing ε/√n-correct Shapley
        pair-wise differences of values with probability 1-δ/(N(N-1)).

    .. versionadded:: 0.4.0

    """
    constants = _constants(n=n, epsilon=eps, delta=delta, utility_range=utility_range)
    return int(constants.T)


def _group_testing_shapley(
    u: Utility, n_iterations: int, progress: bool = False, job_id: int = 1
):
    """Helper function for :func:`group_testing_shapley`.

    Computes utilities of sets sampled using the strategy for estimating the
    differences in Shapley values.

    :param u: Utility object with model, data, and scoring function
    :param n_iterations: total number of permutations to use
    :param progress: Whether to display progress bars for each job.
    :param job_id: id to use for reporting progress (e.g. to place
    progres bars)
    :return:
    """
    rng = np.random.default_rng()
    n = len(u.data.indices)
    const = _constants(n, 1, 1, 1)  # don't care about eps,delta,range

    betas = np.zeros(shape=(n_iterations, n), dtype=np.int_)  # indicator vars
    uu = np.empty(n_iterations)  # utilities

    for t in maybe_progress(n_iterations, progress=progress, position=job_id):
        k = rng.choice(const.kk, size=1, p=const.q).item()
        s = random_subset_of_size(u.data.indices, k)
        uu[t] = u(s)
        betas[t, s] = 1
    return uu, betas


def group_testing_shapley(
    u: Utility,
    n_iterations: int,
    epsilon: float,
    delta: float,
    *,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    progress: bool = False,
) -> ValuationResult:
    """Implements group testing for approximation of Shapley values as described
    in :footcite:t:`jia_efficient_2019`.

    .. warning::
       This method is extremely inefficient. It requires several orders of
       magnitude more evaluations of the utility than others in
       :mod:`~pydvl.value.shapley.montecarlo`. It also uses several intermediate
       objects like the results from the runners and the constraint matrices
       which can become rather large.

    By picking a specific distribution over subsets, the differences in Shapley
    values can be approximated with a Monte Carlo sum. These are then used to
    solve for the individual values in a feasibility problem.

    :param u: Utility object with model, data, and scoring function
    :param n_iterations: Number of tests to perform. Use
        :func:`num_samples_eps_delta` to estimate this.
    :param epsilon: From the (ε,δ) sample bound. Use the same as for the
        estimation of ``n_iterations``.
    :param delta: From the (ε,δ) sample bound. Use the same as for the
        estimation of ``n_iterations``.
    :param n_jobs: Number of parallel jobs to use. Each worker performs a chunk
        of all tests (i.e. utility evaluations).
    :param config: Object configuring parallel computation, with cluster
        address, number of cpus, etc.
    :param progress: Whether to display progress bars for each job.
    :return: Object with the data values.

    .. versionadded:: 0.4.0

    """

    n = len(u.data.indices)
    const = _constants(
        n=n,
        epsilon=epsilon,
        delta=delta,
        utility_range=u.score_range.max() - u.score_range.min(),
    )
    T = n_iterations
    if T < const.T:
        log.warning(
            f"max iterations of {T} are below the required {const.T} for the "
            f"ε={epsilon:.02f} guarantee at δ={1 - delta:.02f} probability"
        )

    iterations_per_job = max(1, n_iterations // effective_n_jobs(n_jobs, config))

    def reducer(
        results_it: Iterable[Tuple[NDArray, NDArray]]
    ) -> Tuple[NDArray, NDArray]:
        return np.concatenate(list(x[0] for x in results_it)).astype(
            np.float_
        ), np.concatenate(list(x[1] for x in results_it)).astype(np.int_)

    map_reduce_job: MapReduceJob[Utility, Tuple[NDArray, NDArray]] = MapReduceJob(
        u,
        map_func=_group_testing_shapley,
        reduce_func=reducer,
        map_kwargs=dict(n_iterations=iterations_per_job, progress=progress),
        config=config,
        n_jobs=n_jobs,
    )
    uu, betas = map_reduce_job()

    # Matrix of estimated differences. See Eqs. (3) and (4) in the paper.
    C = np.zeros(shape=(n, n))
    for i in range(n):
        for j in range(i + 1, n):
            C[i, j] = np.dot(uu, betas[:, i] - betas[:, j])
    C *= const.Z / T
    total_utility = u(u.data.indices)

    ###########################################################################
    # Solution of the constraint problem with cvxpy

    v = cp.Variable(n)
    constraints = [cp.sum(v) == total_utility]
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append(v[i] - v[j] <= epsilon + C[i, j])
            constraints.append(v[j] - v[i] <= epsilon - C[i, j])

    problem = cp.Problem(cp.Minimize(0), constraints)
    problem.solve(solver=cp.SCS)

    if problem.status != "optimal":
        log.warning(f"cvxpy returned status {problem.status}")
        values = (
            np.nan * np.ones_like(u.data.indices)
            if not hasattr(v.value, "__len__")
            else v.value
        )
        return ValuationResult(
            algorithm="group_testing_shapley",
            status=Status.Failed,
            values=values,
            data_names=u.data.data_names,
            solver_status=problem.status,
        )

    return ValuationResult(
        algorithm="group_testing_shapley",
        status=Status.Converged,
        values=v.value,
        data_names=u.data.data_names,
    )
