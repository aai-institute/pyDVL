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

import numpy as np
import scipy as sp
from numpy.typing import NDArray

from pydvl.utils import MapReduceJob, ParallelConfig, Utility, maybe_progress
from pydvl.utils.numeric import random_subset_of_size
from pydvl.value import ValuationResult, ValuationStatus

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

    log.warning("This may be bogus. Do not use.")
    constants = _constants(n=n, epsilon=eps, delta=delta, utility_range=utility_range)
    return int(constants.T)


def _build_gt_constraints(
    n: int, bound: float, C: NDArray[np.float_]
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    r"""Builds a matrix and vector modelling the pairwise constraints:

    $$ | s_i - s_j - C_{i j}| \leq \rho, $$

    where $\rho = \epsilon/(2 \sqrt{N})$ in the paper and

    $$C_{i j} = \frac{Z}{T} \sum_{t=1}^T U(S_t) (\beta_{t i} - \beta_{t j}),$$

    for $i, j \in [N], j \gte i$.

    For every $i$, each such constraint is converted into two:

    $$ s_i - s_j \leq \rho + C_{i j}, $$
    $$ s_j - s_i \leq \rho - C_{i j}, $$

    and there are $N-i$ of these, for $j \in \{i, i+1, ..., N\}$. We build
    matrices $A^{(i)} \in \mathbb{R}^{N-i \times N}$ representing the first set
    of constraints $\leq \rho + C_{i j}$ for every $j \geq i$ and stack them
    vertically.

    The second set of constraints is just the negation of the first.

    :param n: Number of samples
    :param bound: Upper bound, typically $\rho = \epsilon/(2 \sqrt{N})$
    :param C: Monte Carlo estimate of the pair-wise differences of values
    :return: Constraint matrix ``A_ub`` and vector ``b_ub`` for
        ``sp.optimize.linprog``
    """
    A = -np.identity(n)
    A[:, 0] = 1
    A = A[1:, :]
    assert A.shape == (n - 1, n)

    assert C.shape == (n, n)
    c = C[0, 1:]
    assert c.shape == (n - 1,)

    def rshift(M: NDArray, copy: bool = True):
        if copy:
            M = np.copy(M)
        M[:, 1:] = M[:, :-1]
        M[:, 0] = 0
        return M

    chunk = A.copy()
    for k in range(1, n - 1):
        chunk = rshift(chunk, copy=True)[:-1]
        A = np.row_stack((A, chunk))
        c = np.concatenate((c, C[k, k + 1 :]))

    assert A.shape[0] == c.shape[0]

    # Append lower bound constraints:
    A = np.row_stack((A, -A))
    c = np.concatenate((c, -c))

    return A, bound + c


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
    :return: a matrix with each row being a different permutation and each
        column being the score of a different data point
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
    eps: float,
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
    :param eps: Epsilon in the (ε,δ) sample bound. Use the same as for the
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
        epsilon=eps,
        delta=0.05,
        utility_range=u.score_range.max() - u.score_range.min(),
    )
    T = n_iterations
    if T < const.T:
        log.warning(
            f"max iterations of {T} are below the required {const.T} for the "
            f"ε={eps:.02f} guarantee at .95 probability"
        )

    iterations_per_job = max(1, n_iterations // n_jobs)

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

    ###########################################################################
    # Sanity check: Another way of building the constraints
    # CC = np.zeros(shape=(n, n))
    # for t in range(T):
    #     # A matrix with n columns copies of beta[t]:
    #     Bt = np.repeat(betas[t], repeats=n).reshape((n, n))
    #     # C_ij += u_t * (β_i - β_j)
    #     CC += uu[t] * (Bt - Bt.T)
    # CC *= Z / T
    # assert np.allclose(np.triu(C), np.triu(CC))

    ###########################################################################
    # Solution of the constraint problem with scipy

    A_ub, b_ub = _build_gt_constraints(n, bound=eps / (2 * np.sqrt(n)), C=C)
    c = np.zeros_like(u.data.indices)
    total_utility = u(u.data.indices)
    # A trivial bound for the values from the definition is max_utility * (n-1)
    bounds = tuple(u.score_range * n)  # u.score_range defaults to (-inf, inf)
    result: sp.optimize.OptimizeResult = sp.optimize.linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=np.ones((1, n)),
        b_eq=total_utility,
        bounds=bounds,
        method="highs",
        options={},
    )

    ###########################################################################
    # Sanity check: Solution of the constraint problem with cvxpy
    #
    # import cvxpy as cp
    #
    # v = cp.Variable(n)
    # constraints = [cp.sum(v) == total_utility]
    # for i in range(n):
    #     for j in range(i + 1, n):
    #         constraints.append(v[i] - v[j] <= eps + C[i, j])
    #         constraints.append(v[j] - v[i] <= eps - C[i, j])
    #
    # cp_result = cp.Problem(cp.Minimize(0), constraints).solve(solver=cp.SCS)
    #
    # if not np.allclose(result.x, v.value, rtol=0.01):
    #     diff = result.x - v.value
    #     raise RuntimeError(
    #         f"Mismatch > 1% between the solutions by scipy and cvxpy: "
    #         f"mean={diff.mean()}, stdev={diff.std()}"
    #     )
    ###########################################################################

    if result.status == 0:
        return ValuationResult(
            algorithm="group_testing_shapley",
            status=ValuationStatus.Converged,
            values=result.x,
            stderr=None,
            data_names=u.data.data_names,
        )
    else:
        return ValuationResult(
            algorithm="group_testing_shapley",
            status=ValuationStatus.Failed,
            values=np.nan * np.ones_like(u.data.indices),
            stderr=None,
            data_names=u.data.data_names,
        )
