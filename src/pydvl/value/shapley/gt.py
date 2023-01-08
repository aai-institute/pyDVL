"""
This module implements Group Testing for the approximation of Shapley values, as
introduced in :footcite:t:`jia_efficient_2019`. The sampling of index subsets is
done in such a way that an approximation to the true Shapley values can be
computed with guarantees.

.. warning::
   This module is still experimental and should not be used in production.

"""
import logging
from typing import Tuple, TypeVar, cast

import numpy as np
import scipy as sp
from numpy.typing import NDArray

from pydvl.utils import ParallelConfig, Utility, maybe_progress
from pydvl.utils.numeric import random_subset_of_size
from pydvl.value import ValuationResult, ValuationStatus

__all__ = ["group_testing_shapley", "num_samples_eps_delta"]

log = logging.getLogger(__name__)


def num_samples_eps_delta(eps: float, delta: float, n: int, u_range: float) -> int:
    r"""Implements the formula in Theorem 3 of :footcite:t:`jia_efficient_2019`
    which gives a lower bound on the number of samples required to obtain an
    (ε/√n,δ/(N(N-1))-approximation to all pair-wise differences of Shapley values, wrt.
    $\ell_2$ norm.

    .. warning::
       There is something wrong with the computation. Do not use this function.

    :param eps:
    :param delta:
    :param n: Number of samples
    :param u_range: Range of the :class:`~pydvl.utils.utility.Utility` function
    :return: Number of samples from $2^{[n]}$ guaranteeing ε/√n-correct Shapley
        pair-wise differences of values with probability 1-δ/(N(N-1)).

    """

    log.warning("This bound is bogus. Do not use.")

    T = TypeVar("T", NDArray[np.float_], float)

    def h(u: T) -> T:
        return cast(T, (1 + u) * np.log(1 + u) - u)

    # FIXME: avoid duplication
    kk = np.arange(1, n)
    Z = 2 * (1.0 / kk).sum()
    qq = (1 / kk + (1 / (n - kk))) / Z

    kk = np.arange(2, n)
    q_tot = (n - 2) / n * qq[0] + np.inner(
        qq[1:], 1 + (2 * kk * (kk - n)) / (n * (n - 1))
    )
    bound = 8 * np.log(n * (n - 1) / (2 * delta))
    bound /= (1 - q_tot**2) * h(eps / (Z * u_range * np.sqrt(n) * (1 - q_tot**2)))
    return int(bound)


def _build_gt_constraints(
    n: int, bound: float, C: NDArray[np.float_]
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    r"""Builds a matrix and vector modelling the pairwise constraints:

    $$ | s_i - s_j - C_{i j}| \leq \delta, $$

    where $\delta = \epsilon/(2 \sqrt{N})$ in the paper and

    $$C_{i j} = \frac{Z}{T} \sum_{t=1}^T U(S_t) (\beta_{t i} - \beta_{t j}),$$

    for $i \leq j \in [N]$.

    For every $i$, each such constraint is converted into two:

    $$ s_i - s_j \leq \delta + C_{i j}, $$
    $$ s_j - s_i \leq \delta - C_{i j}, $$

    and there are $N-i$ of these, for $j \in \{i, i+1, ..., N\}$. We build
    matrices $A^{(i)} \in \mathbb{R}^{N-i \times N}$ representing the first set
    of constraints $\leq \delta + C_{i j}$ for every $j \geq i$ and stack them
    vertically.

    The second set of constraints is just the negation of the first.

    :param n: Number of samples
    :param bound: Upper bound, typically $\delta = \epsilon/(2 \sqrt{N})$
    :param C: Monte Carlo estimate of the pair-wise value differences
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

    assert (
        A.shape[0] == c.shape[0]
    ), f"Shape mismatch between A: {A.shape} and c {c.shape}"

    # Append lower bound constraints:
    A = np.row_stack((A, -A))
    c = np.concatenate((c, -c))

    return A, bound + c


def group_testing_shapley(
    u: Utility,
    max_iterations: int,
    *,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    progress: bool = False,
) -> ValuationResult:
    """Implements group testing for approximation of Shapley values as described
    in :footcite:t:`jia_efficient_2019`.

    By picking a specific distribution over subsets, the differences in Shapley
    values can be approximated with a Monte Carlo sum. These are then used to
    solve for the individual values in a feasibility problem.

    .. todo::
       Document this properly


    :param u: Utility object with model, data, and scoring function
    :param max_iterations: Number of tests to perform
    :param n_jobs: Number of parallel jobs to use. Each worker performs a chunk
        of all tests (i.e. utility evaluations).
    :param config: Object configuring parallel computation, with cluster
        address, number of cpus, etc.
    :param progress: Whether to display progress bars for each job.
    :return: Object with the data values.

    .. versionadded:: 0.4.0

    """
    rng = np.random.default_rng()

    indices = u.data.indices
    n = len(indices)
    kk = np.arange(1, n)  # sample sizes
    Z = 2 * (1.0 / kk).sum()
    qq = (1 / kk + (1 / (n - kk))) / Z

    T = max_iterations
    betas = np.zeros(shape=(T, n), dtype=np.int_)
    uu = []  # utilities

    for t in maybe_progress(T, progress=progress):  # TODO: parallelize here
        k = rng.choice(kk, size=1, p=qq).item()
        s = random_subset_of_size(indices, k)
        uu.append(u(s))
        betas[t][s] = 1

    # Matrix of estimated differences. See Eqs. (3) and (4) in the paper.
    C = np.zeros(shape=(n, n))
    for t in range(T):  # FIXME: maybe vectorise and avoid building Bt
        # A matrix with n columns copies of beta[t]:
        Bt = np.repeat(betas[t], repeats=n).reshape((n, n))
        # C_ij += u_t * (β_i - β_j)
        C += uu[t] * (Bt - Bt.T)
    C *= Z / T

    # Sanity check:
    # CC = np.zeros(shape=(n, n))
    # for i in range(n):
    #     for j in range(i + 1, n):
    #         CC[i, j] = np.sum([uu[t] * (betas[t][i] - betas[t][j]) for t in range(T)])
    # CC *= Z / T
    # assert np.allclose(np.triu(C), np.triu(CC))

    eps = 0.01  # FIXME: make into parameter

    total_utility = u(indices)

    ###############################################
    # Solution of the constraint problem with scipy

    A_ub, b_ub = _build_gt_constraints(n, bound=eps / (2 * np.sqrt(n)), C=C)

    result: sp.optimize.OptimizeResult = sp.optimize.linprog(
        np.zeros_like(u.data.indices),
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=np.ones((1, len(u.data.indices))),
        b_eq=total_utility,
        bounds=(-np.inf, np.inf),  # FIXME: use range of utility
        method="highs",
        options={},
    )

    ###############################################
    # Solution of the constraint problem with cvxpy

    import cvxpy as cp

    v = cp.Variable(n)
    constraints = [cp.sum(v) == total_utility]
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append(v[i] - v[j] <= eps + C[i, j])
            constraints.append(v[j] - v[i] <= eps - C[i, j])

    cp_result = cp.Problem(cp.Minimize(0), constraints).solve(solver=cp.SCS)

    if not np.allclose(result.x, v.value, rtol=0.01):
        diff = result.x - v.value
        raise RuntimeError(
            f"Mismatch > 1% between the solutions by scipy and cvxpy: "
            f"mean={diff.mean()}, stdev={diff.std()}"
        )

    # cut
    ###############################################

    return ValuationResult(
        algorithm="group_testing_shapley",
        status=ValuationStatus.Converged
        if result.status == 0
        else ValuationStatus.Failed,
        values=result.x,
        stderr=None,
        data_names=u.data.data_names,
    )
