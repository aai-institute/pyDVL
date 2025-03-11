"""
This module implements Group Testing for the approximation of Shapley values, as
introduced in (Jia, R. et al., 2019)[^1]. The sampling of index subsets is
done in such a way that an approximation to the true Shapley values can be
computed with guarantees.

!!! Warning
    This method is very inefficient. Potential improvements to the
    implementation notwithstanding, convergence seems to be very slow (in terms
    of evaluations of the utility required). We recommend other Monte Carlo
    methods instead.

You can read more [in the documentation][data-valuation-intro].

!!! tip "New in version 0.4.0"

## References

[^1]: <a name="jia_efficient_2019"></a>Jia, R. et al., 2019.
    [Towards Efficient Data Valuation Based on the Shapley
    Value](https://proceedings.mlr.press/v89/jia19a.html).
    In: Proceedings of the 22nd International Conference on Artificial
    Intelligence and Statistics, pp. 1167–1176. PMLR.
"""

import logging
from collections import namedtuple
from typing import Iterable, Optional, Tuple, TypeVar, Union, cast

import cvxpy as cp
import numpy as np
from deprecate import deprecated
from numpy.random import SeedSequence
from numpy.typing import NDArray
from tqdm.auto import trange

from pydvl.parallel import (
    MapReduceJob,
    ParallelBackend,
    ParallelConfig,
    _maybe_init_parallel_backend,
)
from pydvl.utils import Utility
from pydvl.utils.numeric import random_subset_of_size
from pydvl.utils.status import Status
from pydvl.utils.types import Seed, ensure_seed_sequence
from pydvl.value import ValuationResult

__all__ = ["group_testing_shapley", "num_samples_eps_delta"]

log = logging.getLogger(__name__)

T = TypeVar("T", NDArray[np.float64], float)
GTConstants = namedtuple("GTConstants", ["kk", "Z", "q", "q_tot", "T"])


def _constants(
    n: int, epsilon: float, delta: float, utility_range: float
) -> GTConstants:
    """A helper function returning the constants for the algorithm. Pretty ugly,
    yes.

    Args:
        n: The number of data points.
        epsilon: The error tolerance.
        delta: The confidence level.
        utility_range: The range of the utility function.

    Returns:
        A namedtuple with the constants. The fields are the same as in the paper:
            - kk: the sample sizes (i.e. an array of 1, 2, ..., n - 1)
            - Z: the normalization constant
            - q: the probability of drawing a sample of size k
            - q_tot: another normalization constant
            - T: the number of iterations. This will be -1 if the utility_range is
                infinite. E.g. because the [Scorer][pydvl.utils.score.Scorer] does
                not define a range.
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
    r"""Implements the formula in Theorem 3 of (Jia, R. et al., 2019)<sup><a href="#jia_efficient_2019">1</a></sup>
    which gives a lower bound on the number of samples required to obtain an
    (ε/√n,δ/(N(N-1))-approximation to all pair-wise differences of Shapley
    values, wrt. $\ell_2$ norm.

    Args:
        eps: ε
        delta: δ
        n: Number of data points
        utility_range: Range of the [Utility][pydvl.utils.utility.Utility] function
    Returns:
        Number of samples from $2^{[n]}$ guaranteeing ε/√n-correct Shapley
            pair-wise differences of values with probability 1-δ/(N(N-1)).

    !!! tip "New in version 0.4.0"

    """
    constants = _constants(n=n, epsilon=eps, delta=delta, utility_range=utility_range)
    return int(constants.T)


def _group_testing_shapley(
    u: Utility,
    n_samples: int,
    progress: bool = False,
    job_id: int = 1,
    seed: Optional[Union[Seed, SeedSequence]] = None,
):
    """Helper function for
    [group_testing_shapley()][pydvl.value.shapley.gt.group_testing_shapley].

    Computes utilities of sets sampled using the strategy for estimating the
    differences in Shapley values.

    Args:
        u: Utility object with model, data, and scoring function.
        n_samples: total number of samples (subsets) to use.
        progress: Whether to display progress bars for each job.
        job_id: id to use for reporting progress (e.g. to place progres bars)
        seed: Either an instance of a numpy random number generator or a seed for it.
    Returns:

    """
    rng = np.random.default_rng(seed)
    n = len(u.data.indices)
    const = _constants(n, 1, 1, 1)  # don't care about eps,delta,range

    betas: NDArray[np.int_] = np.zeros(
        shape=(n_samples, n), dtype=np.int_
    )  # indicator vars
    uu = np.empty(n_samples)  # utilities

    for t in trange(n_samples, disable=not progress, position=job_id):
        k = rng.choice(const.kk, size=1, p=const.q).item()
        s = random_subset_of_size(u.data.indices, k, seed=rng)
        uu[t] = u(s)
        betas[t, s] = 1
    return uu, betas


@deprecated(
    target=True,
    args_mapping={"config": "config"},
    deprecated_in="0.9.0",
    remove_in="0.10.0",
)
def group_testing_shapley(
    u: Utility,
    n_samples: int,
    epsilon: float,
    delta: float,
    *,
    n_jobs: int = 1,
    parallel_backend: Optional[ParallelBackend] = None,
    config: Optional[ParallelConfig] = None,
    progress: bool = False,
    seed: Optional[Seed] = None,
    **options: dict,
) -> ValuationResult:
    """Implements group testing for approximation of Shapley values as described
    in (Jia, R. et al., 2019)<sup><a href="#jia_efficient_2019">1</a></sup>.

    !!! Warning
        This method is very inefficient. It requires several orders of magnitude
        more evaluations of the utility than others in
        [montecarlo][pydvl.value.shapley.montecarlo]. It also uses several intermediate
        objects like the results from the runners and the constraint matrices
        which can become rather large.

    By picking a specific distribution over subsets, the differences in Shapley
    values can be approximated with a Monte Carlo sum. These are then used to
    solve for the individual values in a feasibility problem.

    Args:
        u: Utility object with model, data, and scoring function
        n_samples: Number of tests to perform. Use
            [num_samples_eps_delta][pydvl.value.shapley.gt.num_samples_eps_delta]
            to estimate this.
        epsilon: From the (ε,δ) sample bound. Use the same as for the
            estimation of `n_iterations`.
        delta: From the (ε,δ) sample bound. Use the same as for the
            estimation of `n_iterations`.
        n_jobs: Number of parallel jobs to use. Each worker performs a chunk
            of all tests (i.e. utility evaluations).
        parallel_backend: Parallel backend instance to use
            for parallelizing computations. If `None`,
            use [JoblibParallelBackend][pydvl.parallel.backends.JoblibParallelBackend] backend.
            See the [Parallel Backends][pydvl.parallel.backends] package
            for available options.
        config: (**DEPRECATED**) Object configuring parallel computation,
            with cluster address, number of cpus, etc.
        progress: Whether to display progress bars for each job.
        seed: Either an instance of a numpy random number generator or a seed for it.
        options: Additional options to pass to
            [cvxpy.Problem.solve()](https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options).
            E.g. to change the solver (which defaults to `cvxpy.SCS`) pass
            `solver=cvxpy.CVXOPT`.

    Returns:
        Object with the data values.

    !!! tip "New in version 0.4.0"

    !!! tip "Changed in version 0.5.0"
        Changed the solver to cvxpy instead of scipy's linprog. Added the ability
        to pass arbitrary options to it.

    !!! tip "Changed in version 0.9.0"
        Deprecated `config` argument and added a `parallel_backend`
        argument to allow users to pass the Parallel Backend instance
        directly.
    """

    n = len(u.data.indices)

    const = _constants(
        n=n,
        epsilon=epsilon,
        delta=delta,
        utility_range=u.score_range.max() - u.score_range.min(),
    )
    T = n_samples
    if T < const.T:
        log.warning(
            f"n_samples of {T} are below the required {const.T} for the "
            f"ε={epsilon:.02f} guarantee at δ={1 - delta:.02f} probability"
        )

    parallel_backend = _maybe_init_parallel_backend(parallel_backend, config)

    samples_per_job = max(1, n_samples // parallel_backend.effective_n_jobs(n_jobs))

    def reducer(
        results_it: Iterable[Tuple[NDArray, NDArray]],
    ) -> Tuple[NDArray, NDArray]:
        return np.concatenate(list(x[0] for x in results_it)).astype(
            np.float64
        ), np.concatenate(list(x[1] for x in results_it)).astype(np.int_)

    seed_sequence = ensure_seed_sequence(seed)
    map_reduce_seed_sequence, cvxpy_seed = tuple(seed_sequence.spawn(2))

    map_reduce_job: MapReduceJob[Utility, Tuple[NDArray, NDArray]] = MapReduceJob(
        u,
        map_func=_group_testing_shapley,
        reduce_func=reducer,
        map_kwargs=dict(n_samples=samples_per_job, progress=progress),
        parallel_backend=parallel_backend,
        n_jobs=n_jobs,
    )
    uu, betas = map_reduce_job(seed=map_reduce_seed_sequence)

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
    solver = options.pop("solver", cp.SCS)
    problem.solve(solver=solver, **options)

    if problem.status != "optimal":
        log.warning(f"cvxpy returned status {problem.status}")
        values = (
            np.nan * np.ones_like(u.data.indices)
            if not hasattr(v.value, "__len__")
            else cast(NDArray[np.float64], v.value)
        )
        status = Status.Failed
    else:
        values = cast(NDArray[np.float64], v.value)
        status = Status.Converged

    return ValuationResult(
        algorithm="group_testing_shapley",
        status=status,
        values=values,
        data_names=u.data.data_names,
        solver_status=problem.status,
    )
