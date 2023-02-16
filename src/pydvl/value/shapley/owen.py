import operator
from enum import Enum
from functools import reduce
from typing import Sequence

import numpy as np
from _warnings import warn
from numpy._typing import NDArray

from pydvl.utils import (
    MapReduceJob,
    ParallelConfig,
    Utility,
    maybe_progress,
    random_powerset,
)
from pydvl.value import ValuationResult


class OwenAlgorithm(Enum):
    Standard = "standard"
    Antithetic = "antithetic"


def _owen_sampling_shapley(
    indices: Sequence[int],
    u: Utility,
    method: OwenAlgorithm,
    n_iterations: int,
    max_q: int,
    *,
    progress: bool = False,
    job_id: int = 1,
) -> ValuationResult:
    r"""This is the algorithm as detailed in the paper: to compute the outer
    integral over q ∈ [0,1], use uniformly distributed points for evaluation
    of the integrand. For the integrand (the expected marginal utility over the
    power set), use Monte Carlo.

    .. todo::
        We might want to try better quadrature rules like Gauss or Rombert or
        use Monte Carlo for the double integral.

    :param indices: Indices to compute the value for
    :param u: Utility object with model, data, and scoring function
    :param method: Either :attr:`~OwenAlgorithm.Full` for $q \in [0,1]$ or
        :attr:`~OwenAlgorithm.Halved` for $q \in [0,0.5]$ and correlated samples
    :param n_iterations: Number of subsets to sample to estimate the integrand
    :param max_q: number of subdivisions for the integration over $q$
    :param progress: Whether to display progress bars for each job
    :param job_id: For positioning of the progress bar
    :return: Object with the data values, errors.
    """
    values = np.zeros(len(u.data))

    q_stop = {OwenAlgorithm.Standard: 1.0, OwenAlgorithm.Antithetic: 0.5}
    q_steps = np.linspace(start=0, stop=q_stop[method], num=max_q)

    index_set = set(indices)
    for i in maybe_progress(indices, progress, position=job_id):
        e = np.zeros(max_q)
        subset = np.array(list(index_set.difference({i})))
        for j, q in enumerate(q_steps):
            for s in random_powerset(subset, n_samples=n_iterations, q=q):
                marginal = u({i}.union(s)) - u(s)
                if method == OwenAlgorithm.Antithetic and q != 0.5:
                    s_complement = index_set.difference(s)
                    marginal += u({i}.union(s_complement)) - u(s_complement)
                e[j] += marginal
        e /= n_iterations
        # values[i] = e.mean()
        # Trapezoidal rule
        values[i] = (e[:-1] + e[1:]).sum() / (2 * max_q)

    return ValuationResult(
        algorithm="owen_sampling_shapley_" + str(method), values=values
    )


def owen_sampling_shapley(
    u: Utility,
    n_iterations: int,
    max_q: int,
    *,
    method: OwenAlgorithm = OwenAlgorithm.Standard,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    progress: bool = False,
) -> ValuationResult:
    r"""Owen sampling of Shapley values as described in
    :footcite:t:`okhrati_multilinear_2021`.

    .. warning::
       Antithetic sampling is unstable and not properly tested

    This function computes a Monte Carlo approximation to

    $$v_u(i) = \int_0^1 \mathbb{E}_{S \sim P_q(D_{\backslash \{i\}})}
    [u(S \cup \{i\}) - u(S)]$$

    using one of two methods. The first one, selected with the argument ``mode =
    OwenAlgorithm.Standard``, approximates the integral with:

    $$\hat{v}_u(i) = \frac{1}{Q M} \sum_{j=0}^Q \sum_{m=1}^M [u(S^{(q_j)}_m
    \cup \{i\}) - u(S^{(q_j)}_m)],$$

    where $q_j = \frac{j}{Q} \in [0,1]$ and the sets $S^{(q_j)}$ are such that a
    sample $x \in S^{(q_j)}$ if a draw from a $Ber(q_j)$ distribution is 1.

    The second method, selected with the argument ``mode =
    OwenAlgorithm.Anthithetic``,
    uses correlated samples in the inner sum to reduce the variance:

    $$\hat{v}_u(i) = \frac{1}{Q M} \sum_{j=0}^Q \sum_{m=1}^M [u(S^{(q_j)}_m
    \cup \{i\}) - u(S^{(q_j)}_m) + u((S^{(q_j)}_m)^c \cup \{i\}) - u((S^{(
    q_j)}_m)^c)],$$

    where now $q_j = \frac{j}{2Q} \in [0,\frac{1}{2}]$, and $S^c$ is the
    complement of $S$.

    :param u: :class:`~pydvl.utils.utility.Utility` object holding data, model
        and scoring function.
    :param n_iterations: Numer of sets to sample for each value of q
    :param max_q: Number of subdivisions for q ∈ [0,1] (the element sampling
        probability) used to approximate the outer integral.
    :param method: Selects the algorithm to use, see the description. Either
        :attr:`~OwenAlgorithm.Full` for $q \in [0,1]$ or
        :attr:`~OwenAlgorithm.Halved` for $q \in [0,0.5]$ and correlated samples
    :param n_jobs: Number of parallel jobs to use. Each worker receives a chunk
        of the total of `max_q` values for q.
    :param config: Object configuring parallel computation, with cluster
        address, number of cpus, etc.
    :param progress: Whether to display progress bars for each job.
    :return: Object with the data values.

    .. versionadded:: 0.3.0

    """
    if n_jobs > 1:
        raise NotImplementedError("Parallel Owen sampling not implemented yet")

    if OwenAlgorithm(method) == OwenAlgorithm.Antithetic:
        warn("Owen antithetic sampling not tested and probably bogus")

    map_reduce_job: MapReduceJob[NDArray, ValuationResult] = MapReduceJob(
        u.data.indices,
        map_func=_owen_sampling_shapley,
        reduce_func=lambda results: reduce(operator.add, results),
        map_kwargs=dict(
            u=u,
            method=OwenAlgorithm(method),
            n_iterations=n_iterations,
            max_q=max_q,
            progress=progress,
        ),
        n_jobs=n_jobs,
        config=config,
    )

    return map_reduce_job()
