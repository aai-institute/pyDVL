"""
## References

[^1]: <a name="okhrati_multilinear_2021"></a>Okhrati, R., Lipani, A., 2021.
    [A Multilinear Sampling Algorithm to Estimate Shapley Values](https://ieeexplore.ieee.org/abstract/document/9412511).
    In: 2020 25th International Conference on Pattern Recognition (ICPR), pp. 7992–7999. IEEE.
"""

import operator
from enum import Enum
from functools import reduce
from itertools import cycle, takewhile
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from pydvl.utils import MapReduceJob, ParallelConfig, Utility, random_powerset
from pydvl.utils.types import Seed
from pydvl.value import ValuationResult
from pydvl.value.stopping import MinUpdates

__all__ = ["OwenAlgorithm", "owen_sampling_shapley"]


class OwenAlgorithm(Enum):
    Standard = "standard"
    Antithetic = "antithetic"


def _owen_sampling_shapley(
    indices: Sequence[int],
    u: Utility,
    method: OwenAlgorithm,
    n_samples: int,
    max_q: int,
    *,
    progress: bool = False,
    job_id: int = 1,
    seed: Optional[Seed] = None
) -> ValuationResult:
    r"""This is the algorithm as detailed in the paper: to compute the outer
    integral over q ∈ [0,1], use uniformly distributed points for evaluation
    of the integrand. For the integrand (the expected marginal utility over the
    power set), use Monte Carlo.

    !!! Todo
        We might want to try better quadrature rules like Gauss or Rombert or
        use Monte Carlo for the double integral.

    Args:
        indices: Indices to compute the value for
        u: Utility object with model, data, and scoring function
        method: Either [OwenAlgorithm.Full][pydvl.value.shapley.owen.OwenAlgorithm]
            for q ∈ [0, 1] or [OwenAlgorithm.Halved][pydvl.value.shapley.owen.OwenAlgorithm]
            for q ∈ [0, 0.5] and correlated samples
        n_samples: Number of subsets to sample to estimate the integrand
        max_q: number of subdivisions for the integration over $q$
        progress: Whether to display progress bars for each job
        job_id: For positioning of the progress bar
        seed: Either an instance of a numpy random number generator or a seed for it.

    Returns:
        Object with the data values, errors.
    """
    q_stop = {OwenAlgorithm.Standard: 1.0, OwenAlgorithm.Antithetic: 0.5}
    q_steps = np.linspace(start=0, stop=q_stop[method], num=max_q)

    result = ValuationResult.zeros(
        algorithm="owen_sampling_shapley_" + str(method),
        indices=np.array(indices, dtype=np.int_),
        data_names=[u.data.data_names[i] for i in indices],
    )

    rng = np.random.default_rng(seed)
    done = MinUpdates(1)
    repeat_indices = takewhile(lambda _: not done(result), cycle(indices))
    pbar = tqdm(disable=not progress, position=job_id, total=100, unit="%")
    for idx in repeat_indices:
        pbar.n = 100 * done.completion()
        pbar.refresh()
        e = np.zeros(max_q)
        subset = np.setxor1d(u.data.indices, [idx], assume_unique=True)
        for j, q in enumerate(q_steps):
            for s in random_powerset(subset, n_samples=n_samples, q=q, seed=rng):
                marginal = u({idx}.union(s)) - u(s)
                if method == OwenAlgorithm.Antithetic and q != 0.5:
                    s_complement = np.setxor1d(subset, s, assume_unique=True)
                    marginal += u({idx}.union(s_complement)) - u(s_complement)
                    marginal /= 2
                e[j] += marginal
        e /= n_samples
        result.update(idx, e.mean())
        # Trapezoidal rule
        # TODO: investigate whether this or other quadrature rules are better
        #  than a simple average
        # result.update(idx, (e[:-1] + e[1:]).sum() / (2 * max_q))

    return result


def owen_sampling_shapley(
    u: Utility,
    n_samples: int,
    max_q: int,
    *,
    method: OwenAlgorithm = OwenAlgorithm.Standard,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    progress: bool = False,
    seed: Optional[Seed] = None
) -> ValuationResult:
    r"""Owen sampling of Shapley values as described in
    (Okhrati and Lipani, 2021)<sup><a href="#okhrati_multilinear_2021">1</a></sup>.

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
    OwenAlgorithm.Antithetic``, uses correlated samples in the inner sum to
    reduce the variance:

    $$\hat{v}_u(i) = \frac{1}{2 Q M} \sum_{j=0}^Q \sum_{m=1}^M [u(S^{(q_j)}_m
    \cup \{i\}) - u(S^{(q_j)}_m) + u((S^{(q_j)}_m)^c \cup \{i\}) - u((S^{(
    q_j)}_m)^c)],$$

    where now $q_j = \frac{j}{2Q} \in [0,\frac{1}{2}]$, and $S^c$ is the
    complement of $S$.

    !!! Note
        The outer integration could be done instead with a quadrature rule.

    Args:
        u: [Utility][pydvl.utils.utility.Utility] object holding data, model
            and scoring function.
        n_samples: Numer of sets to sample for each value of q
        max_q: Number of subdivisions for q ∈ [0,1] (the element sampling
            probability) used to approximate the outer integral.
        method: Selects the algorithm to use, see the description. Either
            [OwenAlgorithm.Full][pydvl.value.shapley.owen.OwenAlgorithm] for
            $q \in [0,1]$ or
            [OwenAlgorithm.Halved][pydvl.value.shapley.owen.OwenAlgorithm] for
            $q \in [0,0.5]$ and correlated samples
        n_jobs: Number of parallel jobs to use. Each worker receives a chunk
            of the total of `max_q` values for q.
        config: Object configuring parallel computation, with cluster address,
            number of cpus, etc.
        progress: Whether to display progress bars for each job.
        seed: Either an instance of a numpy random number generator or a seed for it.

    Returns:
        Object with the data values.

    !!! tip "New in version 0.3.0"

    !!! tip "Changed in version 0.5.0"
        Support for parallel computation and enable antithetic sampling.

    """
    map_reduce_job: MapReduceJob[NDArray, ValuationResult] = MapReduceJob(
        u.data.indices,
        map_func=_owen_sampling_shapley,
        reduce_func=lambda results: reduce(operator.add, results),
        map_kwargs=dict(
            u=u,
            method=OwenAlgorithm(method),
            n_samples=n_samples,
            max_q=max_q,
            progress=progress,
        ),
        n_jobs=n_jobs,
        config=config,
    )

    return map_reduce_job(seed=seed)
