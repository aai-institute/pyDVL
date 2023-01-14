import math
import warnings
from itertools import permutations
from typing import List, Sequence

import numpy as np
from numpy.typing import NDArray

from pydvl.utils import MapReduceJob, ParallelConfig, Utility, maybe_progress, powerset
from pydvl.value.results import ValuationResult, ValuationStatus

__all__ = ["permutation_exact_shapley", "combinatorial_exact_shapley"]


def permutation_exact_shapley(u: Utility, *, progress: bool = True) -> ValuationResult:
    r"""Computes the exact Shapley value using the formulation with permutations:

    $$v_u(x_i) = \frac{1}{n!} \sum_{\sigma \in \Pi(n)} [u(\sigma_{i-1} \cup {i}) − u(\sigma_{i})].$$

    See :ref:`data valuation` for details.

    When the length of the training set is > 10 this prints a warning since the
    computation becomes too expensive. Used mostly for internal testing and
    simple use cases. Please refer to the :mod:`Monte Carlo
    <pydvl.value.shapley.montecarlo>` approximations for practical applications.

    :param u: Utility object with model, data, and scoring function
    :param progress: Whether to display progress bars for each job.
    :return: Object with the data values.
    """

    n = len(u.data)
    # Note that the cache in utility saves most of the refitting because we
    # use frozenset for the input.
    if n > 10:
        warnings.warn(
            f"Large dataset! Computation requires {n}! calls to utility()",
            RuntimeWarning,
        )

    values = np.zeros(n)
    for p in maybe_progress(
        permutations(u.data.indices),
        progress,
        desc="Permutation",
        total=math.factorial(n),
    ):
        for i, idx in enumerate(p):
            values[idx] += u(p[: i + 1]) - u(p[:i])
    values /= math.factorial(n)

    return ValuationResult(
        algorithm="permutation_exact_shapley",
        status=ValuationStatus.Converged,
        values=values,
        stderr=None,
        data_names=u.data.data_names,
    )


def _combinatorial_exact_shapley(
    indices: Sequence[int], u: Utility, progress: bool
) -> NDArray:
    """Helper function for :func:`combinatorial_exact_shapley`.

    Computes the marginal utilities for the set of indices passed and returns
    the value of the samples according to the exact combinatorial definition.
    """
    n = len(u.data)
    local_values = np.zeros(n)
    for i in indices:
        subset = np.setxor1d(u.data.indices, [i], assume_unique=True).astype(np.int_)
        for s in maybe_progress(
            powerset(subset),
            progress,
            desc=f"Index {i}",
            total=2 ** (n - 1),
            position=0,
        ):
            local_values[i] += (u({i}.union(s)) - u(s)) / math.comb(n - 1, len(s))
    return local_values / n


def combinatorial_exact_shapley(
    u: Utility,
    *,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    progress: bool = False,
) -> ValuationResult:
    r"""Computes the exact Shapley value using the combinatorial definition.

    $$v_u(i) = \frac{1}{n} \sum_{S \subseteq N \setminus \{i\}} \binom{n-1}{ | S | }^{-1} [u(S \cup \{i\}) − u(S)].$$

    See :ref:`data valuation` for details.

    .. note::
       If the length of the training set is > n_jobs*20 this prints a warning
       because the computation is very expensive. Used mostly for internal testing
       and simple use cases. Please refer to the
       :mod:`Monte Carlo <pydvl.shapley.montecarlo>` approximations for practical
       applications.

    :param u: Utility object with model, data, and scoring function
    :param n_jobs: Number of parallel jobs to use
    :param config: Object configuring parallel computation, with cluster address,
        number of cpus, etc.
    :param progress: Whether to display progress bars for each job.
    :return: Object with the data values.
    """
    # Arbitrary choice, will depend on time required, caching, etc.
    if len(u.data) // n_jobs > 20:
        warnings.warn(
            f"Large dataset! Computation requires 2^{len(u.data)} calls to model.fit()"
        )

    def reduce_fun(results: List[NDArray]) -> NDArray:
        return np.array(results).sum(axis=0)  # type: ignore

    map_reduce_job: MapReduceJob[NDArray, NDArray] = MapReduceJob(
        u.data.indices,
        map_func=_combinatorial_exact_shapley,
        map_kwargs=dict(u=u, progress=progress),
        reduce_func=reduce_fun,
        n_jobs=n_jobs,
        config=config,
    )
    values = map_reduce_job()
    return ValuationResult(
        algorithm="combinatorial_exact_shapley",
        status=ValuationStatus.Converged,
        values=values,
        stderr=None,
        data_names=u.data.data_names,
    )
