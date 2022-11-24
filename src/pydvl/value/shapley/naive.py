import math
import warnings
from collections import OrderedDict
from itertools import permutations

import numpy as np

from pydvl.reporting.scores import sort_values
from pydvl.utils import (
    MapReduceJob,
    ParallelConfig,
    Utility,
    init_parallel_backend,
    maybe_progress,
    powerset,
)

__all__ = ["permutation_exact_shapley", "combinatorial_exact_shapley"]


def permutation_exact_shapley(
    u: Utility, *, progress: bool = True
) -> "OrderedDict[str, float]":
    """Computes the exact Shapley value using permutations.

    When the length of the training set is > 10 this prints a warning since the
    computation becomes too expensive. Used mostly for internal testing and
    simple use cases. Please refer to the Monte Carlo methods for all other
    cases.

    :param u: Utility object with model, data, and scoring function
    :param progress: Whether to display progress bars for each job.
    :return: OrderedDict of exact Shapley values
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

    return sort_values({u.data.data_names[i]: v for i, v in enumerate(values)})


def _combinatorial_exact_shapley(
    indices: np.ndarray, u: Utility, progress: bool
) -> np.ndarray:
    """Helper function for :func:`combinatorial_exact_shapley`.

    Computes the marginal utilities for the set of indices passed and returns
    the value of the samples according to the exact combinatorial definition.
    """
    n = len(u.data)
    local_values = np.zeros(n)
    for i in indices:
        subset = np.setxor1d(u.data.indices, [i], assume_unique=True)
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
) -> "OrderedDict[str, float]":
    r"""Computes the exact Shapley value using the combinatorial definition.

    $$v_u(i) = \frac{1}{n} \sum_{S \subseteq N \setminus \{i\}} \binom{n-1}{ | S | }^{-1} [u(S \cup \{i\}) − u(S)]$$

    If the length of the training set is > n_jobs*20 this prints a warning
    because the computation is very expensive. Used mostly for internal testing
    and simple use cases. Please refer to the Monte Carlo methods for all other
    cases.

    :param u: Utility object with model, data, and scoring function
    :param n_jobs: Number of parallel jobs to use
    :param config: Object configuring parallel computation, with cluster address,
        number of cpus, etc.
    :param progress: Whether to display progress bars for each job.

    :return: Dictionary of {"index or label": exact_value}, sorted by decreasing
        value.
    """
    # Arbitrary choice, will depend on time required, caching, etc.
    if len(u.data) // n_jobs > 20:
        warnings.warn(
            f"Large dataset! Computation requires 2^{len(u.data)} calls to model.fit()"
        )

    parallel_backend = init_parallel_backend(config)
    u_id = parallel_backend.put(u)

    def reduce_fun(results):
        return np.array(results).sum(axis=0)

    map_reduce_job: MapReduceJob[np.ndarray, np.ndarray] = MapReduceJob(
        map_func=_combinatorial_exact_shapley,
        map_kwargs=dict(u=u_id, progress=progress),
        reduce_func=reduce_fun,
        chunkify_inputs=True,
        n_jobs=n_jobs,
    )
    values = map_reduce_job(u.data.indices)[0]
    return sort_values({u.data.data_names[i]: v for i, v in enumerate(values)})
