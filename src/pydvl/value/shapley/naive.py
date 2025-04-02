r"""
This module implements exact Shapley values using either the combinatorial or
permutation definition.

The exact computation of $n$ values takes $\mathcal{O}(2^n)$ evaluations of the
utility and is therefore only possible for small datasets. For larger datasets,
consider using any of the approximations, such as [Monte
Carlo][pydvl.value.shapley.montecarlo], or proxy models like
[kNN][pydvl.value.shapley.knn].

See [Data valuation][data-valuation-intro] for details.
"""

import math
import warnings
from itertools import permutations
from typing import List, Optional

import numpy as np
from deprecate import deprecated
from numpy.typing import NDArray
from tqdm.auto import tqdm

from pydvl.parallel import (
    MapReduceJob,
    ParallelBackend,
    ParallelConfig,
    _maybe_init_parallel_backend,
)
from pydvl.utils import Utility, powerset
from pydvl.utils.status import Status
from pydvl.value.result import ValuationResult

__all__ = ["permutation_exact_shapley", "combinatorial_exact_shapley"]


def permutation_exact_shapley(u: Utility, *, progress: bool = True) -> ValuationResult:
    r"""Computes the exact Shapley value using the formulation with permutations:

    $$v_u(x_i) = \frac{1}{n!} \sum_{\sigma \in \Pi(n)} [u(\sigma_{i-1}
    \cup {i}) − u(\sigma_{i})].$$

    See [Data valuation][data-valuation-intro] for details.

    When the length of the training set is > 10 this prints a warning since the
    computation becomes too expensive. Used mostly for internal testing and
    simple use cases. Please refer to the [Monte Carlo
    approximations][pydvl.value.shapley.montecarlo] for practical applications.

    Args:
        u: Utility object with model, data, and scoring function
        progress: Whether to display progress bars for each job.

    Returns:
        Object with the data values.
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
    for p in tqdm(
        permutations(u.data.indices),
        disable=not progress,
        desc="Permutation",
        total=math.factorial(n),
    ):
        for i, idx in enumerate(p):
            values[idx] += u(p[: i + 1]) - u(p[:i])
    values /= math.factorial(n)

    return ValuationResult(
        algorithm="permutation_exact_shapley",
        status=Status.Converged,
        values=values,
        data_names=u.data.data_names,
    )


def _combinatorial_exact_shapley(
    indices: NDArray[np.int_], u: Utility, progress: bool
) -> NDArray:
    """Helper function for
    [combinatorial_exact_shapley()][pydvl.value.shapley.naive.combinatorial_exact_shapley].

    Computes the marginal utilities for the set of indices passed and returns
    the value of the samples according to the exact combinatorial definition.
    """
    n = len(u.data)
    local_values = np.zeros(n)
    for i in indices:
        subset: NDArray[np.int_] = np.setxor1d(
            u.data.indices, [i], assume_unique=True
        ).astype(np.int_)
        for s in tqdm(  # type: ignore
            powerset(subset),
            disable=not progress,
            desc=f"Index {i}",
            total=2 ** (n - 1),
            position=0,
        ):
            local_values[i] += (u({i}.union(s)) - u(s)) / math.comb(n - 1, len(s))  # type: ignore
    return local_values / n


@deprecated(
    target=True,
    args_mapping={"config": "config"},
    deprecated_in="0.9.0",
    remove_in="0.10.0",
)
def combinatorial_exact_shapley(
    u: Utility,
    *,
    n_jobs: int = 1,
    parallel_backend: Optional[ParallelBackend] = None,
    config: Optional[ParallelConfig] = None,
    progress: bool = False,
) -> ValuationResult:
    r"""Computes the exact Shapley value using the combinatorial definition.

    $$v_u(i) = \frac{1}{n} \sum_{S \subseteq N \setminus \{i\}}
    \binom{n-1}{ | S | }^{-1} [u(S \cup \{i\}) − u(S)].$$

    See [Data valuation][data-valuation-intro] for details.

    !!! Note
        If the length of the training set is > n_jobs*20 this prints a warning
        because the computation is very expensive. Used mostly for internal
        testing and simple use cases. Please refer to the
        [Monte Carlo][pydvl.value.shapley.montecarlo] approximations for
        practical applications.

    Args:
        u: Utility object with model, data, and scoring function
        n_jobs: Number of parallel jobs to use
        parallel_backend: Parallel backend instance to use
            for parallelizing computations. If `None`,
            use [JoblibParallelBackend][pydvl.parallel.backends.JoblibParallelBackend] backend.
            See the [Parallel Backends][pydvl.parallel.backends] package
            for available options.
        config: (**DEPRECATED**) Object configuring parallel computation,
            with cluster address, number of cpus, etc.
        progress: Whether to display progress bars for each job.

    Returns:
        Object with the data values.

    !!! tip "Changed in version 0.9.0"
        Deprecated `config` argument and added a `parallel_backend`
        argument to allow users to pass the Parallel Backend instance
        directly.
    """
    # Arbitrary choice, will depend on time required, caching, etc.
    if len(u.data) // n_jobs > 20:
        warnings.warn(
            f"Large dataset! Computation requires 2^{len(u.data)} calls to model.fit()"
        )

    def reduce_fun(results: List[NDArray]) -> NDArray:
        return np.array(results).sum(axis=0)  # type: ignore

    parallel_backend = _maybe_init_parallel_backend(parallel_backend, config)

    map_reduce_job: MapReduceJob[NDArray, NDArray] = MapReduceJob(
        u.data.indices,
        map_func=_combinatorial_exact_shapley,
        map_kwargs=dict(u=u, progress=progress),
        reduce_func=reduce_fun,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
    )
    values = map_reduce_job()
    return ValuationResult(
        algorithm="combinatorial_exact_shapley",
        status=Status.Converged,
        values=values,
        data_names=u.data.data_names,
    )
