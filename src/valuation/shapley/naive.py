from collections import OrderedDict
from itertools import permutations

import numpy as np

from valuation.reporting.scores import sort_values
from valuation.utils import Utility, maybe_progress, powerset


def permutation_exact_shapley(u: Utility, progress: bool = True) -> OrderedDict:
    """Computes the exact Shapley value using permutations."""

    n = len(u.data)
    # Note that the cache in utility saves most of the refitting because we
    # use frozenset for the input.
    if n > 10:
        raise ValueError(f"Large dataset! Computation requires {n}! calls to utility()")

    values = np.zeros(n)
    for p in maybe_progress(
        permutations(u.data.indices),
        progress,
        desc="Permutation",
        total=np.math.factorial(n),
    ):
        for i, idx in enumerate(p):
            values[idx] += u(p[: i + 1]) - u(p[:i])
    values /= np.math.factorial(n)

    return sort_values({u.data.data_names[i]: v for i, v in enumerate(values)})


def combinatorial_exact_shapley(u: Utility, progress: bool = True) -> OrderedDict:
    """Computes the exact Shapley value using the combinatorial definition."""

    n = len(u.data)
    from valuation.utils.logging import logger

    if n > 20:  # Arbitrary choice, will depend on time required, caching, etc.
        logger.warning(
            f"Large dataset! Computation requires 2^{n} calls to model.fit()"
        )

    values = np.zeros(n)
    for i in u.data.indices:
        subset = np.setxor1d(u.data.indices, [i], assume_unique=True)
        for s in maybe_progress(
            powerset(subset),
            progress,
            desc=f"Index {i}",
            total=2 ** (n - 1),
            position=0,
        ):
            values[i] += (u({i}.union(s)) - u(s)) / np.math.comb(n - 1, len(s))
    values /= n

    return sort_values({u.data.data_names[i]: v for i, v in enumerate(values)})


def exact_shapley(u: Utility, progress: bool = True, use_combinatorial=False):
    """Facade for exact shapley methods. By default, it uses permutation_exact_shapley"""
    if use_combinatorial:
        return combinatorial_exact_shapley(u, progress)
    else:
        return permutation_exact_shapley(u, progress)
