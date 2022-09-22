import math
import warnings
from collections import OrderedDict
from itertools import permutations

import numpy as np

from valuation.reporting.scores import sort_values
from valuation.utils import Utility, maybe_progress, powerset


def permutation_exact_shapley(u: Utility, progress: bool = True) -> OrderedDict:
    """Computes the exact Shapley value using permutations.
    When the length of the training set is > 10 it returns an error since the
    computation becomes too expensive.
    Used mostly for internal testing and simple use cases. Please refer to the
    montecarlo methods for all other cases.

    :param u: Utility object with model, data, and scoring function
    :param progress: set to True to use tqdm progress bars.
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


def combinatorial_exact_shapley(u: Utility, progress: bool = True) -> OrderedDict:
    """Computes the exact Shapley value using the combinatorial definition.
    When the length of the training set is > 20 it returns an error since the
    computation becomes too expensive.
    Used mostly for internal testing and simple use cases. Please refer to the
    montecarlo methods for all other cases.

    :param u: Utility object with model, data, and scoring function
    :param progress: set to True to use tqdm progress bars.
    :return: OrderedDict of exact Shapley values
    """

    n = len(u.data)

    # Arbitrary choice, will depend on time required, caching, etc.
    if n > 20:
        warnings.warn(f"Large dataset! Computation requires 2^{n} calls to model.fit()")

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
            values[i] += (u({i}.union(s)) - u(s)) / math.comb(n - 1, len(s))
    values /= n

    return sort_values({u.data.data_names[i]: v for i, v in enumerate(values)})
