import numpy as np

from collections import OrderedDict
from functools import partial
from itertools import permutations
from typing import Iterable, Tuple
from valuation.reporting.scores import sort_values
from valuation.utils import Dataset, SupervisedModel, maybe_progress, utility,\
    powerset
from valuation.utils.numeric import random_powerset


def exact_permutation_shapley(model: SupervisedModel,
                              data: Dataset,
                              progress: bool = True) -> OrderedDict:
    """ Computes the exact Shapley value using permutations.

     FIXME: it is stated in multiple places that this is equivalent to the
      combinatorial approach, but this is O(n!) while the other is O(2^n). Why
      do montecarlo using permutations instead of the powerset? Ghorbani and
      Zou 2019 sequentially add points from the permutation in order to do
      early stopping when the marginal contributions are below a threshold.
      Also, Maleki 2014 do their analysis for the permutation formulation. Do
      the bounds hold for both?
     """

    n = len(data)
    # Arbitrary choice: 8! = 11.2 hours if 1 sec per fit() + score()
    if n > 7:
        raise ValueError(
            f"Large dataset! Computation requires {n}! calls to model.fit()")

    values = np.zeros(n)
    u = partial(utility, model, data)
    for p in maybe_progress(permutations(data.ilocs), progress,
                            desc="Permutation", total=np.math.factorial(n)):
        for i in range(len(p)):
            values[p[i]] += u(p[:i+1]) - u(p[:i])
    values /= np.math.factorial(n)

    return sort_values({i: v for i, v in enumerate(values)})


def exact_combinatorial_shapley(model: SupervisedModel,
                                data: Dataset,
                                progress: bool = True) -> OrderedDict:
    """ Computes the exact Shapley value using the combinatorial definition. """

    n = len(data)
    # Arbitrary choice ~= 1.14 hours if 1 sec per fit() + score()
    if n > 12:
        raise ValueError(
            f"Large dataset! Computation requires 2^{n} calls to model.fit()")

    values = np.zeros(n)
    u = partial(utility, model, data)
    for i in data.ilocs:
        subset = np.setxor1d(data.ilocs, [i], assume_unique=True)
        for s in maybe_progress(powerset(subset), progress,
                                desc=f"Index {i}", total=2 ** (n - 1),
                                position=0):
            values[i] += (u(tuple({i}.union(s))) - u(tuple(s))) \
                         / np.math.comb(n-1, len(s))
    values /= n

    return sort_values({i: v for i, v in enumerate(values)})


def montecarlo_combinatorial_shapley(model: SupervisedModel,
                                     data: Dataset,
                                     indices: Iterable[int],
                                     progress: bool = True) \
        -> Tuple[OrderedDict, None]:
    """ Computes an approximate Shapley value using the combinatorial
    definition and MonteCarlo samples
     """

    n = len(data)
    max_subsets = 2**n  # temporary, FIXME
    values = np.zeros(n)
    u = partial(utility, model, data)
    for i in indices:
        # Randomly sample subsets of index - {j}
        subset = np.setxor1d(indices, [i], assume_unique=True)
        power_set = enumerate(random_powerset(subset, max_subsets=max_subsets),
                              start=1)
        for j, s in maybe_progress(power_set, progress, desc=f"Index {i}",
                                   total=max_subsets, position=0):
            values[i] += ((u(tuple({i}.union(s))) - u(tuple(s))) - values[i]) / j

    return sort_values({i: v for i, v in enumerate(values)}), None
