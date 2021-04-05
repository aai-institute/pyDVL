import numpy as np

from itertools import permutations
from collections import OrderedDict
from tqdm.auto import tqdm
from valuation.reporting.scores import sort_values
from valuation.utils import Dataset, SupervisedModel, utility, powerset


def exact_permutation_shapley(model: SupervisedModel,
                              data: Dataset,
                              progress: bool = True) -> OrderedDict:
    """ Computes the exact Shapley value. """

    n = len(data)
    # Arbitrary choice: 8! = 11.2 hours if 1 sec per fit() + score()
    if n > 7:
        raise ValueError(
            f"Large dataset! Computation requires {n}! calls to model.fit()")

    values = np.zeros(n)
    if progress:
        wrap = lambda gen: tqdm(gen, desc="Permutation",
                                total=np.math.factorial(n))
    else:
        wrap = lambda val: val
    for p in wrap(permutations(data.index)):
        for i in range(len(p)):
            values[p[i]] += utility(model, data, p[:i+1]) \
                            - utility(model, data, p[:i])
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
    if progress:
        wrap = lambda gen: tqdm(gen, desc="Subset", total=2**(n-1))
    else:
        wrap = lambda val: val

    for i in data.index:
        for s in wrap(powerset(set(data.index) - {i})):
            values[i] += (utility(model, data, tuple({i}.union(s)))
                          - utility(model, data, tuple(s))) \
                         / np.math.comb(n-1, len(s))
    values /= n

    return sort_values({i: v for i, v in enumerate(values)})
