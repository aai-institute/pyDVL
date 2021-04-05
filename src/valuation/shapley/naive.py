import numpy as np

from itertools import permutations
from collections import OrderedDict
from tqdm.auto import tqdm
from valuation.reporting.scores import sort_values
from valuation.utils import Dataset, SupervisedModel, utility


def exact_permutation_shapley(model: SupervisedModel,
                              data: Dataset,
                              progress: bool = True) -> OrderedDict:
    """ Computes the exact Shapley value. """

    # Arbitrary choice ~= 1.14 hours if 1 sec per fit() + score()
    if len(data) > 12:
        raise ValueError(
            f"Large dataset! Computation requires {2 ** len(data)} "
            f"calls to model.fit()")

    values = np.zeros(len(data))
    if progress:
        wrap = lambda gen: tqdm(gen, total=np.math.factorial(len(data)))
    else:
        wrap = lambda val: val
    for p in wrap(permutations(data.index)):
        for i in range(len(p)):
            values[p[i]] += utility(p[:i + 1]) - utility(p[:i])
    values /= np.math.factorial(len(data))

    return sort_values({i: v for i, v in enumerate(values)})

