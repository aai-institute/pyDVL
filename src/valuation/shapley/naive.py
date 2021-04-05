import numpy as np

from functools import lru_cache
from itertools import permutations
from collections import OrderedDict
from valuation.reporting.scores import sort_values
from valuation.utils import Dataset, SupervisedModel


def exact_combinatorial_shapley(model: SupervisedModel,
                                data: Dataset) -> OrderedDict:
    """ Computes the exact Shapley value. """

    # Arbitrary choice ~= 1.14 hours if 1 sec per fit() + score()
    if len(data) > 12:
        raise ValueError(
            f"Large dataset! Computation requires {2 ** len(data)} "
            f"calls to model.fit()")

    # There is a clever way of rearranging the loops to fit just once per
    # list of indices. Or... we can just cache and be done with it.
    @lru_cache
    def utility(indices):
        x = data.x_train.iloc[indices]
        y = data.y_train.iloc[indices]
        model.fit(x, y)
        return model.score(data.x_test, data.y_test)

    values = np.zeros(len(data))
    for p in permutations(data.index):
        for i in range(len(p)):
            values[p[i]] += utility(p[:i + 1]) - utility(p[:i])
    values /= np.math.factorial(len(data))

    return sort_values({i: v for i, v in enumerate(values)})

