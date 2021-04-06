import numpy as np
from functools import partial
from typing import OrderedDict
from valuation.reporting.scores import sort_values
from valuation.utils import Dataset, SupervisedModel, maybe_progress, utility


def naive_loo(model: SupervisedModel,
              data: Dataset,
              progress: bool = True) -> OrderedDict[int, float]:
    """ Computes the LOO score for each training point in the dataset."""
    u = partial(utility, model, data)
    u_total = u(tuple(data.ilocs))
    values = {i: 0.0 for i in data.ilocs}
    for i in maybe_progress(data.ilocs, progress):
        subset = np.setxor1d(data.ilocs, [i], assume_unique=True)
        values[i] = u_total - u(tuple(subset))

    return sort_values(values)
