from typing import OrderedDict

import numpy as np

from valuation.reporting.scores import sort_values
from valuation.utils import Dataset, SupervisedModel, maybe_progress, utility


def naive_loo(
    model: SupervisedModel, data: Dataset, progress: bool = True, **kwargs
) -> OrderedDict[int, float]:
    """Computes the LOO score for each training point in the dataset."""
    u = utility.Utility(model, data, **kwargs)
    compute_utility = lambda x: u(frozenset(x))
    values = {i: 0.0 for i in data.indices}
    for i in maybe_progress(data.indices, progress):
        subset = np.setxor1d(data.indices, [i], assume_unique=True)
        values[i] = compute_utility(data.indices) - compute_utility(subset)

    return sort_values(values)
