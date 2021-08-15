import numpy as np

from typing import OrderedDict
from valuation.reporting.scores import sort_values
from valuation.utils import maybe_progress, Utility


def loo(u: Utility, progress: bool = True) -> OrderedDict[int, float]:
    """ Computes the LOO score for each training point in the dataset."""
    values = {i: 0.0 for i in u.data.indices}
    for i in maybe_progress(u.data.indices, progress):
        subset = np.setxor1d(u.data.indices, [i], assume_unique=True)
        values[i] = u(u.data.indices) - u(subset)

    return sort_values(values)
