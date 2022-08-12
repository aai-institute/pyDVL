from typing import OrderedDict

import numpy as np

from valuation.reporting.scores import sort_values
from valuation.utils import Dataset, SupervisedModel, maybe_progress
from valuation.utils.utility import Utility


def naive_loo(
    model: SupervisedModel, data: Dataset, progress: bool = True, **kwargs
) -> OrderedDict[int, float]:
    """Computes the LOO score for each training point in the dataset."""
    u = Utility(model, data, **kwargs)

    def compute_utility(x: np.ndarray) -> float:
        return u(frozenset(x))

    values = {i: 0.0 for i in data.indices}
    for i in maybe_progress(data.indices, progress):  # type: ignore
        subset = np.setxor1d(data.indices, [i], assume_unique=True)
        values[i] = compute_utility(data.indices) - compute_utility(subset)

    return sort_values(values)
