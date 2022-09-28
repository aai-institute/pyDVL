from typing import OrderedDict

import numpy as np

from valuation.reporting.scores import sort_values
from valuation.utils import Dataset, SupervisedModel, maybe_progress
from valuation.utils.utility import Utility

__all__ = ["naive_loo"]


def naive_loo(
    model: SupervisedModel, data: Dataset, progress: bool = True, **kwargs
) -> OrderedDict[int, float]:
    """Computes leave one out score. No caching nor parallelization is implemented.

    :param model: Any supervised model.
    :param data: a split Dataset
    :param progress: whether to display a progress bar
    """
    u = Utility(model, data, **kwargs)

    def compute_utility(x: np.ndarray) -> float:
        return u(frozenset(x))

    values = {i: 0.0 for i in data.indices}
    for i in maybe_progress(data.indices, progress):  # type: ignore
        subset = np.setxor1d(data.indices, [i], assume_unique=True)
        values[i] = compute_utility(data.indices) - compute_utility(subset)

    return sort_values(values)
