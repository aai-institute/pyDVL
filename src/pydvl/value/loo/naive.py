from typing import OrderedDict

import numpy as np

from pydvl.reporting.scores import sort_values
from pydvl.utils import Utility, maybe_progress

__all__ = ["naive_loo"]


def naive_loo(u: Utility, *, progress: bool = True) -> OrderedDict[int, float]:
    """Computes leave one out score.

    No caching nor parallelization is implemented.

    :param u: Utility object with model, data, and scoring function
    :param progress: If True, display a progress bar
    """

    values = {i: 0.0 for i in u.data.indices}
    all_indices = set(u.data.indices)
    for i in maybe_progress(data.indices, progress):  # type: ignore
        subset = all_indices.difference({i})
        values[i] = u(u.data.indices) - u(subset)

    return sort_values(values)
