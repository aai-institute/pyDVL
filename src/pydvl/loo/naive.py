from typing import OrderedDict

import numpy as np

from pydvl.reporting.scores import sort_values
from pydvl.utils import Utility, maybe_progress

__all__ = ["naive_loo"]


def naive_loo(
    u: Utility, *, progress: bool = True, **kwargs
) -> OrderedDict[int, float]:
    """Computes leave one out score.

    No caching nor parallelization is implemented.

    :param u: Utility object with model, data, and scoring function
    :param progress: If True, display a progress bar
    """

    values = {i: 0.0 for i in u.data.indices}
    for i in maybe_progress(data.indices, progress):  # type: ignore
        subset = np.setxor1d(u.data.indices, [i], assume_unique=True)
        values[i] = u(u.data.indices) - u(subset)

    return sort_values(values)
