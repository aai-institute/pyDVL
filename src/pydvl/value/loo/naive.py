from typing import OrderedDict

import numpy as np

from pydvl.utils import Utility, maybe_progress

__all__ = ["naive_loo"]

from pydvl.value import ValuationResult
from pydvl.value.valuationresult import SortOrder, ValuationStatus


def naive_loo(u: Utility, *, progress: bool = True) -> ValuationResult:
    """Computes leave one out score.

    No caching nor parallelization is implemented.

    :param u: Utility object with model, data, and scoring function
    :param progress: If True, display a progress bar
    :return: Object with the data values.
    """

    values = np.zeros_like(u.data.indices)
    all_indices = set(u.data.indices)
    for i in maybe_progress(data.indices, progress):  # type: ignore
        subset = all_indices.difference({i})
        values[i] = u(u.data.indices) - u(subset)

    return ValuationResult(
        algorithm=naive_loo,
        status=ValuationStatus.Converged,
        values=values,
        stderr=None,
        data_names=u.data.data_names,
        sort=SortOrder.Descending,
    )
