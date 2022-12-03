import numpy as np

from pydvl.utils import SortOrder, Utility, maybe_progress
from pydvl.value import ValuationResult, ValuationStatus

__all__ = ["naive_loo"]


def naive_loo(u: Utility, *, progress: bool = True) -> ValuationResult:
    """Computes leave one out score.

    No caching nor parallelization is implemented.

    :param u: Utility object with model, data, and scoring function
    :param progress: If True, display a progress bar
    :return: Object with the data values.
    """

    values = np.zeros_like(u.data.indices, dtype=np.float_)
    all_indices = set(u.data.indices)
    for i in maybe_progress(u.data.indices, progress):  # type: ignore
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
