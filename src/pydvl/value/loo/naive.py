import numpy as np

from pydvl.utils import Utility, maybe_progress
from pydvl.value.results import ValuationResult, ValuationStatus

__all__ = ["naive_loo"]


def naive_loo(u: Utility, *, progress: bool = True) -> ValuationResult:
    r"""Computes leave one out value:

    $$v(i) = u(D) - u(D \setminus \{i\}) $$

    :param u: Utility object with model, data, and scoring function
    :param progress: If True, display a progress bar
    :return: Object with the data values.
    """

    if len(u.data) < 3:
        raise ValueError("Dataset must have at least 2 elements")

    values = np.zeros_like(u.data.indices, dtype=np.float_)
    all_indices = set(u.data.indices)
    total_utility = u(u.data.indices)
    for i in maybe_progress(u.data.indices, progress):  # type: ignore
        subset = all_indices.difference({i})
        values[i] = total_utility - u(subset)

    return ValuationResult(
        algorithm="naive_loo",
        status=ValuationStatus.Converged,
        values=values,
        stderr=None,
        data_names=u.data.data_names,
    )
