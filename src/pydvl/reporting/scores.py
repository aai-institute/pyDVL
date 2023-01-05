from collections import OrderedDict
from operator import itemgetter
from typing import Dict, Iterable, Mapping, Sequence, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

from pydvl.utils import Utility, maybe_progress
from pydvl.value.results import ValuationResult

__all__ = [
    "sort_values",
    "sort_values_array",
    "sort_values_history",
    "compute_removal_score",
]

KT = TypeVar("KT")
VT = TypeVar("VT")


def sort_values_array(values: np.ndarray) -> Dict[int, "NDArray"]:
    vals = np.mean(values, axis=1)
    return OrderedDict(sorted(enumerate(vals), key=itemgetter(1)))


def sort_values_history(values: Mapping[KT, Sequence[VT]]) -> Dict[KT, Sequence[VT]]:
    """Sorts a dict of sample_id: [values] by the last item in each list."""
    return OrderedDict(sorted(values.items(), key=itemgetter(1, -1)))


def sort_values(values: Mapping[KT, VT]) -> Dict[KT, VT]:
    """Sorts a dict of sample_id: value_float by value."""
    return OrderedDict(sorted(values.items(), key=itemgetter(1)))


def compute_removal_score(
    u: Utility,
    values: ValuationResult,
    percentages: Union["NDArray", Iterable[float]],
    *,
    remove_best: bool = False,
    progress: bool = False,
) -> Dict[float, float]:
    r"""Fits model and computes score on the test set after incrementally removing
    a percentage of data points from the training set, based on their values.

    :param u: Utility object with model, data, and scoring function.
    :param values: Data values of data instances in the training set.
    :param percentages: Sequence of removal percentages.
    :param remove_best: If True, removes data points in order of decreasing valuation.
    :param progress: If True, display a progress bar.
    :return: Dictionary that maps the percentages to their respective scores.
    """
    # Sanity checks
    if np.any([x >= 1.0 or x < 0.0 for x in percentages]):
        raise ValueError("All percentages should be in the range [0.0, 1.0)")

    if len(values) != len(u.data.indices):
        raise ValueError(
            f"The number of values, {len(values) }, should be equal to the number of data indices, {len(u.data.indices)}"
        )

    scores = {}

    # We sort in descending order if we want to remove the best values
    values.sort(reverse=remove_best)

    for pct in maybe_progress(
        percentages,
        display=progress,
        desc="Removal Scores",
    ):
        n_removal = int(pct * len(u.data))
        indices = values.indices[n_removal:]
        score = u(indices)
        scores[pct] = score
    return scores
