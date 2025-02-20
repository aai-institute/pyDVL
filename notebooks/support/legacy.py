"""
DELETE THIS FILE:
Temporary support for least_core_basic.ipynb until it no longer requires
compute_removal_score_legacy
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from pydvl.utils import Utility
from pydvl.value.result import ValuationResult as LegacyValuationResult

__all__ = ["compute_removal_score_legacy"]


def compute_removal_score_legacy(
    u: Utility,
    values: LegacyValuationResult,
    percentages: NDArray[np.float64] | Iterable[float],
    *,
    remove_best: bool = False,
    progress: bool = False,
) -> dict[float, float]:
    r"""Fits model and computes score on the test set after incrementally removing
    a percentage of data points from the training set, based on their values.

    Args:
        u: Utility object with model, data, and scoring function.
        values: Data values of data instances in the training set.
        percentages: Sequence of removal percentages.
        remove_best: If True, removes data points in order of decreasing valuation.
        progress: If True, display a progress bar.

    Returns:
        Dictionary that maps the percentages to their respective scores.
    """

    # Sanity checks
    if np.any([x >= 1.0 or x < 0.0 for x in percentages]):
        raise ValueError("All percentages should be in the range [0.0, 1.0)")

    if len(values) != len(u.data.indices):
        raise ValueError(
            f"The number of values, {len(values)}, should be equal to the number of data indices, {len(u.data.indices)}"
        )

    scores = {}

    # We sort in descending order if we want to remove the best values
    values.sort(reverse=remove_best)

    for pct in tqdm(percentages, disable=not progress, desc="Removal Scores"):
        n_removal = int(pct * len(u.data))
        indices = values.indices[n_removal:]
        score = u(indices)
        scores[pct] = score
    return scores
