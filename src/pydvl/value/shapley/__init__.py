"""
This package holds all routines for the computation of Shapley Data value. Users
will want to use :func:`~pydvl.value.shapley.compute_shapley_values` as single
interface to all methods defined in the modules.

Please refer to :ref:`data valuation` for an overview of Shapley Data value.
"""
from enum import Enum
from typing import Optional, cast

import numpy as np
import pandas as pd
import sklearn.neighbors as skn

from pydvl.utils import Utility
from pydvl.value.shapley.knn import knn_shapley
from pydvl.value.shapley.montecarlo import (
    combinatorial_montecarlo_shapley,
    owen_sampling_shapley,
    permutation_montecarlo_shapley,
    truncated_montecarlo_shapley,
)
from pydvl.value.shapley.naive import (
    combinatorial_exact_shapley,
    permutation_exact_shapley,
)

__all__ = ["compute_shapley_values"]


class ShapleyMode(str, Enum):
    """Supported algorithms for the computation of Shapley values."""

    CombinatorialExact = "combinatorial_exact"
    PermutationExact = "permutation_exact"
    CombinatorialMontecarlo = "combinatorial_montecarlo"
    PermutationMontecarlo = "permutation_montecarlo"
    TruncatedMontecarlo = "truncated_montecarlo"
    OwenSampling = "owen_sampling"
    KNN = "knn"


def compute_shapley_values(
    u: Utility,
    n_jobs: int = 1,
    max_iterations: Optional[int] = None,
    mode: ShapleyMode = ShapleyMode.TruncatedMontecarlo,
    **kwargs,
) -> pd.DataFrame:
    """Umbrella method to compute Shapley values with any of the available
    algorithms.

    See :ref:`data valuation` for an overview.

    The following algorithms are available. Note that the exact methods can only
    work with very small datasets and are thus intended only for testing. Some
    algorithms also accept additional arguments, please refer to the
    documentation of each particular method.

    - 'combinatorial_exact': uses the combinatorial implementation of data
      Shapley. Implemented in
      :func:`~pydvl.value.shapley.naive.combinatorial_exact_shapley`.
    - 'permutation_exact': uses the permutation-based implementation of data
      Shapley. Computation is **not parallelized**. Implemented in
      :func:`~pydvl.value.shapley.naive.permutation_exact_shapley`.
    - 'permutation_montecarlo': uses the approximate Monte Carlo implementation
      of permutation data Shapley. Implemented in
      :func:`~pydvl.value.shapley.montecarlo.permutation_montecarlo_shapley`.
    - 'combinatorial_montecarlo':  uses the approximate Monte Carlo
      implementation of combinatorial data Shapley. Implemented in
      :func:`~pydvl.value.shapley.montecarlo.combinatorial_montecarlo_shapley`.
    - 'truncated_montecarlo': default option, same as permutation_montecarlo but
      stops the computation whenever a certain accuracy is reached.
      Implemented in
      :func:`~pydvl.value.shapley.montecarlo.truncated_montecarlo_shapley`.

    Additionally, one can use model-specific methods:

    - 'knn': use only with K-Nearest neighbour models. Implemented in
      :func:`~pydvl.value.shapley.knn.knn_shapley`.

    :param u: :class:`~pydvl.utils.utility.Utility` object with model, data, and
        scoring function.
    :param max_iterations: total number of iterations, used for Monte Carlo
        methods. **Note:** power set-based methods interpret this differently
        to permutation-based methods. For the former, this is the number of
        subsets to sample for each index or process, whereas for the latter it
        is the total number of permutations to sample.
    :param n_jobs: Number of parallel jobs (available only to some methods)
    :param mode: Choose which shapley algorithm to use. See
        :class:`~pydvl.value.shapley.ShapleyMode` for a list of allowed value.

    :return: pandas DataFrame with index being group names or data indices, and
        columns: `data_value` (calculated shapley value) and `data_value_std`
        (standard deviation of `data_value` for Monte Carlo estimators)

    """
    progress: bool = kwargs.pop("progress", False)

    if mode not in list(ShapleyMode):
        raise ValueError(f"Invalid value encountered in {mode=}")

    stderr: Optional[dict]

    if mode == ShapleyMode.TruncatedMontecarlo:
        # TODO fix progress showing and maybe_progress in remote case
        progress = False
        values, stderr = truncated_montecarlo_shapley(
            u=u,
            max_iterations=max_iterations,
            n_jobs=n_jobs,
            progress=progress,
            **kwargs,
        )
    elif mode == ShapleyMode.CombinatorialMontecarlo:
        if max_iterations is None:
            raise ValueError(
                "max_iterations cannot be None for Combinatorial Montecarlo Shapley"
            )
        values, stderr = combinatorial_montecarlo_shapley(
            u, max_iterations=max_iterations, n_jobs=n_jobs, progress=progress
        )
    elif mode == ShapleyMode.PermutationMontecarlo:
        if max_iterations is None:
            raise ValueError(
                "max_iterations cannot be None for Permutation Montecarlo Shapley"
            )
        values, stderr = permutation_montecarlo_shapley(
            u, max_iterations=max_iterations, n_jobs=n_jobs, progress=progress
        )
    elif mode == ShapleyMode.CombinatorialExact:
        values = combinatorial_exact_shapley(u, n_jobs=n_jobs, progress=progress)
        stderr = None
    elif mode == ShapleyMode.PermutationExact:
        values = permutation_exact_shapley(u, progress=progress)
        stderr = None
    elif mode == ShapleyMode.OwenSampling:
        if max_iterations is None:
            raise ValueError("max_iterations cannot be None for Owen sampling Shapley")
        values, stderr = owen_sampling_shapley(
            u,
            max_iterations=max_iterations,
            max_q=100,  # FIXME!!! add argument / remove max_q
            n_jobs=n_jobs,
        )
    elif mode == ShapleyMode.KNN:
        if not isinstance(u.model, skn.KNeighborsClassifier):
            raise TypeError("KNN Shapley requires a K-Nearest Neighbours model")
        values = knn_shapley(
            u.data, cast(skn.KNeighborsClassifier, u.model), progress=progress
        )
        stderr = None
    else:
        raise ValueError(f"Invalid value encountered in {mode=}")

    df = pd.DataFrame(
        list(values.values()), index=list(values.keys()), columns=["data_value"]
    )

    if stderr is None:
        df["data_value_std"] = np.nan  # FIXME: why NaN? stddev of a constant RV is 0
    else:
        df["data_value_std"] = pd.Series(stderr)

    return df
