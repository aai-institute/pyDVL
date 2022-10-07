from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from pydvl.shapley.knn import compute_knn_shapley
from pydvl.shapley.montecarlo import (
    combinatorial_montecarlo_shapley,
    permutation_montecarlo_shapley,
    truncated_montecarlo_shapley,
)
from pydvl.shapley.naive import combinatorial_exact_shapley, permutation_exact_shapley
from pydvl.utils import Utility

__all__ = [
    "truncated_montecarlo_shapley",
    "permutation_montecarlo_shapley",
    "combinatorial_montecarlo_shapley",
    "combinatorial_exact_shapley",
    "permutation_exact_shapley",
    "compute_knn_shapley",
]


class ShapleyMode(str, Enum):
    """
    Different shapley modes.
    """

    ExactCombinatorial = "combinatorial_exact"
    ExactPermutation = "permutation_exact"
    CombinatorialMontecarlo = "combinatorial_montecarlo"
    PermutationMontecarlo = "permutation_montecarlo"
    TruncatedMontecarlo = "truncated_montecarlo"


def compute_shapley_values(
    u: Utility,
    n_jobs: int = 1,
    max_iterations: Optional[int] = None,
    mode: ShapleyMode = ShapleyMode.TruncatedMontecarlo,
    **kwargs,
) -> pd.DataFrame:
    """
    Given a utility, a max number of iterations and the number of jobs, it calculates
    the Shapley values. Depending on the algorithm used, it also takes additional optional arguments.

    Options for the algorithms are:
    - 'exact_combinatorial': uses combinatorial implementation of data Shapley
    - 'exact_permutation': uses permutation based implementation of data Shapley
    - 'permutation_montecarlo': uses the approximate montecarlo implementation of permutation data Shapley.
        It parallelizes computation only within the local machine
    - 'combinatorial_montecarlo':  uses montecarlo implementation of combinatorial data Shapley.
        It parallelizes computation only within the local machine
    - 'truncated_montecarlo': default option, uses permutation_montecarlo implementation but stops the
        computation whenever a certain accuracy is reached. It runs also on a cluster if the address is passed.

    :param u: Utility object with model, data, and scoring function
    :param max_iterations: total number of iterations, used for montecarlo methods
    :param n_jobs: Number of parallel jobs. Defaults to 1
    :param mode: Choose which shapley algorithm to use. Options are
        'truncated_montecarlo', 'exact_combinatorial', 'exact_permutation',
        'combinatorial_montecarlo', 'permutation_montecarlo'. Defaults to 'truncated_montecarlo'
    :return: dataframe with columns being data keys (group names or data indices), shapley_dval
        (calculated shapley values) and dval_std, being the montecarlo standard deviation of
        shapley_dval
    """
    progress: bool = kwargs.pop("progress", False)

    if mode not in list(ShapleyMode):
        raise ValueError(f"Invalid value encountered in {mode=}")

    val_std: Optional[dict]

    if mode == ShapleyMode.TruncatedMontecarlo:
        # TODO fix progress showing and maybe_progress in remote case
        progress = False
        val, val_std = truncated_montecarlo_shapley(
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
        val, val_std = combinatorial_montecarlo_shapley(
            u, max_iterations=max_iterations, n_jobs=n_jobs, progress=progress
        )
    elif mode == ShapleyMode.PermutationMontecarlo:
        if max_iterations is None:
            raise ValueError(
                "max_iterations cannot be None for Permutation Montecarlo Shapley"
            )
        val, val_std = permutation_montecarlo_shapley(
            u, max_iterations=max_iterations, n_jobs=n_jobs, progress=progress
        )
    elif mode == ShapleyMode.ExactCombinatorial:
        val = combinatorial_exact_shapley(u, progress=progress)
        val_std = None
    elif mode == ShapleyMode.ExactPermutation:
        val = permutation_exact_shapley(u, progress=progress)
        val_std = None
    else:
        raise ValueError(f"Invalid value encountered in {mode=}")

    df = pd.DataFrame(
        list(val.values()), index=list(val.keys()), columns=["data_value"]
    )

    if val_std is None:
        df["data_value_std"] = np.nan
    else:
        df["data_value_std"] = pd.Series(val_std)

    return df
