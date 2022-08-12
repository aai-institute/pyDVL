import pandas as pd

from valuation.shapley.montecarlo import (
    combinatorial_montecarlo_shapley,
    permutation_montecarlo_shapley,
    truncated_montecarlo_shapley,
)
from valuation.shapley.naive import (
    combinatorial_exact_shapley,
    permutation_exact_shapley,
)
from valuation.utils import Utility

__all__ = [
    "truncated_montecarlo_shapley",
    "permutation_montecarlo_shapley",
    "combinatorial_montecarlo_shapley",
    "combinatorial_exact_shapley",
    "permutation_exact_shapley",
]


def get_shapley_values(
    u: Utility,
    max_iterations: int,
    num_workers: int = 1,
    mode="truncated_montecarlo",
    **kwargs,
):
    """
    #TODO write better docstring for this method

    Facade for all shapley methods. By default, it uses permutation_montecarlo_shapley

    :param u: Utility object
    :param iterations_per_job: number of montecarlo iterations for each separate job
    :param num_jobs: Number of parallel jobs to run. Defaults to 1
    :param mode: Choose which shapley algorithm to use. Options are
        'truncated_montecarlo', 'exact_combinatorial', 'exact_permutation',
        ''
    :return: dataframe with data keys (group names or data indices), shapley_dval
        (calculated shapley values) and dval_std, being the montecarlo standard deviation of
        shapley_dval
    """
    # TODO fix progress showing and maybe_progress
    progress = False
    if mode == "truncated_montecarlo":
        dval, dval_std = truncated_montecarlo_shapley(
            u=u,
            max_iterations=max_iterations,
            num_workers=num_workers,
            progress=progress,
            **kwargs,
        )
    elif mode == "combinatorial_exact":
        dval, dval_std = combinatorial_exact_shapley(u, progress)
    elif mode == "permutation_exact":
        dval, dval_std = permutation_exact_shapley(u, progress)
    elif mode == "combinatorial_montecarlo":
        dval, dval_std = combinatorial_montecarlo_shapley(
            u, max_iterations, num_workers, progress
        )
    elif mode == "permutation_montecarlo":
        dval, dval_std = permutation_montecarlo_shapley(
            u, max_iterations, num_workers, progress
        )
    else:
        raise ValueError(f"Invalid value encountered in {mode=}")

    return pd.DataFrame(
        list(zip(dval.keys(), dval.values(), dval_std.values())),
        columns=["data_key", "shapley_dval", "dval_std"],
    )
