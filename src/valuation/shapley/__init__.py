import pandas as pd

from valuation.shapley.montecarlo import (
    combinatorial_montecarlo_shapley,
    permutation_montecarlo_shapley,
    serial_truncated_montecarlo_shapley,
    truncated_montecarlo_shapley,
)
from valuation.shapley.naive import (
    combinatorial_exact_shapley,
    permutation_exact_shapley,
)
from valuation.utils import Utility

__all__ = [
    "truncated_montecarlo_shapley",
    "serial_truncated_montecarlo_shapley",
    "permutation_montecarlo_shapley",
    "combinatorial_montecarlo_shapley",
    "combinatorial_exact_shapley",
    "permutation_exact_shapley",
]


def get_shapley_values(
    u: Utility,
    iterations_per_job: int,
    num_jobs: int = 1,
    use_combinatorial=False,
    use_exact=False,
):
    """Facade for all shapley methods. By default, it uses permutation_montecarlo_shapley

    :param u: Utility object
    :param iterations_per_job: number of montecarlo iterations for each separate job
    :param num_jobs: Number of parallel jobs to run. Defaults to 1
    :param use_combinatorial: True to use the combinatorial montecarlo Shapley calculation.
        Default to False, which uses permutation montecarlo Shapley
    :param use_exact: True to use exact shapley calculation. Defaults to False.
    :return: dataframe with data keys (group names or data indices), shapley_dval
        (calculated shapley values) and dval_std, being the montecarlo standard deviation of
        shapley_dval
    """
    if num_jobs == 1:
        progress = True
    else:
        progress = False
    if use_exact:
        if use_combinatorial:
            return combinatorial_exact_shapley(u, progress)
        else:
            return permutation_exact_shapley(u, progress)
    else:
        if use_combinatorial:
            dval, dval_std = combinatorial_montecarlo_shapley(
                u, iterations_per_job, num_jobs, progress
            )
        else:
            dval, dval_std = permutation_montecarlo_shapley(
                u, iterations_per_job, num_jobs, progress
            )
    return pd.DataFrame(
        list(zip(dval.keys(), dval.values(), dval_std.values())),
        columns=["data_key", "shapley_dval", "dval_std"],
    )
