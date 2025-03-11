"""
!!! tip "New in version 0.4.0"

This package holds all routines for the computation of Least Core data values.

Please refer to [Data valuation][data-valuation-intro] for an overview.

In addition to the standard interface via
[compute_least_core_values()][pydvl.value.least_core.compute_least_core_values], because computing the
Least Core values requires the solution of a linear and a quadratic problem
*after* computing all the utility values, there is the possibility of performing
each step separately. This is useful when running multiple experiments: use
[lc_prepare_problem()][pydvl.value.least_core.naive.lc_prepare_problem] or
[mclc_prepare_problem()][pydvl.value.least_core.montecarlo.mclc_prepare_problem] to prepare a
list of problems to solve, then solve them in parallel with
[lc_solve_problems()][pydvl.value.least_core.common.lc_solve_problems].

Note that [mclc_prepare_problem()][pydvl.value.least_core.montecarlo.mclc_prepare_problem] is
parallelized itself, so preparing the problems should be done in sequence in this
case. The solution of the linear systems can then be done in parallel.

"""

from enum import Enum
from typing import Optional

from pydvl.utils.utility import Utility
from pydvl.value.least_core.montecarlo import *
from pydvl.value.least_core.naive import *
from pydvl.value.result import ValuationResult

__all__ = ["compute_least_core_values", "LeastCoreMode"]


class LeastCoreMode(Enum):
    """Available Least Core algorithms."""

    MonteCarlo = "montecarlo"
    Exact = "exact"


def compute_least_core_values(
    u: Utility,
    *,
    n_jobs: int = 1,
    n_iterations: Optional[int] = None,
    mode: LeastCoreMode = LeastCoreMode.MonteCarlo,
    non_negative_subsidy: bool = False,
    solver_options: Optional[dict] = None,
    progress: bool = False,
    **kwargs,
) -> ValuationResult:
    """Umbrella method to compute Least Core values with any of the available
    algorithms.

    See [Data valuation][data-valuation-intro] for an overview.

    The following algorithms are available. Note that the exact method can only
    work with very small datasets and is thus intended only for testing.

    - `exact`: uses the complete powerset of the training set for the constraints
      [combinatorial_exact_shapley()][pydvl.value.shapley.naive.combinatorial_exact_shapley].
    - `montecarlo`:  uses the approximate Monte Carlo Least Core algorithm.
      Implemented in [montecarlo_least_core()][pydvl.value.least_core.montecarlo.montecarlo_least_core].

    Args:
        u: Utility object with model, data, and scoring function
        n_jobs: Number of jobs to run in parallel. Only used for Monte Carlo
            Least Core.
        n_iterations: Number of subsets to sample and evaluate the utility on.
            Only used for Monte Carlo Least Core.
        mode: Algorithm to use. See
            [LeastCoreMode][pydvl.value.least_core.LeastCoreMode] for available
            options.
        non_negative_subsidy: If True, the least core subsidy $e$ is constrained
            to be non-negative.
        solver_options: Optional dictionary of options passed to the solvers.

    Returns:
        Object with the computed values.

    !!! tip "New in version 0.5.0"
    """

    if mode == LeastCoreMode.MonteCarlo:
        # TODO fix progress showing in remote case
        progress = False
        if n_iterations is None:
            raise ValueError("n_iterations cannot be None for Monte Carlo Least Core")
        return montecarlo_least_core(  # type: ignore
            u=u,
            n_iterations=n_iterations,
            n_jobs=n_jobs,
            progress=progress,
            non_negative_subsidy=non_negative_subsidy,
            solver_options=solver_options,
            **kwargs,
        )
    elif mode == LeastCoreMode.Exact:
        return exact_least_core(
            u=u,
            progress=progress,
            non_negative_subsidy=non_negative_subsidy,
            solver_options=solver_options,
        )

    raise ValueError(f"Invalid value encountered in {mode=}")
