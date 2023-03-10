"""
.. versionadded:: 0.4.0

This package holds all routines for the computation of Least Core data values.

Please refer to :ref:`data valuation` for an overview.

In addition to the standard interface via
:func:`~pydvl.value.least_core.compute_least_core_values`, because computing the
Least Core values requires the solution of a linear and a quadratic problem
*after* computing all the utility values, there is the possibility of performing
each step separately. This is useful when running multiple experiments: use
:func:`~pydvl.value.least_core.naive.lc_prepare_problem` or
:func:`~pydvl.value.least_core.montecarlo.mclc_prepare_problem` to prepare a
list of problems to solve, then solve them in parallel with
:func:`~pydvl.value.least_core.common.lc_solve_problems`.

Note that :func:`~pydvl.value.least_core.montecarlo.mclc_prepare_problem` is
parallelized itself, so preparing the problems should be done in sequence in this
case. The solution of the linear systems can then be done in parallel.

"""
import warnings
from enum import Enum
from typing import Optional

from deprecation import DeprecatedWarning

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
    **kwargs,
) -> ValuationResult:
    """Umbrella method to compute Least Core values with any of the available
    algorithms.

    See :ref:`data valuation` for an overview.

    The following algorithms are available. Note that the exact method can only
    work with very small datasets and is thus intended only for testing.

    - ``exact``: uses the complete powerset of the training set for the constraints
      :func:`~pydvl.value.shapley.naive.combinatorial_exact_shapley`.
    - ``montecarlo``:  uses the approximate Monte Carlo Least Core algorithm.
      Implemented in :func:`~pydvl.value.least_core.montecarlo.montecarlo_least_core`.

    :param u: Utility object with model, data, and scoring function
    :param n_jobs: Number of jobs to run in parallel. Only used for Monte Carlo
        Least Core.
    :param n_iterations: Number of subsets to sample and evaluate the utility on.
        Only used for Monte Carlo Least Core.
    :param mode: Algorithm to use. See :class:`LeastCoreMode` for available
        options.
    :param non_negative_subsidy: If True, the least core subsidy $e$ is constrained
        to be non-negative.
    :param solver_options: Optional dictionary of options passed to the solvers.

    :return: ValuationResult object with the computed values.

    .. versionadded:: 0.5.0
    """
    progress: bool = kwargs.pop("progress", False)

    # TODO: remove this before releasing version 0.6.0
    if kwargs:
        warnings.warn(
            DeprecatedWarning(
                "Passing solver options as kwargs",
                deprecated_in="0.5.1",
                removed_in="0.6.0",
                details="Use solver_options instead.",
            )
        )
        if solver_options is None:
            solver_options = kwargs
        else:
            solver_options.update(kwargs)

    if mode == LeastCoreMode.MonteCarlo:
        # TODO fix progress showing and maybe_progress in remote case
        progress = False
        if n_iterations is None:
            raise ValueError("n_iterations cannot be None for Monte Carlo Least Core")
        return montecarlo_least_core(
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
